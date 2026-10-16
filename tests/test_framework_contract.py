"""CPU checks for scientific validity and the native command boundary."""

from pathlib import Path
import json
import tempfile
import unittest

from benchkit.prepare_data import code_tasks, session_rows
from benchkit.reward import audit_rollouts, compute_score, extract_answer
from benchkit.verl_backend import build_commands
from benchkit.verl_entry import normalize_nvidia_visibility, synchronized_interval
from benchkit.model_assets import prepare_model_assets
from benchkit.artifacts import ArtifactRecorder, file_sha256


class FrameworkContractTests(unittest.TestCase):
    def test_maximal_capture_and_checkpoint_defaults(self):
        cfg = dict(mode='grpo', model='test/model', gpus=2, global_batch_size=2)
        command = build_commands(cfg, Path('/tmp/capture-contract'))[0]
        self.assertIn(f'BENCH_ARTIFACT_ROOT={Path("/tmp/capture-contract/artifacts").resolve()}', command)
        self.assertIn('trainer.save_freq=1', command)
        self.assertIn('trainer.max_actor_ckpt_to_keep=null', command)
        self.assertIn('actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra"]', command)
        with self.assertRaises(ValueError):
            build_commands({**cfg, 'checkpoint_keep_all': False}, Path('/tmp/capture-contract'))
        with self.assertRaises(ValueError):
            build_commands({**cfg, 'steps': 10, 'checkpoint_policy': 'final_only'}, Path('/tmp/capture-contract'))
        final = build_commands({**cfg, 'steps': 10, 'checkpoint_policy': 'final_only',
                                'checkpoint_every_steps': 10}, Path('/tmp/capture-contract'))[0]
        self.assertIn('trainer.save_freq=10', final)

    def test_artifact_tensor_roundtrip_and_hash(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch roundtrip runs in the GPU framework environment')
        with tempfile.TemporaryDirectory() as directory:
            recorder = ArtifactRecorder(directory)
            live = torch.tensor([[1, 2, 3]])
            record = recorder.capture('test_input', {'input_ids': live, 'mask': live > 1}, step=1)
            live.zero_()
            path = Path(directory) / record['payload']
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved['input_ids'].tolist(), [[1, 2, 3]])
            self.assertEqual(record['payload_sha256'], file_sha256(path))
            self.assertEqual(record['structure']['input_ids']['shape'], [1, 3])
            self.assertEqual(len(list(Path(directory).glob('manifest-*.jsonl'))), 1)

    def test_template_bundle_preserves_weights_and_original_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / 'original'
            source.mkdir()
            weights = source / 'model.safetensors'
            weights.write_bytes(b'unchanged checkpoint bytes')
            tokenizer = source / 'tokenizer_config.json'
            tokenizer.write_text('{"model_max_length": 2048}')
            (source / 'chat_template.jinja').write_text('old template')
            before = tokenizer.read_bytes()
            template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            cfg = {'model': str(source), 'requested_model': 'test/model',
                   'resolved_model_revision': 'exact-commit', 'chat_template': template}
            result = prepare_model_assets(cfg, base / 'run')
            bundle = Path(result['model'])
            self.assertEqual(tokenizer.read_bytes(), before)
            self.assertTrue((bundle / 'model.safetensors').is_symlink())
            self.assertEqual((bundle / 'model.safetensors').resolve(), weights.resolve())
            self.assertEqual(json.loads((bundle / 'tokenizer_config.json').read_text())['chat_template'], template)
            self.assertEqual((bundle / 'chat_template.jinja').read_text(), template)
            self.assertEqual(result['resolved_model_revision'], 'exact-commit')
            self.assertNotEqual(result['derived_tokenizer_config_sha256'], json.loads((base / 'run/model_assets_manifest.json').read_text())['original_tokenizer_config_sha256'])
    def test_training_timer_fences_before_and_after_device_work(self):
        events = []
        ticks = iter([10.0, 11.25])
        def fence():
            events.append('fence')
        def clock():
            events.append('clock')
            return next(ticks)
        def train():
            events.append('optimizer update')
            return 'result'
        result, elapsed = synchronized_interval(train, fence, clock)
        self.assertEqual((result, elapsed), ('result', 1.25))
        self.assertEqual(events, ['fence', 'clock', 'optimizer update', 'fence', 'clock'])
    def test_slurm_visibility_cleanup_preserves_nvidia_allocation(self):
        env = {'CUDA_VISIBLE_DEVICES': '2,3', 'ROCR_VISIBLE_DEVICES': '2,3', 'HIP_VISIBLE_DEVICES': '2,3'}
        normalize_nvidia_visibility(env, is_cuda=True)
        self.assertEqual(env, {'CUDA_VISIBLE_DEVICES': '2,3'})
        amd = {'ROCR_VISIBLE_DEVICES': '0'}
        self.assertEqual(normalize_nvidia_visibility(amd, is_cuda=False), [])
        self.assertIn('ROCR_VISIBLE_DEVICES', amd)
    def test_rewards_depend_on_generated_final_answer(self):
        row = code_tasks(1, 42)[0]
        expected = row['reward_model']['ground_truth']
        args = dict(data_source=row['data_source'], ground_truth=expected, extra_info=row['extra_info'])
        self.assertEqual(compute_score(solution_str=f'<answer>{expected}</answer>', **args)['score'], 1.0)
        self.assertEqual(compute_score(solution_str='<answer>999999</answer>', **args)['score'], 0.1)
        self.assertEqual(compute_score(solution_str='import os; os.system("ignored")', **args)['score'], 0)
        self.assertIsNone(extract_answer('<answer>' + '1' * 1000 + '</answer>'))
        self.assertEqual(extract_answer('<answer>4</answer> then \\boxed{-2}'), -2)

    def test_validation_tasks_are_disjoint_and_deterministic(self):
        self.assertEqual(code_tasks(10, 8), code_tasks(10, 8))
        train = {r['extra_info']['task_id'] for r in code_tasks(10, 8)}
        valid = {r['extra_info']['task_id'] for r in code_tasks(10, 9)}
        self.assertFalse(train & valid)

    def test_grpo_uses_framework_rollout_and_custom_reward(self):
        cfg = dict(mode='grpo', model='Qwen/Qwen2.5-0.5B-Instruct', gpus=2,
                   global_batch_size=2, micro_batch_size=1, rollout_n=2)
        command = build_commands(cfg, Path('/tmp/contract-run'))[0]
        self.assertIn('benchkit.verl_entry', command)
        self.assertIn('algorithm.adv_estimator="grpo"', command)
        self.assertIn('actor_rollout_ref.rollout.n=2', command)
        self.assertTrue(any(arg.startswith('reward.custom_reward_function.path=') for arg in command))
        self.assertFalse(any('random' in arg.lower() for arg in command))
        with self.assertRaises(ValueError):
            build_commands({**cfg, 'rollout_n': 1}, Path('/tmp/contract-run'))
        with self.assertRaises(ValueError):
            build_commands({**cfg, 'framework_lane': 'a6000-dev', 'enable_prefix_caching': False}, Path('/tmp/contract-run'))

    def test_bad_batch_or_unresolved_revision_fails_before_gpu(self):
        cfg = dict(mode='sft', model='meta-llama/Llama-3.1-8B', gpus=2, global_batch_size=3)
        with self.assertRaises(ValueError):
            build_commands(cfg, Path('/tmp/contract-run'))
        with self.assertRaises(ValueError):
            build_commands({**cfg, 'global_batch_size': 2, 'model_revision': 'abcd'}, Path('/tmp/contract-run'))
        with self.assertRaises(ValueError):
            build_commands({**cfg, 'global_batch_size': 2, 'extra_overrides': ['trainer.n_gpus_per_node=8']}, Path('/tmp/contract-run'))
        with self.assertRaises(ValueError):
            build_commands({**cfg, 'global_batch_size': 2, 'dtype': 'float32'}, Path('/tmp/contract-run'))

    def test_rejects_data_without_supervised_response(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'empty.jsonl'
            source.write_text('{"messages":[{"role":"user","content":"hello"}]}\n')
            with self.assertRaises(ValueError):
                session_rows(source)

    def test_constant_rewards_do_not_claim_learning_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            rollout = Path(directory) / 'rollouts'
            rollout.mkdir()
            rows = [{'step': 2, 'task_id': 'same', 'score': 1.0} for _ in range(4)]
            (rollout / '2.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            audit = audit_rollouts(Path(directory), warmup_steps=1, rollout_n=4)
            self.assertEqual(audit['reward/group_nonzero_std_fraction'], 0)
            self.assertFalse(audit['reward/learning_signal_detected'])
            rows[-1]['score'] = 0
            (rollout / '2.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            self.assertTrue(audit_rollouts(Path(directory), rollout_n=4)['reward/learning_signal_detected'])


if __name__ == '__main__':
    unittest.main()
