"""CPU boundaries for the staged H100 campaign; never launches a framework job."""
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchkit.campaign import plan, run_campaign
from benchkit.cli import ROOT, load_config, resolve_model
from benchkit.prepare_data import code_tasks
from benchkit.run_evidence import inventory
from benchkit.verl_backend import build_commands


class CampaignTests(unittest.TestCase):
    def test_stage_order_is_ten_then_hundred_then_thousand(self):
        cases = plan()
        expected = [(steps, model, mode) for steps in (10, 100, 1000)
                    for model in ('llama8b', 'qwen27b') for mode in ('sft', 'grpo')]
        self.assertEqual([(x['steps'], x['model'], x['mode']) for x in cases], expected)
        self.assertEqual(len({x['id'] for x in cases}), 12)
        for bad in ([], [100, 10], [10, 10], [2], [10.0]):
            with self.assertRaises(ValueError):
                plan(bad)

    def test_dry_run_does_not_create_outputs_or_resolve_models(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'must-not-exist'
            args = Namespace(steps=[10, 100, 1000], models=['llama8b', 'qwen27b'],
                             modes=['sft', 'grpo'], output=str(output), dry_run=True,
                             budget_minutes=None, case_timeout_hours=24)
            stream = StringIO()
            with patch('benchkit.campaign.resolve_model', side_effect=AssertionError('no model access')):
                with patch('pathlib.Path.mkdir', side_effect=AssertionError('no filesystem mutation')):
                    with redirect_stdout(stream):
                        self.assertEqual(run_campaign(args), 0)
            self.assertFalse(output.exists())
            result = json.loads(stream.getvalue())
            self.assertEqual(len(result['cases']), 12)
            self.assertIsNone(result['global_budget_minutes'])

    def test_final_checkpoint_cadence_follows_explicit_stage_steps(self):
        for steps in (10, 100, 1000):
            for model in ('llama8b', 'qwen27b'):
                for mode in ('sft', 'grpo'):
                    config = load_config(ROOT / 'configs' / f'h100_quick_{model}_{mode}.json',
                                         overrides=[f'steps={steps}'])
                    self.assertEqual(config['checkpoint_every_steps'], steps)
                    self.assertEqual(config['warmup_steps'], 2)
                    self.assertTrue(config['model_cache_only'])
                    self.assertEqual(config['gpus'], 8)
                    command = build_commands(dict(config, _dry_run=True), Path('/tmp/stage-contract'))[0]
                    self.assertIn(f'trainer.total_training_steps={steps}', command)
                    self.assertIn(f'trainer.save_freq={steps}', command)

    def test_cache_only_resolution_passes_local_only_and_checks_weight_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / ('a' * 40)
            snapshot.mkdir()
            (snapshot / 'config.json').write_text('{}')
            (snapshot / 'tokenizer_config.json').write_text('{}')
            (snapshot / 'model.safetensors.index.json').write_text(json.dumps({
                'weight_map': {'layer.weight': 'model-00001-of-00001.safetensors'}}))
            shard = snapshot / 'model-00001-of-00001.safetensors'
            shard.write_bytes(b'fixture checkpoint')
            downloader = Mock(return_value=str(snapshot))
            config = {'model': 'test/pinned-model', 'model_revision': 'a' * 40, 'model_cache_only': True}
            with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(snapshot_download=downloader)}):
                resolved = resolve_model(config)
                self.assertEqual(resolved['resolved_model_revision'], 'a' * 40)
                self.assertIs(downloader.call_args.kwargs['local_files_only'], True)
                self.assertEqual(downloader.call_args.kwargs['revision'], 'a' * 40)
                shard.unlink()
                with self.assertRaisesRegex(RuntimeError, 'Incomplete model cache'):
                    resolve_model(config)
                self.assertTrue(all(call.kwargs['local_files_only'] for call in downloader.call_args_list))

    def test_large_file_hash_is_explicitly_deferred_without_reading_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small = root / 'metrics.jsonl'; small.write_bytes(b'abc')
            large = root / 'model.pt'; large.write_bytes(b'weights' * 32)
            link = root / 'model-reference'; link.symlink_to(large)
            original_open = Path.open
            def guarded_open(path, *args, **kwargs):
                if path == large:
                    raise AssertionError('Deferred checkpoint must not be read')
                return original_open(path, *args, **kwargs)
            with patch.object(Path, 'open', guarded_open):
                result = inventory(root, max_hash_bytes=4)
            by_path = {row['path']: row for row in result['files']}
            self.assertEqual(by_path['metrics.jsonl']['sha256'], hashlib.sha256(b'abc').hexdigest())
            self.assertIsNone(by_path['model.pt']['sha256'])
            self.assertEqual(by_path['model.pt']['hash_status'], 'deferred_size_limit')
            self.assertEqual(by_path['model-reference']['kind'], 'symlink')
            self.assertEqual(result['deferred_hash_count'], 1)
            self.assertEqual(large.read_bytes(), b'weights' * 32)

    def test_rl_duration_stages_share_a_deterministic_task_prefix(self):
        short = code_tasks(80, seed=42)
        longer = code_tasks(800, seed=42)
        self.assertEqual(short, longer[:80])
        self.assertEqual(len({r['extra_info']['task_id'] for r in longer}), 800)
        # IDs identify rollout groups; they are not a claim of unique programs.
        self.assertEqual({r['ability'] for r in longer}, {'code_output_prediction'})


if __name__ == '__main__':
    unittest.main()
