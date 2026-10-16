"""Lossless CPU snapshots of exposed native trainer data, with explicit scope.

These files contain pickle-backed torch payloads. Only load trusted run files.
Hooks do not capture every internal activation or transient CUDA kernel buffer.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from collections.abc import Mapping
from functools import wraps
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def describe(value):
    """JSON metadata only; payload preserves values in torch serialization."""
    import torch
    if isinstance(value, torch.Tensor):
        if value.is_nested:
            return {'type': 'tensor', 'nested': True, 'dtype': str(value.dtype), 'layout': str(value.layout)}
        return {'type': 'tensor', 'shape': list(value.shape), 'dtype': str(value.dtype), 'layout': str(value.layout)}
    if isinstance(value, Mapping):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {'type': type(value).__name__, 'length': len(value), 'items': [describe(item) for item in value]}
    return {'type': type(value).__name__}


def cpu_snapshot(value):
    import torch
    if isinstance(value, torch.Tensor):
        # CPU copy must be independent of the live batch, which is mutated later.
        return value.detach().to(device='cpu', copy=True)
    if hasattr(value, 'batch') and hasattr(value, 'non_tensor_batch') and hasattr(value, 'meta_info'):
        return {'type': 'DataProto', 'batch': cpu_snapshot(value.batch),
                'non_tensor_batch': cpu_snapshot(value.non_tensor_batch), 'meta_info': cpu_snapshot(value.meta_info)}
    if hasattr(value, 'to_dict') and type(value).__module__.startswith('tensordict'):
        return {'type': type(value).__name__, 'data': cpu_snapshot(value.to_dict())}
    if isinstance(value, Mapping):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if type(value).__module__.startswith('numpy') and hasattr(value, 'copy'):
        return value.copy()
    return value


class ArtifactRecorder:
    def __init__(self, root, every=1):
        self.root = Path(root)
        self.every = int(every)
        if self.every < 1:
            raise ValueError('capture_every_step must be positive')
        self.process = f'{socket.gethostname()}-pid{os.getpid()}-rank{os.environ.get("RANK", "driver")}'
        self.folder = self.root / self.process
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / f'manifest-{self.process}.jsonl'
        self.count = 0
        self.total_seconds = 0.0
        self.total_bytes = 0
        self.stages = set()

    def capture(self, stage, value, step=0):
        if step and step != 1 and step % self.every:
            return None
        import torch
        started = time.perf_counter()
        self.count += 1
        stem = f'{stage}-step{int(step):06d}-{self.count:06d}'
        payload = self.folder / f'{stem}.pt'
        copied = cpu_snapshot(value)
        torch.save(copied, payload)
        record = {'schema_version': 1, 'stage': stage, 'step': int(step), 'process': self.process,
                  'rank': os.environ.get('RANK'), 'payload': str(payload.relative_to(self.root)),
                  'payload_sha256': file_sha256(payload), 'payload_bytes': payload.stat().st_size,
                  'structure': describe(copied),
                  'scope': 'exposed native trainer data; no internal activation or kernel-buffer capture'}
        record['elapsed_s'] = time.perf_counter() - started
        metadata = payload.with_suffix('.json')
        record['metadata'] = str(metadata.relative_to(self.root))
        metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        with self.manifest.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.total_seconds += record['elapsed_s']
        self.total_bytes += record['payload_bytes']
        self.stages.add(stage)
        return record

    def config(self, config, label='native_resolved_config'):
        from omegaconf import OmegaConf
        data = OmegaConf.to_container(config, resolve=True)
        (self.folder / f'{label}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        return self.capture(label, data)


_RECORDER = None
_ACTIVE_STEP = 0


def recorder():
    global _RECORDER
    root = os.environ.get('BENCH_ARTIFACT_ROOT')
    if not root:
        return None
    if _RECORDER is None:
        _RECORDER = ArtifactRecorder(root, os.environ.get('BENCH_CAPTURE_EVERY_STEP', '1'))
    return _RECORDER


def capture(stage, value, step=0):
    current = recorder()
    return current.capture(stage, value, step) if current else None


def _resolve_batch(value):
    if hasattr(value, 'partition_id') and hasattr(value, 'keys') and not callable(value.keys):
        import transfer_queue as tq
        return {'type': 'KVBatchMeta', 'keys': list(value.keys), 'partition_id': value.partition_id,
                'tags': getattr(value, 'tags', None), 'extra_info': getattr(value, 'extra_info', None),
                'fields': tq.kv_batch_get(keys=value.keys, partition_id=value.partition_id)}
    return value


def _wrap_method(cls, method, before=None, after=None):
    original = getattr(cls, method)  # A missing pinned API is fatal.
    @wraps(original)
    def wrapped(self, *args, **kwargs):
        global _ACTIVE_STEP
        _ACTIVE_STEP = int(getattr(self, 'global_steps', 0))
        batch = args[0] if args else kwargs.get('batch')
        if before:
            capture(before, _resolve_batch(batch), _ACTIVE_STEP)
        result = original(self, *args, **kwargs)
        if after:
            capture(after, _resolve_batch(result), _ACTIVE_STEP)
        return result
    setattr(cls, method, wrapped)


def install_rl_hooks():
    """Install only in the native trainer-driver actor, never in vLLM workers."""
    import verl.trainer.main_ppo as main_ppo
    current = recorder()
    if current is None:
        raise RuntimeError('Artifact capture was requested but BENCH_ARTIFACT_ROOT is missing')
    if hasattr(main_ppo, 'TaskRunnerV1'):
        from verl.trainer.ppo.v1.trainer_base import PPOTrainer as cls
        pairs = [('_submit_batch_to_rollout', 'rl_prompt_batch', None),
                 ('_balance_batch', 'rl_generated_reward_batch', None),
                 ('_compute_old_log_prob', None, 'rl_old_logprob_batch'),
                 ('_compute_ref_log_prob', None, 'rl_reference_batch'),
                 ('_compute_advantage', 'rl_reward_logprob_batch', 'rl_advantage_batch'),
                 ('_update_actor', 'rl_update_input', 'rl_update_output')]
        interface = 'verl_v091_transfer_queue'
    else:
        from verl.trainer.ppo import ray_trainer
        cls = ray_trainer.RayPPOTrainer
        pairs = [('_get_gen_batch', 'rl_prompt_batch', None),
                 ('_compute_old_log_prob', 'rl_generated_reward_batch', 'rl_old_logprob_batch'),
                 ('_compute_ref_log_prob', None, 'rl_reference_batch'),
                 ('_update_actor', 'rl_update_input', 'rl_update_output')]
        original_adv = ray_trainer.compute_advantage
        @wraps(original_adv)
        def advantage(*args, **kwargs):
            batch = args[0] if args else kwargs['data']
            capture('rl_reward_logprob_batch', batch, _ACTIVE_STEP)
            result = original_adv(*args, **kwargs)
            capture('rl_advantage_batch', result, _ACTIVE_STEP)
            return result
        ray_trainer.compute_advantage = advantage
        interface = 'verl_v080_dataproto'
    for method, before, after in pairs:
        _wrap_method(cls, method, before, after)
    original_fit = cls.fit
    @wraps(original_fit)
    def fit(self, *args, **kwargs):
        current.config(self.config)
        capture('capture_hook_contract', {'interface': interface, 'hooks': pairs,
                'gpu_validation': 'must be established by this run', 'capture_version': 1})
        result = original_fit(self, *args, **kwargs)
        expected = {'rl_prompt_batch', 'rl_generated_reward_batch', 'rl_advantage_batch', 'rl_update_output'}
        if int(self.config.trainer.total_training_steps) >= current.every and not expected <= current.stages:
            raise RuntimeError(f'Required native captures missing: {sorted(expected - current.stages)}')
        current.config(self.config, 'native_final_config')
        return result
    cls.fit = fit


class CapturingTaskRunner:
    """Composition adapter preserving the native TaskRunner implementation."""
    def run(self, config):
        import verl.trainer.main_ppo as main_ppo
        install_rl_hooks()
        if hasattr(main_ppo, 'TaskRunnerV1'):
            if not config.trainer.use_v1:
                raise ValueError('H100 capture requires the pinned native V1 trainer')
            # Pinned Ray ActorClass retains its Python implementation here.
            metadata = getattr(main_ppo.TaskRunnerV1, '__ray_metadata__', None)
            implementation = getattr(metadata, 'modified_class', None)
            if implementation is None:
                raise RuntimeError('Pinned Ray TaskRunnerV1 Python implementation is unavailable')
            native = implementation()
        else:
            native = main_ppo.TaskRunner()
        return native.run(config)


def launch_captured_grpo():
    import ray
    import verl.trainer.main_ppo as main_ppo
    original = main_ppo.run_ppo
    @wraps(original)
    def run_ppo(config, task_runner_class=None):
        from omegaconf import OmegaConf, open_dict
        with open_dict(config.ray_kwargs.ray_init):
            runtime_env = config.ray_kwargs.ray_init.get('runtime_env', {})
            config.ray_kwargs.ray_init.runtime_env = OmegaConf.merge(runtime_env, {'env_vars': {
                'BENCH_ARTIFACT_ROOT': os.environ['BENCH_ARTIFACT_ROOT'],
                'BENCH_CAPTURE_EVERY_STEP': os.environ.get('BENCH_CAPTURE_EVERY_STEP', '1'),
            }})
        return original(config, task_runner_class=ray.remote(num_cpus=1)(CapturingTaskRunner))
    main_ppo.run_ppo = run_ppo
    main_ppo.main()
