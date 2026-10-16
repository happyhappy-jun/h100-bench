"""Native trainer entrypoints with a narrow SFT timing instrumentation hook.

The SFT hook brackets the engine update with explicit CUDA fences. It changes
neither training data nor loss/optimizer behavior. RL uses native stage timers.
"""

from __future__ import annotations

import runpy
import sys
import os
import time
from functools import wraps


def synchronized_interval(operation, synchronize, clock=time.perf_counter):
    """Measure completed device work, excluding already queued prior work."""
    synchronize()
    started = clock()
    output = operation()
    synchronize()
    return output, clock() - started


def normalize_nvidia_visibility(environment, *, is_cuda: bool) -> list[str]:
    """Remove only conflicting AMD aliases on a verified NVIDIA allocation.

Some Slurm sites export all three vendor masks. verl rejects that ambiguity.
Keep CUDA_VISIBLE_DEVICES byte-for-byte so GPU allocation isolation survives.
    """
    removed = []
    if is_cuda and "CUDA_VISIBLE_DEVICES" in environment:
        for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
            if name in environment:
                del environment[name]
                removed.append(name)
    return removed


def instrument_sft() -> None:
    import torch
    from verl.utils import tensordict_utils as tu
    from verl.workers.engine_workers import TrainingWorker
    from verl.workers.engine.base import BaseEngine

    original_train_batch = BaseEngine.train_batch

    @wraps(original_train_batch)
    def synchronized_train_batch(self, *args, **kwargs):
        from benchkit.artifacts import capture, recorder
        self._bench_capture_step = getattr(self, "_bench_capture_step", 0) + 1
        step = self._bench_capture_step
        current = recorder()
        capture_start = current.total_seconds if current else 0.0
        data = args[0] if args else kwargs.get("data")
        capture("sft_engine_input", data, step)
        output, elapsed = synchronized_interval(
            lambda: original_train_batch(self, *args, **kwargs), torch.cuda.synchronize,
        )
        self._bench_synchronized_train_seconds = elapsed
        capture("sft_engine_output", output, step)
        if current and isinstance(output, dict) and "metrics" in output:
            output["metrics"]["bench/artifact_capture_s"] = current.total_seconds - capture_start
            output["metrics"]["bench/artifact_capture_version"] = 1
        return output

    BaseEngine.train_batch = synchronized_train_batch

    original = TrainingWorker._postprocess_output

    def with_timing(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
        result = original(
            self, output, global_token_num=global_token_num,
            delta_time=delta_time, forward_only=forward_only, images_seqlens=images_seqlens,
        )
        if not forward_only:
            # FSDP runs this on every data-parallel rank. The slowest rank
            # determines throughput; also retain global peak memory across ranks.
            metrics = tu.get(result, "metrics")
            required_memory = ("perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb")
            if any(key not in metrics or metrics[key] is None for key in required_memory):
                raise RuntimeError("Pinned verl did not emit required memory metrics; refusing to substitute zero")
            if not hasattr(self.engine, "_bench_synchronized_train_seconds"):
                raise RuntimeError("SFT engine bypassed the synchronized training timer")
            values = torch.tensor([
                float(self.engine._bench_synchronized_train_seconds),
                float(metrics["perf/max_memory_allocated_gb"]),
                float(metrics["perf/max_memory_reserved_gb"]),
                float(delta_time),
            ], device=torch.cuda.current_device(), dtype=torch.float64)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX, group=self.engine.get_data_parallel_group())
            elapsed, allocated, reserved, native_elapsed = values.tolist()
            metrics["timing_s/train_step"] = elapsed
            metrics["timing_s/train_step_native"] = native_elapsed
            metrics["perf/max_memory_allocated_gb"] = allocated
            metrics["perf/max_memory_reserved_gb"] = reserved
            metrics["bench/timing_instrumented"] = 1
            metrics["bench/timing_instrumentation_version"] = 2
            tu.assign_non_tensor(result, metrics=metrics)
        return result

    TrainingWorker._postprocess_output = with_timing

    if os.environ.get("BENCH_ARTIFACT_ROOT"):
        from verl.trainer.sft_trainer import SFTTrainer
        from benchkit.artifacts import recorder
        original_fit = SFTTrainer.fit
        @wraps(original_fit)
        def captured_fit(self, *args, **kwargs):
            recorder().config(self.config)
            result = original_fit(self, *args, **kwargs)
            if not {"sft_engine_input", "sft_engine_output"} <= recorder().stages:
                raise RuntimeError("Required SFT engine artifact captures are missing")
            recorder().config(self.config, "native_final_config")
            return result
        SFTTrainer.fit = captured_fit


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"sft", "grpo"}:
        raise SystemExit("usage: python -m benchkit.verl_entry {sft|grpo} [native Hydra overrides...]")
    mode = sys.argv.pop(1)
    import torch
    removed = normalize_nvidia_visibility(
        os.environ, is_cuda=torch.version.cuda is not None and torch.version.hip is None,
    )
    if removed:
        print(f"NVIDIA Slurm visibility normalization: removed {','.join(removed)}; kept CUDA_VISIBLE_DEVICES", flush=True)
    if mode == "sft":
        instrument_sft()
        # Import once: runpy would create a second uninstrumented SFTTrainer.
        from verl.trainer.sft_trainer import main as native_main
        native_main()
        return
    if os.environ.get("BENCH_ARTIFACT_ROOT"):
        from benchkit.artifacts import launch_captured_grpo
        launch_captured_grpo()
        return
    module = "verl.trainer.sft_trainer" if mode == "sft" else "verl.trainer.main_ppo"
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
