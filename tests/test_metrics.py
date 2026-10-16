import json
import tempfile
import unittest
from pathlib import Path

from benchkit.cli import load_config, nvidia_device_ids
from benchkit.metrics import read_metrics, summarize, gpu_telemetry


class MetricsTests(unittest.TestCase):
    def test_nvidia_telemetry_uses_allocated_uuid_formats(self):
        self.assertEqual(nvidia_device_ids(["abc", "GPU-def", "MIG-ghi"]),
                         ["GPU-abc", "GPU-def", "MIG-ghi"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gpu.csv"
            path.write_text("timestamp, uuid, utilization.gpu [%], memory.used [MiB], power.draw [W]\n"
                            "2026/10/16, GPU-abc, 45 %, 1024 MiB, 120 W\n"
                            "2026/10/16, GPU-abc, 80 %, 2048 MiB, [N/A]\n")
            s = gpu_telemetry(path)
            self.assertEqual(s["devices"]["GPU-abc"]["memory_used_mib"]["max"], 2048)
            self.assertEqual(s["devices"]["GPU-abc"]["power_w"]["n"], 1)

    def config(self, **kw):
        return dict(name="test_fixture", mode="sft", model="fixture", gpus=2,
                    nodes=1, steps=3, warmup_steps=1, **kw)

    def rows(self):
        return [{"step": i, "timing_s/train_step": t, "train/global_tokens": n,
                 "train/loss": 1.0, "perf/max_memory_allocated_gb": 5.0}
                for i, t, n in [(1, 100, 100), (2, 2, 200), (3, 4, 800)]]

    def test_weighted_token_throughput_and_warmup(self):
        s = summarize(self.config(), self.rows(), 0, 110)
        self.assertEqual(s["status"], "passed")
        self.assertAlmostEqual(s["tokens_per_s"], 1000 / 6)
        self.assertAlmostEqual(s["tokens_per_gpu_s"], 1000 / 12)
        self.assertEqual(s["measured_steps"], 2)

    def test_exit_failure_cannot_be_hidden_by_metrics(self):
        self.assertEqual(summarize(self.config(), self.rows(), 1, 110)["status"], "failed")

    def test_rl_native_memory_and_timers_do_not_confuse_counts_with_seconds(self):
        config = self.config(); config["mode"] = "grpo"
        rows = [{"step": step, "actor/pg_loss": 0.01, "timing_s/step": 2,
                 "perf/total_num_tokens": 100, "actor/perf/max_memory_allocated_gb": 7.5,
                 "timing_s/gen": 1, "timing_s/agent_loop/slowest/prompt_length": 200,
                 "timing_s/agent_loop/num_preempted/mean": -1} for step in (1, 2, 3)]
        result = summarize(config, rows, 0, 10)
        self.assertEqual(result["peak_memory_gib"], 7.5)
        self.assertEqual(set(result["stages"]), {"timing_s/step", "timing_s/gen"})

    def test_missing_and_nan_tokens_fail(self):
        for value in (None, float("nan"), 0):
            rows = self.rows()
            rows[1]["train/global_tokens"] = value
            s = summarize(self.config(), rows, 0, 110)
            self.assertEqual(s["status"], "failed")
            self.assertTrue(s["problems"])

    def test_partial_run_and_no_measured_steps_fail(self):
        s = summarize(self.config(), self.rows()[:1], 0, 110)
        self.assertEqual(s["status"], "failed")
        self.assertIsNone(s["tokens_per_s"])

    def test_duplicate_step_merges_and_validation_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "metrics.jsonl"
            path.write_text('\n'.join(json.dumps(r) for r in [
                {"step": 0, "data": {"val/loss": 1}},
                {"step": 1, "data": {"train/loss": 2}},
                {"step": 1, "data": {"timing_s/train_step": 3, "train/global_tokens": 9}},
            ]))
            rows = read_metrics(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["train/global_tokens"], 9)

    def test_config_requires_fixed_batch_divisibility(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            path.write_text(json.dumps(dict(name="test", model="fixture", mode="sft", gpus=4,
                nodes=1, global_batch_size=6, micro_batch_size=1, steps=3, warmup_steps=1, seq_length=128)))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
