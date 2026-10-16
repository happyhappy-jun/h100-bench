"""CLI evidence contracts using mocks only, without GPU or model dependencies."""

import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from benchkit import cli
from benchkit.verl_backend import build_commands


def config(**changes):
    return {
        "name": "fixture-only", "mode": "sft", "model": "fixture/pretrained-model",
        "model_revision": "a" * 40, "gpus": 2, "nodes": 1,
        "global_batch_size": 4, "micro_batch_size": 1, "seq_length": 128,
        "steps": 3, "warmup_steps": 1, "seed": 42,
        "dtype": "bfloat16", "strategy": "fsdp2", "timeout_s": 10,
        "input_data": "data/sessions.jsonl", "extra_overrides": [],
        "expected_gpu": "H100", "framework_lane": "h100", **changes,
    }


class CliEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fixture-only-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_pinned_model_dry_run_downloads_nothing_and_writes_nothing(self):
        output = self.root / "attempt"
        stdout = io.StringIO()
        download = Mock(side_effect=AssertionError("A dry run cannot download weights"))
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}), \
             patch.object(cli, "doctor", side_effect=AssertionError("A dry run cannot probe the GPU")), \
             patch.object(cli, "resolve_model", side_effect=AssertionError("A dry run cannot resolve weights")), \
             redirect_stdout(stdout):
            result = cli.run(config(), output, dry_run=True)
        self.assertEqual(result, 0)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.iterdir()), [])
        download.assert_not_called()
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["config"]["model_revision"], "a" * 40)
        self.assertEqual(plan["config"]["model"], "fixture/pretrained-model")
        self.assertTrue(plan["commands"])

    def test_doctor_checks_cuda_visible_devices_not_host_inventory(self):
        dataset = self.root / "dataset.jsonl"
        dataset.write_text("")
        environment = {"gpu": {"stdout": "NVIDIA H100, fixture-uuid, 81920 MiB, fixture-driver"}}
        framework = {"available": True, "missing": [], "pinned_revision_match": True,
                     "source_clean": True, "lane": "h100"}
        for devices in (["NVIDIA H100", "NVIDIA RTX A6000"], ["NVIDIA RTX A6000", "NVIDIA RTX A6000"]):
            with self.subTest(devices=devices):
                probe = {"returncode": 0, "stdout": json.dumps({"available": True, "count": 2, "devices": devices})}
                with patch.object(cli, "environment", return_value=environment), \
                     patch.object(cli, "capture", return_value=probe), \
                     patch("benchkit.verl_backend.validate_framework", return_value=framework):
                    result = cli.doctor(config(input_data=str(dataset)))
                self.assertIn("allocated_gpu_mismatch=H100", result["problems"])

    def test_local_snapshot_name_does_not_prove_weight_revision(self):
        snapshot = self.root / "cache" / "snapshots" / ("b" * 40)
        snapshot.mkdir(parents=True)
        payload = '{"fixture_only": true}\n'
        (snapshot / "config.json").write_text(payload)
        download = Mock(side_effect=AssertionError("A local checkpoint cannot trigger a download"))
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}):
            resolved = cli.resolve_model(config(model=str(snapshot)))
        self.assertIsNone(resolved["resolved_model_revision"])
        self.assertEqual(resolved["requested_model"], str(snapshot))
        self.assertEqual(resolved["local_model_metadata_sha256"]["config.json"], hashlib.sha256(payload.encode()).hexdigest())
        download.assert_not_called()

    def test_native_overrides_cannot_change_recorded_measurement_identity(self):
        overridden = config(extra_overrides=["trainer.n_gpus_per_node=1"])
        path = self.root / "config.json"
        path.write_text(json.dumps(overridden))
        with self.assertRaises(ValueError):
            cli.load_config(path)
        with self.assertRaises(ValueError):
            build_commands(overridden, self.root / "output")

    def test_nonzero_framework_exit_preserves_failed_attempt_with_complete_metrics(self):
        output = self.root / "failed-attempt"
        selected = config()
        rows = [{"step": step, "timing_s/train_step": 1.0, "train/global_tokens": 100,
                 "train/loss": .5, "train/grad_norm": 1.0} for step in range(1, 4)]
        telemetry = Mock()
        telemetry.wait.return_value = 0
        worker = Mock()
        worker.wait.return_value = 7
        with ExitStack() as stack:
            stack.enter_context(patch.object(cli, "doctor", return_value={
                "environment": {}, "framework": {}, "problems": [],
                "cuda": {"returncode": 0, "stdout": json.dumps({"uuids": ["fixture-gpu-0", "fixture-gpu-1"]})},
            }))
            stack.enter_context(patch.object(cli, "resolve_model", return_value=selected))
            stack.enter_context(patch("benchkit.prepare_data.prepare_data", return_value={}))
            stack.enter_context(patch("benchkit.verl_backend.build_commands", return_value=[["fixture-trainer"]]))
            stack.enter_context(patch.object(cli.subprocess, "Popen", side_effect=[telemetry, worker]))
            stack.enter_context(patch.object(cli, "read_metrics", return_value=rows))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            result = cli.run(selected, output)
        self.assertEqual(result, 1)
        summary = json.loads((output / "summary.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(summary["training_steps"], selected["steps"])
        self.assertIn("framework_exit_code=7", summary["problems"])
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(manifest["state"], "failed")
        self.assertNotEqual(manifest["exit_code"], 0)
        self.assertTrue((output / "framework-0.log").is_file())


if __name__ == "__main__":
    unittest.main()
