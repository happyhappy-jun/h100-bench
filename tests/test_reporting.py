"""Synthetic fixtures for report semantics; these are not measured results."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchkit.reporting import generate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_run(root, name, gpus=2, rate=100, hardware="NVIDIA H100 80GB HBM3", **overrides):
    """Write intentionally artificial records into a temporary test directory."""
    directory = root / name
    config = {
        "name": name, "model": "fixture/model", "model_revision": "a" * 40,
        "mode": "sft", "gpus": gpus, "nodes": 1, "global_batch_size": 8,
        "micro_batch_size": 1, "seq_length": 128, "dtype": "bfloat16",
        "steps": 3, "warmup_steps": 1, "seed": 42, "evidence_class": "fixture_only",
        **overrides,
    }
    summary = {
        "schema_version": 1, "name": name, "status": "passed", "problems": [],
        "model": config["model"], "model_revision": config["model_revision"],
        "mode": config["mode"], "gpus": gpus, "nodes": 1, "config": config,
        "measured_steps": 2, "warmup_steps": 1, "measured_time_s": 10,
        "measured_tokens": rate * 10, "tokens_per_s": rate,
        # Deliberately inconsistent: the reporter must divide the all-rank
        # measured rate itself, not multiply by GPU count or trust this field.
        "tokens_per_gpu_s": 99999,
        "token_scope": "unpadded input tokens (including masked prompt tokens)",
        "step_time_s": {"mean": 5, "median": 5, "p95": 6},
        "stages": {"timing_s/train_step": {"n": 2, "mean": 5, "median": 5, "p95": 6, "min": 4, "max": 6}},
    }
    write_json(directory / "summary.json", summary)
    write_json(directory / "config.json", config)
    write_json(directory / "manifest.json", {"state": "passed", "exit_code": 0})
    write_json(directory / "data" / "manifest.json", {"source_sha256": "d" * 64, "sft_rows": 24})
    write_json(directory / "environment.json", {
        "environment": {
            "gpu": {"stdout": f"{hardware}, GPU-fixture, 81920 MiB, fixture-driver"},
            "versions": {"torch": "fixture-torch", "transformers": "fixture-transformers", "verl": "fixture-verl"},
            "git": {"returncode": 0, "stdout": "c" * 40},
            "git_dirty": {"returncode": 0, "stdout": ""},
        },
        "framework": {"verl_commit": "b" * 40, "source_clean": True},
    })
    return directory


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.output = self.root / "report"
        self.plotter = patch("benchkit.reporting._figures", return_value=([], []))
        self.mock_figures = self.plotter.start()
        self.addCleanup(self.plotter.stop)

    def read_csv(self, name):
        with (self.output / name).open() as source:
            return list(csv.DictReader(source))

    def test_all_rank_tokens_are_divided_once_and_missing_stays_empty(self):
        fixture_run(self.runs, "sft", gpus=4, rate=120)
        result = generate(self.runs, self.output)
        row = self.read_csv("measurements.csv")[0]
        self.assertEqual(result["runs"], 1)  # Ignore nested data/manifest.json.
        self.assertEqual(float(row["tokens_per_s"]), 120)
        self.assertEqual(float(row["tokens_per_gpu_s"]), 30)
        self.assertEqual(row["peak_memory_gib"], "")
        self.assertIn("including masked prompts", (self.output / "report.md").read_text())

    def test_preflight_and_partial_failures_remain_visible(self):
        failed = self.runs / "preflight"
        write_json(failed / "config.json", {"model": "fixture/blocked", "mode": "grpo", "gpus": 8})
        write_json(failed / "manifest.json", {"state": "failed", "exit_code": 1, "error": "fixture access failure"})
        partial = fixture_run(self.runs, "interrupted")
        write_json(partial / "manifest.json", {"state": "failed", "exit_code": 1, "error": "fixture interruption"})
        result = generate(self.runs, self.output)
        rows = {row["run_id"]: row for row in self.read_csv("measurements.csv")}
        self.assertEqual(result["runs"], 2)
        self.assertEqual(result["passed"], 0)
        self.assertEqual(rows["preflight"]["tokens_per_s"], "")
        self.assertEqual(rows["interrupted"]["status"], "failed")
        self.assertIn("fixture access failure", rows["preflight"]["problems"])
        self.assertEqual(self.read_csv("scaling.csv"), [])

    def test_evidence_collection_failure_overrides_passed_training_summary(self):
        failed = fixture_run(self.runs, "capture-failed", gpus=2, rate=100)
        fixture_run(self.runs, "complete", gpus=4, rate=240)
        write_json(failed / "manifest.json", {
            "state": "failed", "training_state": "passed", "exit_code": 1,
            "evidence_errors": ["Fixture: checkpoint inventory read failed"],
        })
        result = generate(self.runs, self.output)
        rows = {row["run_id"]: row for row in self.read_csv("measurements.csv")}
        self.assertEqual(result["passed"], 1)
        self.assertEqual(rows["capture-failed"]["status"], "failed")
        self.assertEqual(float(rows["capture-failed"]["tokens_per_s"]), 100)
        self.assertEqual(self.read_csv("scaling.csv"), [])
        self.assertEqual([run["run_id"] for run in self.mock_figures.call_args.args[0]], ["complete"])

    def test_scaling_uses_all_repetitions_and_matches_workload(self):
        fixture_run(self.runs, "two-a", gpus=2, rate=100)
        fixture_run(self.runs, "two-b", gpus=2, rate=200)
        fixture_run(self.runs, "four", gpus=4, rate=240)
        fixture_run(self.runs, "different-batch", gpus=8, rate=900, global_batch_size=16)
        fixture_run(self.runs, "different-gpu", gpus=8, rate=900, hardware="NVIDIA RTX A6000")
        fixture_run(self.runs, "unresolved", gpus=8, rate=900, model_revision="main")
        generate(self.runs, self.output)
        rows = self.read_csv("scaling.csv")
        self.assertEqual(len(rows), 2)
        self.assertEqual([int(row["gpus"]) for row in rows], [2, 4])
        self.assertEqual(float(rows[0]["mean_tokens_per_s"]), 150)
        self.assertEqual(int(rows[0]["replicates"]), 2)
        self.assertAlmostEqual(float(rows[1]["efficiency"]), .8)
        self.assertEqual(rows[1]["std_tokens_per_s"], "")

    def test_framework_revision_separates_scaling_groups(self):
        fixture_run(self.runs, "two", gpus=2)
        changed = fixture_run(self.runs, "four", gpus=4)
        path = changed / "environment.json"
        environment = json.loads(path.read_text())
        environment["framework"]["verl_commit"] = "e" * 40
        write_json(path, environment)
        generate(self.runs, self.output)
        self.assertEqual(self.read_csv("scaling.csv"), [])

    def test_malformed_summary_does_not_erase_attempt(self):
        broken = fixture_run(self.runs, "malformed")
        (broken / "summary.json").write_text("{broken")
        result = generate(self.runs, self.output)
        self.assertEqual(result["runs"], 1)
        row = self.read_csv("measurements.csv")[0]
        self.assertEqual(row["status"], "failed")
        self.assertIn("unreadable:summary.json", row["problems"])

    def test_zero_learning_signal_is_visible_without_hiding_execution(self):
        directory = fixture_run(self.runs, "zero-signal")
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        summary.update({
            "nonzero_gradient_observed": False,
            "gradient_norm": {"mean": 0, "max": 0},
            "reward_audit": {"reward/learning_signal_detected": False,
                             "reward/group_nonzero_std_fraction": 0},
            "warnings": ["Fixture: no learning signal"],
        })
        write_json(path, summary)
        generate(self.runs, self.output)
        row = self.read_csv("measurements.csv")[0]
        self.assertEqual(row["status"], "passed")
        self.assertEqual(row["nonzero_gradient_observed"], "False")
        self.assertEqual(row["reward_learning_signal_detected"], "False")
        self.assertEqual(row["reward_group_nonzero_std_fraction"], "0")
        self.assertIn("Fixture: no learning signal", (self.output / "report.md").read_text())

    def test_source_tree_hash_supports_matching_rsync_runs(self):
        for name, count in (("two", 2), ("four", 4)):
            directory = fixture_run(self.runs, name, gpus=count)
            path = directory / "environment.json"
            environment = json.loads(path.read_text())
            environment["environment"].update({
                "source_tree_sha256": "f" * 64,
                "git": {"returncode": 128, "stdout": ""},
                "git_dirty": {"returncode": 128, "stdout": ""},
            })
            write_json(path, environment)
        generate(self.runs, self.output)
        self.assertEqual(len(self.read_csv("scaling.csv")), 2)

    def test_unresolved_local_revision_never_falls_back_to_requested_revision(self):
        directory = fixture_run(self.runs, "local-unverified", resolved_model_revision=None)
        (directory / "summary.json").unlink()
        write_json(directory / "manifest.json", {"state": "running"})
        generate(self.runs, self.output)
        row = self.read_csv("measurements.csv")[0]
        self.assertEqual(row["model_revision"], "")

    def test_response_metrics_and_device_telemetry_keep_distinct_scopes(self):
        directory = fixture_run(self.runs, "proxy", mode="grpo")
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        summary.update({
            "responses_per_s": 1.6, "gpu_seconds_per_response": 1.25,
            "reward_audit": {"reward/samples": 16}, "peak_memory_gib": 3,
            "telemetry": {
                "scope": "allocated GPUs, complete framework subprocess lifetime",
                "devices": {"fixture-gpu-0": {
                    "utilization_pct": {"n": 3, "mean": 20},
                    "memory_used_mib": {"n": 3, "max": 5120},
                    "power_w": {"n": 3, "mean": 120},
                }},
            },
        })
        write_json(path, summary)
        fixture_run(self.runs, "missing-response-metrics", mode="grpo")
        generate(self.runs, self.output)
        rows = {row["run_id"]: row for row in self.read_csv("measurements.csv")}
        self.assertEqual(float(rows["proxy"]["responses_per_s"]), 1.6)
        self.assertEqual(float(rows["proxy"]["gpu_seconds_per_response"]), 1.25)
        self.assertEqual(rows["missing-response-metrics"]["responses_per_s"], "")
        self.assertEqual(rows["proxy"]["peak_memory_gib"], "3")
        self.assertEqual(json.loads(rows["proxy"]["gpu_telemetry"])["fixture-gpu-0"]["memory_used_mib"]["max"], 5120)
        text = (self.output / "report.md").read_text()
        self.assertIn("Native peak allocated (GiB)", text)
        self.assertIn("Sampled peak memory (MiB)", text)
        self.assertIn("including initialization and warmup", text)

    def test_review_exclusion_preserves_observations_but_removes_comparisons(self):
        original = fixture_run(self.runs, "prototype", gpus=2, rate=100)
        fixture_run(self.runs, "eligible", gpus=4, rate=240)
        original_summary = (original / "summary.json").read_bytes()
        write_json(original / "review.json", {
            "exclude_from_comparisons": True,
            "reason": "Fixture: runtime cache setting differed from declared configuration",
        })
        result = generate(self.runs, self.output)
        rows = {row["run_id"]: row for row in self.read_csv("measurements.csv")}
        row = rows["prototype"]
        self.assertEqual(result["passed"], 2)
        self.assertEqual(result["comparison_excluded"], 1)
        self.assertEqual(result["comparison_eligible"], 1)
        self.assertEqual(row["status"], "passed")
        self.assertEqual(float(row["tokens_per_s"]), 100)
        self.assertEqual(row["comparison_excluded"], "True")
        self.assertIn("runtime cache setting differed", row["comparison_exclusion_reason"])
        self.assertEqual(rows["eligible"]["comparison_excluded"], "False")
        self.assertEqual(self.read_csv("scaling.csv"), [])
        plotted_runs = self.mock_figures.call_args.args[0]
        self.assertEqual([run["run_id"] for run in plotted_runs], ["eligible"])
        self.assertEqual(len(self.read_csv("stages.csv")), 2)
        self.assertEqual((original / "summary.json").read_bytes(), original_summary)
        self.assertIn("runtime cache setting differed", (self.output / "report.md").read_text())

    def test_invalid_review_cannot_silently_enable_comparison(self):
        directory = fixture_run(self.runs, "invalid-review")
        write_json(directory / "review.json", {"exclude_from_comparisons": "false"})
        generate(self.runs, self.output)
        row = self.read_csv("measurements.csv")[0]
        self.assertEqual(row["status"], "passed")
        self.assertEqual(row["comparison_excluded"], "True")
        self.assertIn("must be boolean", row["comparison_exclusion_reason"])

    def test_hashed_tokenizer_assets_ignore_runtime_paths_only(self):
        for name, count in (("two", 2), ("four", 4)):
            fixture_run(self.runs, name, gpus=count, chat_template=f"/fixture/{name}/template.jinja",
                        chat_template_sha256="e" * 64, derived_tokenizer_config_sha256="f" * 64,
                        model_source_path=f"/fixture/{name}/snapshot",
                        model_assets_manifest=f"/fixture/{name}/model_assets_manifest.json",
                        tokenizer_assets=f"/fixture/{name}/tokenizer")
        generate(self.runs, self.output)
        self.assertEqual(len(self.read_csv("scaling.csv")), 2)
        path = self.runs / "four" / "summary.json"
        summary = json.loads(path.read_text())
        summary["config"]["derived_tokenizer_config_sha256"] = "9" * 64
        write_json(path, summary)
        generate(self.runs, self.output)
        self.assertEqual(self.read_csv("scaling.csv"), [])

    def test_nested_native_counts_are_not_reported_as_seconds(self):
        directory = fixture_run(self.runs, "legacy-native")
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        summary["stages"]["timing_s/agent_loop/slowest/prompt_length"] = {"n": 2, "mean": 200}
        summary["stages"]["timing_s/agent_loop/num_preempted"] = {"n": 2, "mean": 3}
        summary["stages"]["timing_s/gen"] = {"n": 2, "mean": .5}
        write_json(path, summary)
        original_summary = path.read_bytes()
        generate(self.runs, self.output)
        rows = self.read_csv("stages.csv")
        self.assertEqual({row["stage"] for row in rows}, {"timing_s/gen", "timing_s/train_step"})
        self.assertNotIn("prompt_length", (self.output / "report.md").read_text())
        self.assertEqual(path.read_bytes(), original_summary)

    def test_artifact_index_preserves_failed_attempts_and_does_not_follow_symlinks(self):
        directory = fixture_run(self.runs, "artifact-attempt", artifact_capture=True,
                                capture_every_step=1, save_checkpoint=True,
                                checkpoint_every_steps=1, checkpoint_keep_all=True)
        write_json(directory / "manifest.json", {"state": "failed", "exit_code": 1})
        checkpoint = directory / "checkpoints" / "global_step_1" / "fixture weights.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixture-only")
        external = self.root / "external-cache"
        external.mkdir()
        (external / "must-not-index.bin").write_bytes(b"external")
        (directory / "model-reference").symlink_to(external, target_is_directory=True)
        write_json(directory / "artifact_inventory.json", {
            "schema_version": 1, "hash_elapsed_s": .25, "errors": [],
            "files": [{"path": "checkpoints/global_step_1/fixture weights.pt", "kind": "file",
                       "bytes": 12, "sha256": "a" * 64}],
        })
        generate(self.runs, self.output)
        rows = self.read_csv("artifacts.csv")
        checkpoint_row = next(row for row in rows if row["category"] == "checkpoint")
        self.assertEqual(checkpoint_row["status"], "failed")
        self.assertEqual(checkpoint_row["size_bytes"], "12")
        self.assertEqual(checkpoint_row["recorded_sha256"], "a" * 64)
        self.assertIn("fixture%20weights.pt", checkpoint_row["link"])
        link = next(row for row in rows if row["path"] == "model-reference")
        self.assertEqual(link["file_type"], "symlink")
        self.assertEqual(link["size_bytes"], "")
        self.assertEqual(link["external_reference"], "True")
        self.assertFalse(any("must-not-index" in row["path"] for row in rows))
        report = (self.output / "report.md").read_text()
        self.assertIn("artifacts.md", report)
        self.assertIn("none observed", report)
        self.assertIn("Post-run hashing (s)", report)
        self.assertIn("0.25", report)


if __name__ == "__main__":
    unittest.main()
