"""Temporary fixture-only campaign lifecycle tests; no subprocess or GPU work."""

from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchkit import campaign
from benchkit.deadline import RunResult


class CampaignLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def execute(self, name="campaign", failures=None, report_failures=None, cleanup_case=None):
        """Simulate only native output files and supervised command outcomes."""
        output = self.root / name
        args = Namespace(output=str(output), steps=[10, 100, 1000],
                         models=["llama8b", "qwen27b"], modes=["sft", "grpo"],
                         dry_run=False, budget_minutes=None, case_timeout_hours=24,
                         report_timeout_minutes=10, report_python="fixture-report-python")
        failures, report_failures = set(failures or []), set(report_failures or [])
        events = []

        def fake_run(command, **kwargs):
            self.assertNotIn("RAY_ADDRESS", kwargs["env"])
            destination = Path(command[command.index("--output") + 1])
            identity = destination.relative_to(output).as_posix()
            action = command[3]
            events.append((action, identity))
            destination.mkdir(parents=True, exist_ok=True)
            if action == "report":
                (destination / "report.md").write_text("# Fixture-only report\n", encoding="utf-8")
                return RunResult(2 if identity in report_failures else 0, False, .01, [])
            self.assertEqual(action, "run")
            exit_code = 1 if identity in failures else 0
            status = "failed" if exit_code else "passed"
            campaign.write_json(destination / "manifest.json", {"state": status, "exit_code": exit_code})
            campaign.write_json(destination / "summary.json", {
                "status": status, "fixture_only": True, "problems": []})
            if identity == cleanup_case:
                return RunResult(1, False, .01, ["Fixture: owned child survived cleanup"])
            return RunResult(exit_code, False, .01, [])

        with patch("benchkit.deadline.run", side_effect=fake_run) as runner, \
                patch.object(campaign.sys, "platform", "linux"), \
                patch.dict(campaign.os.environ, {"RAY_ADDRESS": "fixture-unrelated-cluster"}), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = campaign.run_campaign(args)
        record = json.loads((output / "campaign.json").read_text())
        return code, record, events, output, runner.call_count

    def test_all_cases_run_in_stage_order_with_stage_and_final_reports(self):
        code, record, events, output, calls = self.execute()
        expected = []
        for step in (10, 100, 1000):
            expected.extend(("run", f"steps-{step:04d}/{model}-{mode}")
                            for model in ("llama8b", "qwen27b") for mode in ("sft", "grpo"))
            expected.append(("report", f"reports/steps-{step:04d}"))
        expected.append(("report", "report"))
        self.assertEqual(events, expected)
        self.assertEqual(calls, 16)
        self.assertEqual(code, 0)
        self.assertEqual(record["state"], "passed")
        self.assertEqual(len(record["cases"]), 12)
        self.assertTrue(all(case["state"] == "passed" for case in record["cases"]))
        self.assertEqual([report["stage"] for report in record["reports"]], [10, 100, 1000, None])
        latest = json.loads((output / "latest_report.json").read_text())
        self.assertEqual(latest["path"], "report/report.md")

    def test_last_case_failure_blocks_only_its_later_cases_and_preserves_reports(self):
        code, record, events, output, _ = self.execute(failures={"steps-0010/qwen27b-grpo"})
        states = {case["id"]: case for case in record["cases"]}
        self.assertEqual(code, 1)
        self.assertEqual(record["state"], "incomplete")
        self.assertEqual(states["steps-0010/qwen27b-grpo"]["state"], "failed")
        for step in (100, 1000):
            identity = f"steps-{step:04d}/qwen27b-grpo"
            self.assertEqual(states[identity]["state"], "blocked")
            self.assertIn("earlier_stage_failed", states[identity]["reason"])
            self.assertNotIn(("run", identity), events)
            self.assertEqual(states[f"steps-{step:04d}/qwen27b-sft"]["state"], "passed")
            self.assertIn(("report", f"reports/steps-{step:04d}"), events)
        self.assertEqual(events[-1], ("report", "report"))
        self.assertTrue((output / "report" / "report.md").is_file())
        self.assertTrue((output / "steps-0010" / "qwen27b-grpo" / "summary.json").is_file())

    def test_cleanup_error_stops_remaining_cases_and_records_partial_state(self):
        code, record, events, output, _ = self.execute(cleanup_case="steps-0010/llama8b-sft")
        self.assertEqual(code, 1)
        self.assertEqual(events, [("run", "steps-0010/llama8b-sft")])
        self.assertTrue(record["cleanup_stop"])
        self.assertEqual(record["state"], "incomplete")
        self.assertEqual(record["cases"][0]["state"], "failed")
        self.assertTrue(all(case["state"] == "not_run" for case in record["cases"][1:]))
        manifest = json.loads((output / "steps-0010/llama8b-sft/manifest.json").read_text())
        self.assertEqual(manifest["training_state"], "passed")
        self.assertTrue(manifest["campaign_watchdog"]["cleanup_errors"])

    def test_report_failure_keeps_training_success_but_returns_nonzero(self):
        code, record, events, output, _ = self.execute(report_failures={"report"})
        self.assertEqual(code, 1)
        self.assertEqual(record["state"], "incomplete")
        self.assertTrue(all(case["state"] == "passed" for case in record["cases"]))
        self.assertEqual(record["reports"][-1]["returncode"], 2)
        self.assertEqual(events[-1], ("report", "report"))
        latest = json.loads((output / "latest_report.json").read_text())
        self.assertEqual(latest["path"], "reports/steps-1000/report.md")


if __name__ == "__main__":
    unittest.main()
