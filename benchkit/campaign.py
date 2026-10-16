"""Run independent 10/100/1000-step cases on one eight-GPU H100 allocation."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .cli import ROOT, load_config, resolve_model, write_json
from .metrics import gpu_telemetry, read_metrics, summarize

MODELS = ("llama8b", "qwen27b")
MODES = ("sft", "grpo")


def now():
    return datetime.now(timezone.utc).isoformat()


def plan(steps=(10, 100, 1000), models=MODELS, modes=MODES):
    if not steps or list(steps) != sorted(set(steps)) or any(type(s) is not int or s <= 2 for s in steps):
        raise ValueError("steps must be distinct increasing integers greater than the two warmup steps")
    if (not models or not modes or len(set(models)) != len(models) or len(set(modes)) != len(modes)
            or not set(models) <= set(MODELS) or not set(modes) <= set(MODES)):
        raise ValueError("models and modes must be nonempty unique selections from the supported cases")
    cases = []
    for count in steps:
        for model in models:
            for mode in modes:
                config = ROOT / "configs" / f"h100_quick_{model}_{mode}.json"
                cases.append({"id": f"steps-{count:04d}/{model}-{mode}", "model": model,
                              "mode": mode, "steps": count, "config": str(config), "state": "pending"})
    return cases


def prepare(output, models=MODELS):
    """Download pinned snapshots before timed execution; no GPU allocation needed."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Preparation record exists: {output}. Choose a new output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"state": "preparing", "started_utc": now(), "models": []}
    write_json(output, record)
    try:
        for model in models:
            config = load_config(ROOT / "configs" / f"h100_quick_{model}_sft.json")
            config["model_cache_only"] = False
            print(f"Preparing pinned model: {config['model']}", flush=True)
            resolved = resolve_model(config)
            record["models"].append({"id": resolved["requested_model"],
                "revision": resolved["resolved_model_revision"], "cache_path": resolved["model"]})
            write_json(output, record)
        record["state"] = "prepared"
        return 0
    except Exception as exc:
        record.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["finished_utc"] = now()
        write_json(output, record)


def finalize_attempt(directory, config, outcome):
    """Preserve partial native observations after a watchdog or startup failure."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"state": "failed"}
    evidence_changed = False
    if not (directory / "config.json").exists():
        write_json(directory / "config.json", config)
        evidence_changed = True
    if outcome.returncode or outcome.timed_out or outcome.cleanup_errors:
        manifest["training_state"] = manifest.get("training_state", manifest.get("state"))
        manifest.update(state="failed", exit_code=outcome.returncode or 1,
                        campaign_watchdog=asdict(outcome), finished_utc=now())
        write_json(manifest_path, manifest)
        evidence_changed = True
    if not (directory / "summary.json").exists():
        evidence_changed = True
        manifest.update(state="failed", exit_code=outcome.returncode or 1)
        manifest.setdefault("evidence_errors", []).append("missing_native_summary; partial metrics recovered where possible")
        write_json(manifest_path, manifest)
        recorded = directory / "config.json"
        config = json.loads(recorded.read_text()) if recorded.exists() else config
        try:
            result = summarize(config, read_metrics(directory / "metrics.jsonl"), outcome.returncode or 1,
                               outcome.elapsed, gpu_telemetry(directory / "gpu.csv"))
            result["warnings"] = ["Partial attempt recovered by campaign; original native metrics retained."]
            write_json(directory / "summary.json", result)
        except (ValueError, OSError) as exc:
            # A killed writer can leave a partial last JSONL line. Never repair
            # the original stream in place or report invented completion.
            manifest.setdefault("evidence_errors", []).append(f"Partial summary unavailable: {exc}")
            manifest.update(state="failed", exit_code=outcome.returncode or 1)
            write_json(manifest_path, manifest)
    if evidence_changed:
        # The parent appended watchdog evidence. Refresh small-file hashes;
        # full checkpoint data remains retained and can be hashed after the run.
        from .run_evidence import inventory
        inventory(directory, max_hash_bytes=64 * 1024 * 1024)
    return manifest.get("state", "failed")


def run_campaign(args):
    cases = plan(args.steps, args.models, args.modes)
    output = Path(args.output).resolve()
    if args.dry_run:
        print(json.dumps({"cases": cases, "global_budget_minutes": args.budget_minutes,
                          "case_timeout_hours": args.case_timeout_hours,
                          "checkpoint_policy": "final_only", "capture_every_step": 1}, indent=2))
        return 0
    if sys.platform != "linux":
        raise RuntimeError("Campaign supervision requires Linux; dry-run works on other platforms")
    if output.exists():
        raise ValueError(f"Campaign output already exists: {output}. Use a new run ID.")
    output.mkdir(parents=True)
    from .deadline import run as supervised_run
    started = time.monotonic()
    budget_end = started + args.budget_minutes * 60 if args.budget_minutes is not None else None
    record = {"schema_version": 1, "state": "running", "started_utc": now(), "cases": cases,
              "global_budget_minutes": args.budget_minutes, "case_timeout_hours": args.case_timeout_hours,
              "order": "stage first, then model and mode; fresh pinned initialization for every attempt",
              "reports": [], "preparation_included": False}
    env = dict(os.environ)
    # Never connect to another run's Ray workers. Native main_ppo starts this
    # invocation's own cluster; the supervisor owns its entire process tree.
    env.pop("RAY_ADDRESS", None)
    failed = set()
    safety_stop = False

    def persist():
        record["elapsed_s"] = time.monotonic() - started
        record["allocated_gpu_hours_wall"] = record["elapsed_s"] * 8 / 3600
        write_json(output / "campaign.json", record)

    def report(stage=None):
        destination = output / "reports" / f"steps-{stage:04d}" if stage is not None else output / "report"
        outcome = supervised_run([args.report_python, "-m", "benchkit.cli", "report",
                    "--input", str(output), "--output", str(destination)],
                    cwd=ROOT, env=env, log_path=output / "logs" / f"report-{stage}.log",
                    timeout_s=args.report_timeout_minutes * 60)
        record["reports"].append({"stage": stage, "path": str(destination.relative_to(output)), **asdict(outcome)})
        if outcome.returncode == 0:
            report_path = destination / "report.md"
            if report_path.is_file():
                lines = ["\n## Campaign plan and execution status\n",
                         "All cases use eight GPUs. Target step counts are not observed completion counts.\n",
                         "| Case | Target steps | State | Reason |", "|---|---:|---|---|"]
                for case in cases:
                    reason = str(case.get("reason", "")).replace("|", "\\|").replace("\n", " ")
                    lines.append(f"| {case['id']} | {case['steps']} | {case['state']} | {reason} |")
                lines.append("\nNative per-step metrics and campaign.json retain the underlying evidence.\n")
                with report_path.open("a") as stream:
                    stream.write("\n".join(lines))
            write_json(output / "latest_report.json", {"path": str((destination / "report.md").relative_to(output))})
        persist()
        return bool(outcome.cleanup_errors)

    previous_term = signal.getsignal(signal.SIGTERM)
    def stop_campaign(signum, frame):
        raise KeyboardInterrupt(f"campaign received signal {signum}")
    signal.signal(signal.SIGTERM, stop_campaign)
    persist()
    try:
        for stage, stage_cases in itertools.groupby(cases, key=lambda c: c["steps"]):
            for case in stage_cases:
                key = (case["model"], case["mode"])
                remaining = None if budget_end is None else budget_end - time.monotonic()
                if remaining is not None and remaining < 30:
                    case.update(state="not_run", reason="global_budget_exhausted")
                    persist()
                    continue
                if key in failed:
                    case.update(state="blocked", reason="earlier_stage_failed; repair before retrying unchanged case")
                    persist()
                    continue
                timeout = args.case_timeout_hours * 3600
                if remaining is not None:
                    timeout = min(timeout, remaining - 5)
                overrides = [f"steps={case['steps']}", f"timeout_s={max(1, int(timeout - 20))}",
                             "artifact_hash_max_bytes=67108864"]
                config = load_config(case["config"], overrides=overrides)
                if config["gpus"] != 8 or config["nodes"] != 1 or config.get("ray_address"):
                    raise ValueError("Campaign requires one node, eight GPUs and a private local Ray cluster")
                directory = output / case["id"]
                case.update(state="running", started_utc=now(), timeout_s=timeout)
                persist()
                print(f"Running {case['id']} ({case['steps']} optimizer steps, 8 GPUs)", flush=True)
                command = [sys.executable, "-m", "benchkit.cli", "run", "--config", case["config"],
                           "--output", str(directory)]
                for override in overrides:
                    command += ["--set", override]
                outcome = supervised_run(command, cwd=ROOT, env=env,
                           log_path=output / "logs" / (case["id"].replace("/", "-") + ".log"), timeout_s=timeout)
                directory.mkdir(parents=True, exist_ok=True)
                state = finalize_attempt(directory, config, outcome)
                case.update(state=state, finished_utc=now(), **asdict(outcome))
                if state != "passed":
                    failed.add(key)
                persist()
                if outcome.cleanup_errors:
                    safety_stop = True
                    break
            if safety_stop or report(stage):
                safety_stop = True
                break
    except KeyboardInterrupt:
        record["interrupted"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(record["error"], file=sys.stderr)
    finally:
        for case in cases:
            if case["state"] in {"running", "pending"}:
                case.update(state="not_run" if case["state"] == "pending" else "interrupted",
                            reason="campaign_stopped")
        if not safety_stop:
            try:
                safety_stop = report()
            except KeyboardInterrupt:
                record["report_error"] = "Report interrupted; retained raw case evidence"
            except Exception as exc:
                record["report_error"] = f"{type(exc).__name__}: {exc}"
        complete = (all(c["state"] == "passed" for c in cases)
                    and not record.get("report_error") and not record.get("error")
                    and all(r["returncode"] == 0 for r in record["reports"]))
        record.update(state="passed" if complete else "incomplete",
                      cleanup_stop=safety_stop, finished_utc=now())
        persist()
        signal.signal(signal.SIGTERM, previous_term)
    print(f"Campaign {record['state']}: {output / 'campaign.json'}", flush=True)
    return 0 if record["state"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="Download exact model revisions before execution")
    prep.add_argument("--output", required=True)
    prep.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    run = sub.add_parser("run", help="Run 10/100/1000-step stages and retain partial evidence")
    run.add_argument("--output", required=True)
    run.add_argument("--steps", nargs="+", type=int, default=[10, 100, 1000])
    run.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    run.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    run.add_argument("--budget-minutes", type=float, help="Optional global budget; default is complete all stages")
    run.add_argument("--case-timeout-hours", type=float, default=24)
    run.add_argument("--report-timeout-minutes", type=float, default=10)
    run.add_argument("--report-python", default=str(ROOT / ".venv-report/bin/python"))
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        return prepare(args.output, args.models)
    for name in ("budget_minutes", "case_timeout_hours", "report_timeout_minutes"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
