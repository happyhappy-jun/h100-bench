"""Build an evidence report from recorded runs, without hardware extrapolation."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import stat
import statistics
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from .metrics import finite, read_metrics


MEASUREMENT_FIELDS = [
    "run_id", "status", "comparison_excluded", "comparison_exclusion_reason",
    "evidence_class", "hardware", "gpu_models", "gpu_identity_source", "cuda_runtime", "model",
    "model_revision", "mode", "gpus", "nodes", "token_scope", "measured_steps",
    "warmup_steps", "measured_tokens", "measured_time_s", "tokens_per_s",
    "tokens_per_gpu_s", "responses_per_s", "gpu_seconds_per_response", "response_samples",
    "step_time_mean_s", "step_time_median_s", "step_time_p95_s",
    "peak_memory_gib", "framework_mfu_mean", "wall_time_s", "verl_commit",
    "framework_versions", "source_tree_sha256", "gradient_norm_n", "gradient_norm_mean", "gradient_norm_max",
    "nonzero_gradient_observed", "reward_learning_signal_detected",
    "reward_group_nonzero_std_fraction", "reward_proxy_accuracy", "reward_audit",
    "telemetry_scope", "gpu_telemetry", "warnings", "problems", "scaling_exclusion",
]
STAGE_FIELDS = [
    "run_id", "status", "comparison_excluded", "comparison_exclusion_reason",
    "hardware", "model", "mode", "gpus", "stage",
    "n", "mean_s", "median_s", "p95_s", "min_s", "max_s",
]
SCALING_FIELDS = [
    "group_id", "hardware", "gpu_models", "model", "mode", "baseline_gpus",
    "gpus", "nodes", "gpus_per_node", "replicates", "run_ids", "mean_tokens_per_s", "std_tokens_per_s",
    "mean_tokens_per_gpu_s", "speedup", "efficiency",
]
ARTIFACT_FIELDS = [
    "run_id", "status", "comparison_excluded", "category", "path", "file_type",
    "size_bytes", "recorded_sha256", "external_reference", "link", "error",
]
_TRANSIENT_CONFIG = {
    "name", "gpus", "nodes", "evidence_class", "timeout_s", "input_data",
    "data_paths", "python", "node_rank", "master_addr", "master_port",
    "ray_address", "model", "model_path", "resolved_model_path",
}


def _read_json(path, issues, required=False):
    if not path.exists():
        if required:
            issues.append(f"missing:{path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        issues.append(f"unreadable:{path.name}:{type(error).__name__}")
        return {}


def _number(value):
    return value if finite(value) else None


def _positive(value):
    return finite(value) and value > 0


def _sha256_identity(value):
    return isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) is not None


def _comparison_review(directory):
    """Read an additive QA overlay without rewriting execution evidence."""
    issues = []
    review = _read_json(directory / "review.json", issues)
    if issues:
        return True, "; ".join(issues)
    excluded = review.get("exclude_from_comparisons", False)
    reason = review.get("reason", "")
    if not isinstance(excluded, bool):
        return True, "invalid:review.json:exclude_from_comparisons must be boolean"
    if excluded and (not isinstance(reason, str) or not reason.strip()):
        return True, "invalid:review.json:comparison exclusion requires a reason"
    return excluded, reason.strip() if excluded else ""


def _comparison_runs(runs):
    return [run for run in runs if run["status"] == "passed" and not run["comparison_excluded"]]


def _nested_number(mapping, key):
    return _number(mapping.get(key)) if isinstance(mapping, dict) else None


def _gpu_environment(environment):
    """Parse recorded nvidia-smi inventory, never use its count as allocation."""
    captured = environment.get("gpu", {})
    if not isinstance(captured, dict):
        captured = {}
    names, memories, drivers = set(), set(), set()
    for fields in csv.reader(captured.get("stdout", "").splitlines()):
        if len(fields) < 4:
            continue
        name, _uuid, memory, driver = (field.strip() for field in fields[:4])
        if name.lower() == "name":
            continue
        names.add(name)
        memories.add(memory)
        drivers.add(driver)
    models = ", ".join(sorted(names))
    if names and all("H100" in name.upper() for name in names):
        family = "H100"
    elif names and all("A6000" in name.upper() for name in names):
        family = "A6000"
    else:
        family = "other" if names else "unknown"
    return family, models, ", ".join(sorted(memories)), ", ".join(sorted(drivers))


def _visible_gpu_evidence(captured):
    probe = captured.get("cuda", {})
    if not isinstance(probe, dict) or probe.get("returncode") != 0:
        return {}
    try:
        value = json.loads(probe.get("stdout", "").splitlines()[-1])
        return value if isinstance(value, dict) else {}
    except (IndexError, TypeError, ValueError):
        return {}


def _host_environment(environment):
    cpu = environment.get("cpu", {})
    cpu = cpu if isinstance(cpu, dict) else {}
    fields = dict(line.split(":", 1) for line in cpu.get("stdout", "").splitlines() if ":" in line)
    model = fields.get("Model name", "").strip()
    cores = fields.get("CPU(s)", "").strip()
    memory = environment.get("host_memory", {})
    memory = memory if isinstance(memory, dict) else {}
    total_bytes = None
    for line in memory.get("stdout", "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "Mem:":
            try:
                total_bytes = int(parts[1])
            except ValueError:
                pass
    return model, cores, total_bytes / 2 ** 30 if total_bytes is not None else None


def _discover(input_root):
    directories = {path.parent for path in input_root.rglob("summary.json")}
    for manifest in input_root.rglob("manifest.json"):
        # data/manifest.json records dataset provenance and is not a run.
        if (manifest.parent / "config.json").is_file():
            directories.add(manifest.parent)
            continue
        value = _read_json(manifest, [])
        if "state" in value or "started_utc" in value:
            directories.add(manifest.parent)
    return sorted(directories)


def _load_run(directory, input_root):
    issues = []
    summary = _read_json(directory / "summary.json", issues)
    manifest = _read_json(directory / "manifest.json", issues, required=True)
    stored_config = _read_json(directory / "config.json", issues)
    config = summary.get("config") or stored_config
    if not isinstance(config, dict):
        issues.append("invalid:config")
        config = stored_config
    captured = _read_json(directory / "environment.json", issues)
    environment = captured.get("environment", captured)
    if not isinstance(environment, dict):
        environment = {}
    framework = captured.get("framework", {})
    if not isinstance(framework, dict):
        framework = {}
    data_manifest = _read_json(directory / "data" / "manifest.json", issues)
    recorded_problems = summary.get("problems") or []
    problems = list(recorded_problems) if isinstance(recorded_problems, list) else [str(recorded_problems)]
    problems.extend(issues)
    if manifest.get("error"):
        problems.append(str(manifest["error"]))
    evidence_errors = manifest.get("evidence_errors") or []
    if not isinstance(evidence_errors, list):
        evidence_errors = [evidence_errors]
    problems.extend("evidence: " + str(error) for error in evidence_errors)
    state = manifest.get("state", "missing_manifest")
    if state == "passed" and summary.get("status") == "passed" and manifest.get("exit_code") == 0 and not problems:
        status = "passed"
    elif state == "failed" or summary.get("status") == "failed" or problems:
        status = "failed"
    elif state in {"preparing", "running"}:
        status = state
    else:
        status = "incomplete"
    if not summary:
        problems.append("missing_or_unreadable:summary.json")
        if status == "passed":
            status = "incomplete"
    hardware, gpu_models, gpu_memory, driver = _gpu_environment(environment)
    identity_source = "nvidia-smi inventory"
    visible = _visible_gpu_evidence(captured)
    devices = visible.get("devices")
    if isinstance(devices, list) and devices and all(isinstance(name, str) for name in devices):
        gpu_models = ", ".join(sorted(set(devices)))
        identity_source = "CUDA-visible device probe"
        if all("H100" in name.upper() for name in devices):
            hardware = "H100"
        elif all("A6000" in name.upper() for name in devices):
            hardware = "A6000"
        else:
            hardware = "other"
    versions = environment.get("versions") or framework.get("packages") or {}
    if not isinstance(versions, dict):
        versions = {}
    reward = summary.get("reward_audit") or {}
    if not isinstance(reward, dict):
        reward = {}
    telemetry = summary.get("telemetry") or {}
    if not isinstance(telemetry, dict):
        telemetry = {}
    warning_values = summary.get("warnings") or []
    if not isinstance(warning_values, list):
        warning_values = [str(warning_values)]
    gpus = _number(summary.get("gpus"))
    if gpus is None and _positive(config.get("gpus")) and _positive(config.get("nodes", 1)):
        gpus = config["gpus"] * config.get("nodes", 1)
    tokens = _number(summary.get("measured_tokens"))
    duration = _number(summary.get("measured_time_s"))
    # summarize() already aggregates token counts across ranks. Never multiply
    # this rate by world size. Per-GPU throughput uses exactly one division.
    if tokens is not None and duration is not None:
        throughput = tokens / duration if _positive(tokens) and _positive(duration) else None
    else:
        throughput = _number(summary.get("tokens_per_s"))
    per_gpu = throughput / gpus if _positive(throughput) and _positive(gpus) else None
    run_id = directory.relative_to(input_root).as_posix()
    if run_id == ".":
        run_id = directory.name
    if "resolved_model_revision" in summary:
        revision = summary["resolved_model_revision"]
    elif "resolved_model_revision" in config:
        revision = config["resolved_model_revision"]
    else:
        revision = summary.get("model_revision", config.get("model_revision"))
    comparison_excluded, comparison_reason = _comparison_review(directory)
    run = {
        "run_id": run_id, "status": status,
        "comparison_excluded": comparison_excluded,
        "comparison_exclusion_reason": comparison_reason,
        "evidence_class": summary.get("evidence_class", config.get("evidence_class", "unclassified")),
        "hardware": hardware, "gpu_models": gpu_models,
        "gpu_identity_source": identity_source, "cuda_runtime": visible.get("cuda_runtime"),
        "gpu_memory": gpu_memory, "driver": driver,
        "model": summary.get("model", config.get("model")),
        "model_revision": revision,
        "mode": summary.get("mode", config.get("mode")), "gpus": gpus,
        "nodes": summary.get("nodes", config.get("nodes")),
        "token_scope": summary.get("token_scope"),
        "measured_steps": _number(summary.get("measured_steps")),
        "warmup_steps": summary.get("warmup_steps", config.get("warmup_steps")),
        "measured_tokens": tokens, "measured_time_s": duration,
        "tokens_per_s": throughput, "tokens_per_gpu_s": per_gpu,
        "responses_per_s": _number(summary.get("responses_per_s")),
        "gpu_seconds_per_response": _number(summary.get("gpu_seconds_per_response")),
        "response_samples": _number(reward.get("reward/samples")),
        "step_time_mean_s": _nested_number(summary.get("step_time_s"), "mean"),
        "step_time_median_s": _nested_number(summary.get("step_time_s"), "median"),
        "step_time_p95_s": _nested_number(summary.get("step_time_s"), "p95"),
        "peak_memory_gib": _number(summary.get("peak_memory_gib")),
        "framework_mfu_mean": _nested_number(summary.get("framework_mfu"), "mean"),
        "wall_time_s": _number(summary.get("wall_time_s", manifest.get("elapsed_s"))),
        "verl_commit": framework.get("verl_commit"),
        "framework_versions": json.dumps(versions, sort_keys=True),
        "source_tree_sha256": environment.get("source_tree_sha256"),
        "gradient_norm_n": _nested_number(summary.get("gradient_norm"), "n"),
        "gradient_norm_mean": _nested_number(summary.get("gradient_norm"), "mean"),
        "gradient_norm_max": _nested_number(summary.get("gradient_norm"), "max"),
        "nonzero_gradient_observed": summary.get("nonzero_gradient_observed"),
        "reward_learning_signal_detected": reward.get("reward/learning_signal_detected", reward.get("learning_signal_detected")),
        "reward_group_nonzero_std_fraction": _number(reward.get("reward/group_nonzero_std_fraction")),
        "reward_proxy_accuracy": _number(reward.get("reward/proxy_accuracy")),
        "reward_audit": json.dumps(reward, sort_keys=True) if reward else "",
        "telemetry_scope": telemetry.get("scope"),
        "gpu_telemetry": json.dumps(telemetry.get("devices"), sort_keys=True) if telemetry.get("devices") else "",
        "warnings": "; ".join(str(value) for value in warning_values),
        "problems": "; ".join(str(value) for value in problems),
        "config": config, "summary": summary, "manifest": manifest,
        "environment": environment, "framework": framework, "versions": versions,
        "data_manifest": data_manifest, "directory": directory,
    }
    signature, exclusion = _scaling_signature(run)
    run["scaling_signature"], run["scaling_exclusion"] = signature, exclusion
    return run


def _scaling_signature(run):
    """Conservatively require identical work and immutable source identities."""
    missing = []
    if run["comparison_excluded"]:
        missing.append("review exclusion: " + run["comparison_exclusion_reason"])
    if run["status"] != "passed" or not _positive(run["tokens_per_s"]):
        missing.append("completed measurement required")
    revision = str(run.get("model_revision") or "")
    if not re.fullmatch(r"[a-fA-F0-9]{40,64}", revision):
        missing.append("immutable model revision missing")
    if not run["gpu_models"]:
        missing.append("GPU model missing")
    packages = ("torch", "transformers", "verl") + (("vllm", "ray") if run["mode"] == "grpo" else ())
    if any(not run["versions"].get(package) for package in packages):
        missing.append("framework versions missing")
    if not run["verl_commit"]:
        missing.append("verl commit missing")
    if run["framework"].get("source_clean") is not True:
        missing.append("clean verl source evidence missing")
    git = run["environment"].get("git", {})
    dirty = run["environment"].get("git_dirty", {})
    tree = run["environment"].get("source_tree_sha256")
    if isinstance(tree, str) and re.fullmatch(r"[a-fA-F0-9]{64}", tree):
        benchmark_source = {"source_tree_sha256": tree}
    else:
        benchmark_source = {"git_commit": git.get("stdout") if isinstance(git, dict) else None}
        if not isinstance(git, dict) or git.get("returncode") != 0 or not git.get("stdout"):
            missing.append("benchmark source hash or code revision missing")
        if not isinstance(dirty, dict) or dirty.get("returncode") != 0 or dirty.get("stdout"):
            missing.append("source hash or clean benchmark source evidence missing")
    source_hash = run["data_manifest"].get("source_sha256")
    if not source_hash:
        missing.append("dataset content hash missing")
    required = ("global_batch_size", "micro_batch_size", "seq_length", "dtype", "steps", "warmup_steps", "seed")
    if run["mode"] == "grpo":
        required += ("prompt_length", "response_length", "rollout_n", "rollout_tp")
    if any(key not in run["config"] for key in required):
        missing.append("workload settings missing")
    if not _positive(run["gpus"]) or not run.get("token_scope"):
        missing.append("allocation or token scope missing")
    if missing:
        return None, "; ".join(missing)
    config = run["config"]
    transient = set(_TRANSIENT_CONFIG)
    if _sha256_identity(config.get("chat_template_sha256")):
        transient.add("chat_template")
        metadata = config.get("local_model_metadata_sha256") or {}
        if (_sha256_identity(config.get("derived_tokenizer_config_sha256"))
                or isinstance(metadata, dict) and _sha256_identity(metadata.get("tokenizer_config.json"))):
            # Immutable model revision and tokenizer/template hashes identify
            # these assets. Run-local paths do not change the workload.
            transient.update({"model_source_path", "model_assets_manifest"})
            if isinstance(config.get("tokenizer_assets"), str):
                transient.add("tokenizer_assets")
    workload = {key: value for key, value in config.items() if key not in transient}
    provenance = {key: value for key, value in run["data_manifest"].items()
                  if key not in {"source", "rl_validation_rows"}}
    value = {
        "model": run["model"], "model_revision": revision, "mode": run["mode"],
        "gpu_models": run["gpu_models"], "gpu_memory": run["gpu_memory"],
        "driver": run["driver"], "versions": run["versions"],
        "cuda_runtime": run["cuda_runtime"],
        "verl_commit": run["verl_commit"], "benchmark_source": benchmark_source,
        "workload": workload, "data": provenance,
        "token_scope": run["token_scope"], "measured_steps": run["measured_steps"],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), ""


def _scaling_rows(runs):
    groups = defaultdict(list)
    for run in runs:
        if run["scaling_signature"]:
            groups[run["scaling_signature"]].append(run)
    rows = []
    for signature, members in sorted(groups.items()):
        by_count = defaultdict(list)
        for run in members:
            by_count[run["gpus"]].append(run)
        if len(by_count) < 2:
            continue
        if any(len({(run["nodes"], run["config"].get("gpus")) for run in group}) > 1 for group in by_count.values()):
            # Equal total GPU counts on different node layouts are different
            # configurations, not repetitions. Do not choose one layout.
            for run in members:
                run["scaling_exclusion"] = "same GPU count has different node layouts"
            continue
        base_count = min(by_count)
        baseline = statistics.mean(run["tokens_per_s"] for run in by_count[base_count])
        for count, repetitions in sorted(by_count.items()):
            rates = [run["tokens_per_s"] for run in repetitions]
            mean = statistics.mean(rates)
            exemplar = repetitions[0]
            rows.append({
                "group_id": signature[:12], "hardware": exemplar["hardware"],
                "gpu_models": exemplar["gpu_models"], "model": exemplar["model"],
                "mode": exemplar["mode"], "baseline_gpus": base_count, "gpus": count,
                "nodes": exemplar["nodes"], "gpus_per_node": exemplar["config"].get("gpus"),
                "replicates": len(repetitions),
                "run_ids": "; ".join(run["run_id"] for run in repetitions),
                "mean_tokens_per_s": mean,
                "std_tokens_per_s": statistics.stdev(rates) if len(rates) > 1 else None,
                "mean_tokens_per_gpu_s": mean / count,
                "speedup": mean / baseline,
                "efficiency": mean / baseline * base_count / count,
            })
    return rows


def _stage_rows(runs):
    rows = []
    for run in runs:
        stages = run["summary"].get("stages") or {}
        for stage, values in sorted(stages.items()):
            # Older summaries collected nested agent-loop fields under
            # timing_s/. Some are token counts, not seconds. Only top-level
            # native timers have the seconds contract.
            if (not isinstance(values, dict) or not stage.startswith("timing_s/")
                    or "/" in stage.removeprefix("timing_s/")):
                continue
            rows.append({
                **{key: run[key] for key in ("run_id", "status", "comparison_excluded", "comparison_exclusion_reason",
                                            "hardware", "model", "mode", "gpus")},
                "stage": stage, "n": _number(values.get("n")),
                **{key + "_s": _number(values.get(key)) for key in ("mean", "median", "p95", "min", "max")},
            })
    return rows


def _telemetry_rows(runs):
    rows = []
    for run in runs:
        telemetry = run["summary"].get("telemetry") or {}
        if not isinstance(telemetry, dict) or not isinstance(telemetry.get("devices"), dict):
            continue
        for uuid, fields in sorted(telemetry["devices"].items()):
            if not isinstance(fields, dict):
                continue
            utilization = fields.get("utilization_pct")
            memory = fields.get("memory_used_mib")
            power = fields.get("power_w")
            rows.append([
                run["run_id"], run["status"], uuid, telemetry.get("scope"),
                " / ".join(_display(_nested_number(value, "n")) for value in (utilization, memory, power)),
                _nested_number(utilization, "mean"), _nested_number(memory, "max"),
                _nested_number(power, "mean"),
            ])
    return rows


def _artifact_rows(runs, output_root):
    """Index local files without reading checkpoint payloads or symlink targets."""
    rows = []
    for run in runs:
        directory = run["directory"]
        observed = []
        inventory_issues = []
        inventory = _read_json(directory / "artifact_inventory.json", inventory_issues)
        recorded_files = inventory.get("files") or []
        recorded_hashes = {item.get("path"): item.get("sha256") for item in recorded_files
                           if isinstance(item, dict) and _sha256_identity(item.get("sha256"))}

        def record(path, error=None):
            relative = path.relative_to(directory).as_posix()
            first = relative.split("/", 1)[0]
            category = {"artifacts": "native_capture", "checkpoints": "checkpoint", "rollouts": "rollout",
                        "data": "dataset", "model_assets": "model_asset"}.get(first, "run_evidence")
            if path.name.startswith("framework-") and path.suffix == ".log":
                category = "framework_log"
            kind, size, external = "unreadable", None, None
            if error is None:
                try:
                    info = path.lstat()
                    if stat.S_ISLNK(info.st_mode):
                        kind = "symlink"
                        external = not path.resolve(strict=False).is_relative_to(directory.resolve())
                    elif stat.S_ISREG(info.st_mode):
                        kind, size = "file", info.st_size
                    else:
                        kind = "special"
                except OSError as exc:
                    error = type(exc).__name__
            observed.append({
                "run_id": run["run_id"], "status": run["status"],
                "comparison_excluded": run["comparison_excluded"], "category": category,
                "path": relative, "file_type": kind, "size_bytes": size,
                "recorded_sha256": recorded_hashes.get(relative),
                "external_reference": external,
                "link": quote(os.path.relpath(path, output_root).replace(os.sep, "/"), safe="/"),
                "error": str(error) if error else "",
            })

        def walk_error(error):
            record(Path(error.filename), type(error).__name__)

        for parent, directories, filenames in os.walk(directory, followlinks=False, onerror=walk_error):
            parent = Path(parent)
            for name in list(directories):
                path = parent / name
                if path.is_symlink():
                    record(path)
                    directories.remove(name)
                elif path.resolve() == output_root:
                    directories.remove(name)
            for name in sorted(filenames):
                record(parent / name)
        observed.sort(key=lambda row: row["path"])
        run["artifact_inventory"] = {
            "files": sum(row["file_type"] == "file" for row in observed),
            "logical_bytes": sum(row["size_bytes"] or 0 for row in observed),
            "symlinks": sum(row["file_type"] == "symlink" for row in observed),
            "capture_files": sum(row["category"] == "native_capture" and row["file_type"] == "file" for row in observed),
            "checkpoint_files": sum(row["category"] == "checkpoint" and row["file_type"] == "file" for row in observed),
            "errors": sum(bool(row["error"]) for row in observed),
            "hash_elapsed_s": _number(inventory.get("hash_elapsed_s")),
            "capture_inventory_errors": inventory.get("errors") or inventory_issues,
        }
        rows.extend(observed)
    return rows


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Preserve numeric values, but prevent source text becoming a
            # spreadsheet formula when these evidence files are opened.
            safe = {key: ("'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
                    for key, value in row.items() if key in fields}
            writer.writerow(safe)


def _display(value):
    if value is None or value == "":
        return "—"
    if finite(value):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return html.escape(str(value)).replace("|", "\\|").replace("\n", "<br>")


def _table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_display(value) for value in row) + " |" for row in rows),
    ])


def _slug(value):
    prefix = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:65] or "run"
    return prefix + "-" + hashlib.sha256(value.encode()).hexdigest()[:8]


def _figures(runs, stages, scaling, output_root):
    target = output_root / "figures"
    target.mkdir(exist_ok=True)
    # This prefix belongs exclusively to this generator. Preserve other files.
    for old in target.glob("benchkit-*"):
        if old.suffix in {".png", ".svg"} and old.is_file():
            old.unlink()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [], ["Figures omitted: matplotlib is unavailable. CSV and Markdown outputs remain complete."]
    figures, warnings = [], []

    def save(fig, name, caption):
        filename = "benchkit-" + _slug(name)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(target / f"{filename}.{suffix}", dpi=160)
        plt.close(fig)
        figures.append({"caption": caption, "png": f"figures/{filename}.png", "svg": f"figures/{filename}.svg"})

    passed = _comparison_runs(runs)
    groups = defaultdict(list)
    for run in passed:
        if _positive(run["tokens_per_s"]):
            groups[(run["hardware"], run["gpu_models"], run["mode"])].append(run)
    for (hardware, models, mode), group in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(10, max(3, len(group) * .55)))
        labels = [f"{run['run_id']} | {str(run['model']).split('/')[-1]} | {run['gpus']} GPUs" for run in group]
        ax.barh(range(len(group)), [run["tokens_per_s"] for run in group], color="#34699a")
        ax.set_yticks(range(len(group)), labels)
        ax.set_xlabel("Measured input + response tokens/s, all GPUs" if mode == "grpo" else "Measured unpadded input tokens/s, all GPUs")
        ax.set_title(f"{models}: {mode} throughput by run")
        ax.grid(axis="x", alpha=.25)
        save(fig, f"throughput-{models}-{mode}", f"{hardware} {mode}: each bar is one completed run. Configurations remain separate.")
        relevant = [row for row in stages if row["run_id"] in {run["run_id"] for run in group}
                    and _positive(row["mean_s"]) and row["stage"] not in {"timing_s/step", "timing_s/train_step"}]
        if relevant:
            fig, ax = plt.subplots(figsize=(10, max(3, len(relevant) * .35)))
            ax.barh(range(len(relevant)), [row["mean_s"] for row in relevant], color="#4b8b70")
            ax.set_yticks(range(len(relevant)), [f"{row['run_id']} | {row['stage'].removeprefix('timing_s/')}" for row in relevant])
            ax.set_xlabel("Native timer mean (seconds)")
            ax.set_title(f"{models}: {mode} native stage timers (can overlap)")
            save(fig, f"stages-{models}-{mode}", f"{hardware} {mode}: stage timers are not additive and may overlap.")
    for run in passed:
        try:
            rows = read_metrics(run["directory"] / "metrics.jsonl")
        except (OSError, ValueError, TypeError) as error:
            warnings.append(f"Trace omitted for {run['run_id']}: {type(error).__name__}")
            continue
        mode = run["mode"]
        timing = "timing_s/train_step" if mode == "sft" else "timing_s/step"
        token = "train/global_tokens" if mode == "sft" else "perf/total_num_tokens"
        loss = "train/loss" if mode == "sft" else "actor/pg_loss"
        train = [row for row in rows if timing in row or loss in row]
        measured = train[int(run["warmup_steps"] or 0):]
        losses = [(row["step"], row[loss]) for row in measured if finite(row.get(loss))]
        rates = [(row["step"], row[token] / row[timing]) for row in measured
                 if _positive(row.get(timing)) and _positive(row.get(token))]
        if not losses and not rates:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        if losses:
            axes[0].plot(*zip(*losses), marker=".")
        if rates:
            axes[1].plot(*zip(*rates), marker=".")
        for ax, title in zip(axes, (loss, "Measured tokens/s (all GPUs)")):
            ax.set_title(title)
            ax.set_xlabel("Training step, warmup excluded")
            ax.grid(alpha=.25)
        fig.suptitle(f"{run['hardware']} | {run['run_id']}")
        save(fig, f"trace-{run['run_id']}", f"{run['run_id']}: observed step traces after warmup exclusion.")
    by_signature = defaultdict(list)
    for row in scaling:
        by_signature[row["group_id"]].append(row)
    for identity, rows in sorted(by_signature.items()):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].plot([row["gpus"] for row in rows], [row["mean_tokens_per_s"] for row in rows], marker="o")
        axes[1].plot([row["gpus"] for row in rows], [row["efficiency"] for row in rows], marker="o")
        axes[0].set_ylabel("Mean measured tokens/s (all GPUs)")
        axes[1].set_ylabel("Strong-scaling efficiency")
        for ax in axes:
            ax.set_xlabel("Allocated GPUs")
            ax.set_xticks([row["gpus"] for row in rows])
            ax.grid(alpha=.25)
        fig.suptitle(f"{rows[0]['hardware']} | {rows[0]['mode']} | signature {identity}")
        save(fig, f"scaling-{identity}", f"Matched workload {identity}: all completed, non-excluded repetitions contribute equally. No best-run selection.")
    return figures, warnings


def generate(input_root: Path, output_root: Path):
    """Write tables for every attempt and plots for qualifying measured runs."""
    input_root, output_root = Path(input_root).resolve(), Path(output_root).resolve()
    if not input_root.is_dir():
        raise ValueError(f"Input directory does not exist: {input_root}")
    runs = [_load_run(directory, input_root) for directory in _discover(input_root)]
    comparison_runs = _comparison_runs(runs)
    stages, scaling = _stage_rows(runs), _scaling_rows(runs)
    scaled_groups = {row["group_id"] for row in scaling}
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = _artifact_rows(runs, output_root)
    _write_csv(output_root / "measurements.csv", MEASUREMENT_FIELDS, runs)
    _write_csv(output_root / "stages.csv", STAGE_FIELDS, stages)
    _write_csv(output_root / "scaling.csv", SCALING_FIELDS, scaling)
    _write_csv(output_root / "artifacts.csv", ARTIFACT_FIELDS, artifacts)
    artifact_sections = [
        "# Raw artifact index",
        "This index includes every discovered attempt. File presence does not establish complete capture or successful training. "
        "Bytes are logical sizes of regular files. Symlink targets are references and are not read or counted as retained bytes. "
        "Recorded SHA256 values come from artifact_inventory.json when available. This report does not rehash payloads. "
        "Native capture manifests retain their recorded hashes and timing scopes. "
        "Links require the report and result directories to keep their relative layout.",
    ]
    for run in runs:
        artifact_sections += ["## " + _display(run["run_id"]),
            _table(["Artifact", "Category", "Type", "Logical bytes", "Recorded SHA256", "External reference", "Index error"],
                   [[f"[{row['path']}]({row['link']})", row["category"], row["file_type"], row["size_bytes"],
                     row["recorded_sha256"], row["external_reference"], row["error"]] for row in artifacts if row["run_id"] == run["run_id"]])]
    (output_root / "artifacts.md").write_text("\n\n".join(artifact_sections) + "\n", encoding="utf-8")
    figures, warnings = _figures(comparison_runs, stages, scaling, output_root)
    sections = [
        "# Performance and scalability evidence",
        "This report contains recorded observations and arithmetic derived from those observations. "
        "It does not extrapolate H100 measurements to GH200 or unmeasured cluster sizes.",
        f"Discovered attempts: {len(runs)}. Completed measurements: {sum(run['status'] == 'passed' for run in runs)}. "
        f"Review exclusions: {sum(run['comparison_excluded'] for run in runs)}. "
        "Failed, running, incomplete, and review-excluded attempts remain in the inventory. "
        "They do not contribute to plots or scaling comparisons. "
        "A review exclusion preserves the recorded execution status and original observations; its reason appears in Table 6.",
        "## Table 1. Environment and evidence",
        _table(["Run", "Status", "Evidence label", "Observed GPU models", "Allocated GPUs / nodes", "GPU inventory memory", "Driver", "verl commit"],
               [[r["run_id"], r["status"], r["evidence_class"], r["gpu_models"], f"{_display(r['gpus'])} / {_display(r['nodes'])}", r["gpu_memory"], r["driver"], r["verl_commit"]] for r in runs]),
        "GPU model labels use the CUDA-visible device probe when available, otherwise the recorded nvidia-smi inventory. "
        "GPU memory describes the nvidia-smi inventory. Allocation counts come from the run configuration, not inventory length. "
        "A6000 and H100 observations appear in separate figures. An evidence label does not override the recorded GPU model.",
        _table(["Run", "Framework versions", "Benchmark commit", "Source tree SHA256", "Source changes"],
               [[r["run_id"], r["versions"], r["environment"].get("git", {}).get("stdout"), r["source_tree_sha256"], r["environment"].get("git_dirty", {}).get("stdout") or "none recorded"] for r in runs]),
        _table(["Run", "CPU", "Logical CPUs", "Host RAM (GiB)", "CUDA runtime", "Framework lane", "Topology captured"],
               [[r["run_id"], *_host_environment(r["environment"]), r["cuda_runtime"], r["framework"].get("lane"),
                 bool(r["environment"].get("topology", {}).get("stdout"))] for r in runs]),
        "## Table 2. Workload configuration",
        _table(["Run", "Model / revision", "Mode", "Sequence limit", "Global / micro batch", "Prompt / response limits", "Rollout group", "dtype", "Measured / warmup steps"],
               [[r["run_id"], f"{r['model'] or 'missing'} @ {r['model_revision'] or 'missing'}", r["mode"], r["config"].get("seq_length"),
                 f"{_display(r['config'].get('global_batch_size'))} / {_display(r['config'].get('micro_batch_size'))}",
                 f"{_display(r['config'].get('prompt_length'))} / {_display(r['config'].get('response_length'))}", r["config"].get("rollout_n"), r["config"].get("dtype"),
                 f"{_display(r['measured_steps'])} / {_display(r['warmup_steps'])}"] for r in runs]),
        "Sequence settings are limits, not observed token counts. SFT uses trajectory-derived supervision. "
        "GRPO uses generated responses and bounded code-output rewards. It does not include repository edits, shell tools, or SWE-bench evaluation.",
        "## Table 3. Measured performance",
        _table(["Run", "Status", "Comparison excluded", "Tokens/s, all GPUs", "Tokens/s/GPU", "Step mean / median / p95 (s)", "Native peak allocated (GiB)", "Tokens observed", "Time observed (s)"],
               [[r["run_id"], r["status"], r["comparison_excluded"], r["tokens_per_s"], r["tokens_per_gpu_s"],
                 " / ".join(_display(r[key]) for key in ("step_time_mean_s", "step_time_median_s", "step_time_p95_s")),
                 r["peak_memory_gib"], r["measured_tokens"], r["measured_time_s"]] for r in runs]),
        "Token counts are already aggregated across all ranks. Tokens/s/GPU divides aggregate tokens/s by total allocated GPUs exactly once. "
        "SFT counts unpadded input tokens, including masked prompts. GRPO counts prompt plus generated response tokens. "
        "These values are not response-only generation rates or supervised-target-only token rates. "
        "Missing values remain empty in CSV and appear as — here. Partial failed-run values are diagnostic observations, not completed benchmark results.",
        "## Proxy response costs",
        _table(["Run", "Status", "Observed response samples", "Responses/s", "GPU-seconds/response"],
               [[r["run_id"], r["status"], r["response_samples"], r["responses_per_s"], r["gpu_seconds_per_response"]]
                for r in runs if r["mode"] == "grpo"]),
        "Response metrics use the actual rollout-dump sample count after warmup and the measured total GRPO step time. "
        "They include the bounded proxy iteration, not generation time alone. A response is one sampled code-output answer, not a multi-turn agent trajectory. "
        "The report leaves these metrics missing when the summary does not establish them.",
        "## GPU device telemetry",
        _table(["Run", "Status", "GPU UUID", "Recorded interval", "Samples: utilization / memory / power", "Mean utilization (%)", "Sampled peak memory (MiB)", "Mean power (W)"],
               _telemetry_rows(runs)),
        "Native peak allocated memory comes from framework allocator statistics. Device telemetry comes from periodic nvidia-smi samples. "
        "Device memory includes allocations outside that allocator, and sampling can miss short peaks. "
        "Telemetry covers the complete framework process, including initialization and warmup. It does not share the post-warmup timing interval. "
        "The per-device statistics and their scope remain in measurements.csv and summary.json. Missing telemetry is not zero utilization.",
        "## Training and reward evidence",
        _table(["Run", "Mode", "Gradient samples / mean / max", "Nonzero gradient observed", "Reward signal observed", "Groups with reward variation", "Proxy accuracy", "Warnings"],
               [[r["run_id"], r["mode"], " / ".join(_display(r[key]) for key in ("gradient_norm_n", "gradient_norm_mean", "gradient_norm_max")),
                 r["nonzero_gradient_observed"], r["reward_learning_signal_detected"], r["reward_group_nonzero_std_fraction"],
                 r["reward_proxy_accuracy"], r["warnings"]] for r in runs]),
        "A passed status establishes completed measurement execution. It does not establish convergence or model improvement. "
        "Zero or missing gradient evidence and zero or unverified reward variation limit the result to measured execution. "
        "Proxy accuracy measures code-output answers only. It is not a repository-repair score. "
        "Gradient sample counts preserve the recorded interval; older captures can include warmup. "
        "The full reward audit, including malformed rows and incomplete groups, remains in measurements.csv and summary.json.",
        "## Table 4. Native stage timings",
        _table(["Run", "Status", "Timer", "Observations", "Mean (s)", "Median (s)", "p95 (s)"],
               [[r[key] for key in ("run_id", "status", "stage", "n", "mean_s", "median_s", "p95_s")] for r in stages]),
        "Native timers may overlap or contain other timers. Do not sum them to infer total RL time. "
        "Only the native total-step timer measures the bounded proxy iteration. "
        "Nested agent-loop fields are omitted because that namespace also contains counts without a seconds contract.",
        "## Table 5. Matched strong scaling",
    ]
    if scaling:
        sections += [_table(["Group", "Hardware", "Mode", "GPUs", "Nodes", "Repetitions", "Mean tokens/s", "Sample SD", "Speedup", "Efficiency", "Runs"],
                            [[r[key] for key in ("group_id", "hardware", "mode", "gpus", "nodes", "replicates", "mean_tokens_per_s", "std_tokens_per_s", "speedup", "efficiency", "run_ids")] for r in scaling])]
    else:
        sections.append("No qualifying matched measurements span two GPU counts. No scaling curve or efficiency is inferred.")
    sections += [
        "Scaling groups require the same model revision, GPU model, framework versions, verl commit, benchmark source, dataset content, and workload configuration. "
        "Benchmark source identity uses the captured source-tree SHA256, or a clean Git commit when no tree hash exists. "
        "Global batch, sequence lengths, rollout group, seed, precision, microbatch, and measured-step count must match. "
        "Each GPU count uses the arithmetic mean of all completed, non-excluded repetitions in that group. No best-run selection occurs. "
        "Efficiency = throughput(N) / throughput(N0) × N0 / N. A single observation has no estimated sample deviation.",
        "## Table 6. Failures, omissions, and comparison limits",
        _table(["Run", "Status", "Comparison excluded", "Review reason", "Problems", "Scaling exclusion"],
               [[r["run_id"], r["status"], r["comparison_excluded"], r["comparison_exclusion_reason"], r["problems"], r["scaling_exclusion"] or
                 ("included in matched scaling" if r["scaling_signature"] and r["scaling_signature"][:12] in scaled_groups
                  else "signature recorded; no matching different GPU count")] for r in runs]),
        "## Capture policy and retained artifacts",
        _table(["Run", "Capture enabled / cadence", "Checkpoint enabled / cadence / keep all", "Indexed files / logical bytes", "Symlinks", "Capture files", "Checkpoint files", "Post-run hashing (s)", "Index / capture errors"],
               [[r["run_id"], " / ".join(_display(r["config"].get(key)) for key in ("artifact_capture", "capture_every_step")),
                 " / ".join(_display(r["config"].get(key)) for key in ("save_checkpoint", "checkpoint_every_steps", "checkpoint_keep_all")),
                 f"{r['artifact_inventory']['files']} / {r['artifact_inventory']['logical_bytes']}",
                 r["artifact_inventory"]["symlinks"], r["artifact_inventory"]["capture_files"] or "none observed",
                 r["artifact_inventory"]["checkpoint_files"] or "none observed", r["artifact_inventory"]["hash_elapsed_s"],
                 f"{r['artifact_inventory']['errors']} / {_display(r['artifact_inventory']['capture_inventory_errors'])}"] for r in runs]),
        "[Raw artifact index with file links](artifacts.md) · [Artifact inventory CSV](artifacts.csv). "
        "Capture and checkpoint policies are requested configuration, not proof of successful capture. "
        "The inventory records files present at report generation, including failed and review-excluded attempts. "
        "Logical file bytes do not measure physical filesystem usage. Symlink targets are not added to the byte total. "
        "Native capture manifests record per-process capture duration and payload hashes when available. "
        "Their elapsed_s covers CPU copy, payload serialization, and hashing, excluding TransferQueue materialization and metadata/manifest writes. "
        "It is not total capture overhead. "
        "Absent timing remains unknown. Do not add concurrent capture durations or subtract them from native step timing. "
        "Checkpoint presence does not establish resume support or model convergence.",
        "## Figures",
    ]
    for figure in figures:
        sections += [f"![{figure['caption']}]({figure['png']})", f"{figure['caption']} [SVG]({figure['svg']})"]
    if not figures:
        sections.append("No figures were generated. See the tables and generation notes for the available evidence.")
    if warnings:
        sections += ["## Generation notes", "\n".join("- " + _display(warning) for warning in warnings)]
    sections += [
        "## Machine-readable outputs",
        "[Measurements](measurements.csv), [native stage timings](stages.csv), and [matched scaling](scaling.csv). "
        "[Raw artifacts](artifacts.md) and [artifact inventory](artifacts.csv) retain links to intermediate evidence. "
        "The original per-run configuration, environment, manifest, summary, metrics, and framework logs remain the primary evidence. "
        "An optional review.json records an additive comparison exclusion with exclude_from_comparisons: true and a reason string. "
        "It does not edit the captured evidence.",
    ]
    (output_root / "report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return {"runs": len(runs), "passed": sum(r["status"] == "passed" for r in runs),
            "comparison_excluded": sum(r["comparison_excluded"] for r in runs),
            "comparison_eligible": len(comparison_runs),
            "scaling_rows": len(scaling), "figures": len(figures), "warnings": warnings}
