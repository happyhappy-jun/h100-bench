"""Run one immutable benchmark case, preserve failures, and generate evidence."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .metrics import read_metrics, summarize, gpu_telemetry

ROOT = Path(__file__).resolve().parents[1]


def nvidia_device_ids(uuids):
    """PyTorch may return bare UUIDs; nvidia-smi requires a GPU-/MIG- prefix."""
    return [value if value.startswith(("GPU-", "MIG-")) else "GPU-" + value for value in uuids]


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def capture(command, timeout=20):
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def load_config(path, model=None, overrides=None):
    c = json.loads(Path(path).read_text())
    c.setdefault("nodes", 1)
    c.setdefault("warmup_steps", 1)
    c.setdefault("seed", 42)
    c.setdefault("dtype", "bfloat16")
    c.setdefault("strategy", "fsdp2")
    c.setdefault("model_revision", "main")
    c.setdefault("timeout_s", 14400)
    c.setdefault("artifact_capture", True)
    c.setdefault("capture_every_step", 1)
    c.setdefault("save_checkpoint", True)
    c.setdefault("checkpoint_every_steps", 1)
    c.setdefault("checkpoint_keep_all", True)
    if model:
        c["model"] = model
        c["model_override"] = True
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep or key not in c:
            raise ValueError(f"Override must name an existing top-level config field: {item}")
        c[key] = json.loads(value)
    if c.get("mode") not in ("sft", "grpo"):
        raise ValueError("mode must be sft or grpo")
    if c["dtype"] != "bfloat16" or c["strategy"] != "fsdp2":
        raise ValueError("This lane supports dtype=bfloat16 and strategy=fsdp2 only")
    if c.get("extra_overrides"):
        raise ValueError("Raw native overrides are disabled: add an explicit recorded config field instead")
    for key in ("gpus", "nodes", "global_batch_size", "micro_batch_size", "steps", "seq_length"):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("capture_every_step", "checkpoint_every_steps"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("artifact_capture", "save_checkpoint", "checkpoint_keep_all"):
        if type(c[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if not 0 <= c["warmup_steps"] < c["steps"]:
        raise ValueError("Require 0 <= warmup_steps < steps")
    if c["global_batch_size"] % (c["gpus"] * c["nodes"] * c["micro_batch_size"]):
        raise ValueError("global_batch_size must divide across all GPUs and micro batches")
    if c["mode"] == "grpo":
        if c.get("rollout_n", 0) < 2:
            raise ValueError("GRPO requires rollout_n >= 2")
        if c["gpus"] % c.get("rollout_tp", 1):
            raise ValueError("rollout_tp must divide GPUs per node")
        if c["prompt_length"] + c["response_length"] > c["seq_length"]:
            raise ValueError("prompt plus response lengths exceed seq_length")
    data = Path(c.get("input_data", "data/sessions.jsonl"))
    c["input_data"] = str((ROOT / data).resolve() if not data.is_absolute() else data.resolve())
    return c


def environment():
    versions = {}
    for name in ("torch", "transformers", "vllm", "verl", "ray", "flash-attn", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    memory = capture(["free", "-b"])
    cpu = capture(["lscpu"])
    source_files = sorted(p for directory in ("benchkit", "configs", "scripts", "templates")
                          for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    tree_hash = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "platform": platform.platform(), "versions": versions,
            "source_tree_sha256": tree_hash, "source_files_sha256": source_hashes,
            "host_memory": memory, "cpu": cpu,
            "git": capture(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "git_dirty": capture(["git", "-C", str(ROOT), "status", "--porcelain"]),
            "gpu": capture(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"]),
            "topology": capture(["nvidia-smi", "topo", "-m"]),
            "slurm": {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_JOB_QOS", "SLURM_JOB_NODELIST", "SLURM_NNODES") if k in os.environ}}


def doctor(config):
    from .verl_backend import validate_framework
    checks = {"environment": environment(), "framework": validate_framework()}
    probe = capture([sys.executable, "-c",
        "import torch,json; n=torch.cuda.device_count(); print(json.dumps({'available':torch.cuda.is_available(),'count':n,'devices':[torch.cuda.get_device_name(i) for i in range(n)],'uuids':[str(getattr(torch.cuda.get_device_properties(i),'uuid','')) for i in range(n)],'cuda_runtime':torch.version.cuda}))"], timeout=180)
    checks["cuda"] = probe
    problems = []
    framework = checks["framework"]
    if not framework["available"]:
        problems.append("missing_framework_packages=" + ",".join(framework["missing"]))
    if not framework["pinned_revision_match"]:
        problems.append("unverified_verl_revision")
    if not framework.get("source_clean"):
        problems.append("modified_or_unverified_verl_source")
    expected_lane = config.get("framework_lane")
    if expected_lane and framework["lane"] != expected_lane:
        problems.append(f"framework_lane_mismatch={framework['lane']} expected={expected_lane}")
    if probe.get("returncode") != 0:
        problems.append("torch_cuda_probe_failed")
    else:
        gpu = json.loads(probe["stdout"].splitlines()[-1])
        if not gpu["available"] or gpu["count"] < config["gpus"]:
            problems.append("insufficient_visible_gpus")
    if not Path(config["input_data"]).is_file():
        problems.append("missing_input_data")
    expected = config.get("expected_gpu")
    if expected and probe.get("returncode") == 0:
        if any(expected not in name for name in gpu["devices"][:config["gpus"]]):
            problems.append(f"allocated_gpu_mismatch={expected}")
    if config["nodes"] != 1:
        problems.append("multi_node_framework_launcher_not_implemented")
    checks["problems"] = problems
    return checks


def resolve_model(config):
    """Resolve a requested revision before execution and record the exact weights."""
    result = dict(config)
    requested = result["model"]
    result["requested_model"] = requested
    local = Path(requested).expanduser()
    if local.is_dir():
        resolved = local.resolve()
        # config/tokenizer hashes identify local metadata; local checkpoints may
        # contain modified weights, so never infer an HF commit from the name.
        result["local_model_metadata_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (resolved / "config.json", resolved / "tokenizer_config.json") if p.is_file()}
        result["resolved_model_revision"] = None
    else:
        from huggingface_hub import snapshot_download
        resolved = Path(snapshot_download(requested, revision=result.get("model_revision", "main"),
                            ignore_patterns=["*.pth", "*.bin", "*.gguf", "*.onnx", "original/*"]))
        result["resolved_model_revision"] = resolved.name
    result["model"] = str(resolved)
    if result.get("chat_template"):
        template = Path(result["chat_template"])
        if not template.is_absolute():
            template = ROOT / template
        result["chat_template"] = str(template.resolve())
        result["chat_template_sha256"] = hashlib.sha256(template.read_bytes()).hexdigest()
    return result


def run(config, output, dry_run=False):
    from .verl_backend import build_commands
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Use a new run ID; prior evidence is immutable.")
    if dry_run:
        planned = dict(config, _dry_run=True)
        print(json.dumps({"config": config, "output": str(output),
                          "note": "Execution resolves the pinned model revision to a local snapshot first.",
                          "commands": build_commands(planned, output)}, indent=2, ensure_ascii=False))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    manifest = {"state": "preparing", "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    code = 1
    telemetry = None
    telemetry_file = None
    host_monitor = None
    ray_temp = None
    active = None
    previous_term = signal.getsignal(signal.SIGTERM)
    def stop_attempt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, stop_attempt)
    try:
        from .run_evidence import HostMonitor, preserve_provenance
        preserve_provenance(ROOT, output)
        host_monitor = HostMonitor(output)
        host_monitor.start()
        checks = doctor(config)
        write_json(output / "environment.json", checks)
        if checks["problems"]:
            raise RuntimeError("Preflight failed: " + ", ".join(checks["problems"]))
        config = resolve_model(config)
        from .model_assets import prepare_model_assets
        config = prepare_model_assets(config, output)
        write_json(output / "config.json", config)
        from .prepare_data import prepare_data
        prepared = prepare_data(config, output)
        write_json(output / "data_paths.json", {k: str(v) for k, v in prepared.items()})
        commands = build_commands(config, output)
        write_json(output / "commands.json", commands)
        env = os.environ.copy()
        env.update({"VERL_FILE_LOGGER_PATH": str(output / "metrics.jsonl"),
                    "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1",
                    "RAY_DEDUP_LOGS": "0",
                    "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")})
        # Configuration must not redirect the evidence logger or Python imports.
        custom_env = config.get("environment", {})
        reserved = {"VERL_FILE_LOGGER_PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "RAY_TMPDIR"}
        if reserved.intersection(custom_env):
            raise ValueError("environment cannot override evidence logger or PYTHONPATH")
        env.update(custom_env)
        # Short path avoids Ray's UNIX socket path-length limit. Retain a copy
        # of session logs at completion; never touch another run's Ray session.
        if config["mode"] == "grpo":
            ray_temp = Path(tempfile.mkdtemp(prefix="bench-ray-"))
            env["RAY_TMPDIR"] = str(ray_temp)
            write_json(output / "runtime_logs.json", {"ray_tmpdir": str(ray_temp),
                       "ray_dedup_logs": env["RAY_DEDUP_LOGS"]})
        telemetry_file = (output / "gpu.csv").open("w")
        try:
            allocation = json.loads(checks["cuda"]["stdout"].splitlines()[-1])
            uuids = allocation.get("uuids", [])[:config["gpus"]]
            if len(uuids) == config["gpus"] and all(uuids):
                telemetry = subprocess.Popen(["nvidia-smi", "-i", ",".join(nvidia_device_ids(uuids)), "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw", "--format=csv", "-l", "1"], stdout=telemetry_file, stderr=subprocess.DEVNULL)
        except OSError:
            pass
        manifest["state"] = "running"
        write_json(output / "manifest.json", manifest)
        for index, command in enumerate(commands):
            print("Executing:", " ".join(command), flush=True)
            with (output / f"framework-{index}.log").open("w") as log:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                code = active.wait(timeout=config["timeout_s"])
                active = None
            if code:
                break
        rows = read_metrics(output / "metrics.jsonl")
        result = summarize(config, rows, code, time.monotonic() - started, gpu_telemetry(output / "gpu.csv"))
        if config["mode"] == "grpo":
            from .reward import audit_rollouts
            result["reward_audit"] = audit_rollouts(output, warmup_steps=config["warmup_steps"], rollout_n=config["rollout_n"])
            audit = result["reward_audit"]
            samples = audit.get("reward/samples", 0)
            if samples and not audit.get("reward/malformed_rows") and result["measured_time_s"] > 0:
                result["responses_per_s"] = samples / result["measured_time_s"]
                result["gpu_seconds_per_response"] = result["measured_time_s"] * result["gpus"] / samples
        result["warnings"] = []
        if not result["nonzero_gradient_observed"]:
            result["warnings"].append("No nonzero gradient recorded; throughput does not establish a useful model update.")
        if config["mode"] == "grpo" and not result["reward_audit"].get("reward/learning_signal_detected"):
            result["warnings"].append("No reward variation observed; GRPO learning signal is unverified.")
        write_json(output / "summary.json", result)
        manifest["state"] = result["status"]
        code = 0 if result["status"] == "passed" else 1
    except BaseException as exc:
        manifest["state"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        print(manifest["error"], file=sys.stderr)
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        if active and active.poll() is None:
            try:
                os.killpg(active.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                active.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(active.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                active.wait()
        if telemetry:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=10)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()
        if telemetry_file:
            telemetry_file.close()
        if host_monitor:
            host_monitor.close()
        if ray_temp:
            for source in ray_temp.glob("**/session_*/logs"):
                if source.parent.is_symlink():
                    continue
                try:
                    shutil.copytree(source, output / "ray_logs" / source.parent.name, symlinks=True)
                except OSError as exc:
                    manifest.setdefault("evidence_errors", []).append(f"Ray log snapshot: {exc}")
        manifest["training_state"] = manifest["state"]
        if manifest.get("evidence_errors"):
            manifest["state"] = "failed"
            code = 1
        manifest.update({"exit_code": code, "elapsed_s": time.monotonic() - started,
                         "artifact_inventory": "artifact_inventory.json",
                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        write_json(output / "manifest.json", manifest)
        from .run_evidence import inventory
        try:
            captured = inventory(output)
            if captured["errors"]:
                print("Artifact inventory has errors; inspect artifact_inventory.json", file=sys.stderr)
                code = 1
                manifest["state"] = "failed"
                manifest["exit_code"] = code
                manifest["evidence_errors"] = manifest.get("evidence_errors", []) + captured["errors"]
                write_json(output / "manifest.json", manifest)
                # Only the manifest changed after indexing; refresh its entry
                # without rereading terabytes of checkpoint payloads.
                payload = (output / "manifest.json").read_bytes()
                for item in captured["files"]:
                    if item["path"] == "manifest.json":
                        captured["logical_bytes"] += len(payload) - item["bytes"]
                        item.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                write_json(output / "artifact_inventory.json", captured)
        except OSError as exc:
            print(f"Artifact inventory failed: {exc}", file=sys.stderr)
            code = 1
            manifest.update(state="failed", exit_code=code)
            manifest.setdefault("evidence_errors", []).append(f"Artifact inventory: {exc}")
            write_json(output / "manifest.json", manifest)
    print(f"{manifest['state']}: {output}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "doctor"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--model", help="Explicit checkpoint path or HF ID; recorded in evidence")
        p.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
        if name == "run":
            p.add_argument("--output", required=True)
            p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "report":
        from .reporting import generate
        generate(Path(args.input), Path(args.output))
        return 0
    config = load_config(args.config, args.model, args.set)
    if args.action == "doctor":
        checks = doctor(config)
        print(json.dumps(checks, indent=2, ensure_ascii=False))
        return int(bool(checks["problems"]))
    return run(config, args.output, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
