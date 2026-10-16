"""Normalize native verl JSONL without substituting configured token counts."""
import json
import csv
import math
import statistics
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_metrics(path):
    """Merge multiple emissions at one training step; ignore validation-only rows."""
    steps = {}
    if not Path(path).exists():
        return []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("step"), int) or not isinstance(row.get("data"), dict):
            raise ValueError(f"Invalid native metric row {number}")
        steps.setdefault(row["step"], {}).update(row["data"])
    return [{"step": step, **values} for step, values in sorted(steps.items())]


def stats(values):
    values = sorted(v for v in values if finite(v))
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": values[max(0, math.ceil(.95 * len(values)) - 1)],
            "min": values[0], "max": values[-1]}


def gpu_telemetry(path):
    """Summarize UUID-filtered nvidia-smi samples, including initialization."""
    samples = {}
    if not Path(path).exists():
        return {"scope": "allocated GPUs, complete framework subprocess lifetime", "devices": {}}
    with Path(path).open() as source:
        for row in csv.reader(source):
            if len(row) != 5 or row[1].strip() == "uuid":
                continue
            uuid = row[1].strip()
            values = samples.setdefault(uuid, {"utilization_pct": [], "memory_used_mib": [], "power_w": []})
            for key, value in zip(values, row[2:]):
                try:
                    number = float(value.strip().split()[0])
                except (ValueError, IndexError):
                    continue
                if finite(number):
                    values[key].append(number)
    return {"scope": "allocated GPUs, complete framework subprocess lifetime",
            "devices": {uuid: {key: stats(values) for key, values in fields.items()} for uuid, fields in samples.items()}}


def summarize(config, rows, returncode, elapsed_s, gpu_samples=None):
    mode = config["mode"]
    time_key = "timing_s/train_step" if mode == "sft" else "timing_s/step"
    token_key = "train/global_tokens" if mode == "sft" else "perf/total_num_tokens"
    loss_key = "train/loss" if mode == "sft" else "actor/pg_loss"
    train = [r for r in rows if time_key in r or loss_key in r]
    measured = train[config["warmup_steps"]:]
    problems = []
    if returncode != 0:
        problems.append(f"framework_exit_code={returncode}")
    if len(train) < config["steps"]:
        problems.append(f"incomplete_steps={len(train)}/{config['steps']}")
    if not measured:
        problems.append("no_measured_steps_after_warmup")
    for r in train:
        for k in (time_key, token_key, loss_key):
            if not finite(r.get(k)) or (k != loss_key and r[k] <= 0):
                problems.append(f"step={r['step']}:invalid_or_missing:{k}")
    valid = [r for r in measured if finite(r.get(time_key)) and r[time_key] > 0
             and finite(r.get(token_key)) and r[token_key] > 0]
    duration = sum(r[time_key] for r in valid)
    tokens = sum(r[token_key] for r in valid)
    world = config["gpus"] * config.get("nodes", 1)
    throughput = tokens / duration if duration else None
    # Native nested agent_loop diagnostics also put token lengths and counts
    # under timing_s/. Only top-level timer entries have the seconds contract.
    stages = {k: stats([r.get(k) for r in measured])
              for k in sorted({k for r in measured for k in r if k.startswith("timing_s/")
                               and "/" not in k.removeprefix("timing_s/")})}
    memory_key = "perf/max_memory_allocated_gb" if mode == "sft" else "actor/perf/max_memory_allocated_gb"
    memory = stats([r.get(memory_key) for r in train])
    grad_key = "train/grad_norm" if mode == "sft" else "actor/grad_norm"
    gradients = stats([r.get(grad_key) for r in measured])
    mfu_key = "train/mfu" if mode == "sft" else "perf/mfu/actor"
    mfu = stats([r.get(mfu_key) for r in measured])
    return {
        "schema_version": 1, "name": config["name"], "mode": mode,
        "model": config.get("requested_model", config["model"]),
        "model_revision": config.get("resolved_model_revision", config.get("model_revision")),
        "status": "passed" if not problems else "failed", "problems": problems,
        "training_steps": len(train), "measured_steps": len(measured),
        "warmup_steps": config["warmup_steps"], "gpus": world,
        "nodes": config.get("nodes", 1), "config": config,
        "wall_time_s": elapsed_s, "measured_time_s": duration,
        "measured_tokens": tokens, "tokens_per_s": throughput,
        "tokens_per_gpu_s": throughput / world if throughput else None,
        "gpu_hours_per_billion_tokens": world * 1e9 / throughput / 3600 if throughput else None,
        "gpu_hours_wall": elapsed_s * world / 3600,
        "step_time_s": stats([r.get(time_key) for r in measured]), "stages": stages,
        "loss": stats([r.get(loss_key) for r in measured]),
        "gradient_norm": gradients,
        "nonzero_gradient_observed": bool(gradients and gradients["max"] > 0),
        "peak_memory_gib": memory["max"] if memory else None,
        "peak_memory_source": memory_key,
        "framework_mfu": mfu, "telemetry": gpu_samples or {},
        "measurement_scope": "SFT training step" if mode == "sft" else "verl GRPO iteration with code-output reward",
        "token_scope": "unpadded input tokens (including masked prompt tokens)" if mode == "sft" else "prompt plus generated response tokens",
        "evidence_class": config.get("evidence_class", "development"),
    }
