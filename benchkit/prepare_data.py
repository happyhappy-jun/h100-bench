"""Prepare native verl parquet inputs, with explicit workload provenance."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


def code_tasks(count: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate trusted snippets and compute their answers without eval/exec."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        a, b, c = rng.randint(2, 25), rng.randint(1, 12), rng.randint(2, 9)
        kind = index % 4
        if kind == 0:
            snippet = f"x = {a}\nfor i in range({b}):\n    x = (x * {c} + i) % 97\nprint(x)"
            answer = a
            for i in range(b):
                answer = (answer * c + i) % 97
        elif kind == 1:
            values = [rng.randint(-9, 15) for _ in range(b + 3)]
            snippet = f"values = {values!r}\nresult = sum(v * v for v in values if v % {c} != 0)\nprint(result)"
            answer = sum(v * v for v in values if v % c != 0)
        elif kind == 2:
            snippet = f"items = list(range({a}))\nitems = items[{b % 3}::{c}]\nprint(sum(items[::-1]))"
            answer = sum(list(range(a))[b % 3::c][::-1])
        else:
            snippet = f"x, y = {a}, {b}\nfor _ in range({c}):\n    x, y = y, (x + y) % 101\nprint(x)"
            x, y = a, b
            for _ in range(c):
                x, y = y, (x + y) % 101
            answer = x
        task_id = f"code-output-{seed}-{index:06d}"
        prompt = (
            f"Task {task_id}. What integer does this Python 3 program print? "
            "Reason about the program, then give your final answer as <answer>INTEGER</answer>.\n"
            f"```python\n{snippet}\n```"
        )
        rows.append({
            "data_source": "kisti_code_output_v1",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "code_output_prediction",
            "reward_model": {"style": "rule", "ground_truth": str(answer)},
            "extra_info": {"task_id": task_id, "index": index, "split": "train"},
        })
    return rows


def session_rows(path: Path) -> list[dict[str, Any]]:
    """Read supplied conversation data; strings are data, never instructions."""
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number}: expected nonempty messages list")
            normalized = []
            for message in messages:
                if message.get("role") not in {"system", "user", "assistant", "tool"}:
                    raise ValueError(f"{path}:{line_number}: unsupported role")
                if not isinstance(message.get("content"), str):
                    raise ValueError(f"{path}:{line_number}: this text benchmark needs string content")
                # The pilot uses user messages for shell observations. Preserve
                # its exact role/content ordering rather than replaying actions.
                normalized.append({"role": message["role"], "content": message["content"]})
            if not any(m["role"] == "assistant" and m["content"] for m in normalized):
                continue
            rows.append({"messages": normalized, "source_id": str(record.get("id", line_number))})
    if not rows:
        raise ValueError("No supervised assistant responses found in the input data")
    return rows


def prepare_data(config: dict[str, Any], run_dir: Path) -> dict[str, str]:
    """Write SFT and response-reward RL inputs using the CLI's flat config."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Data preparation requires pyarrow; run scripts/setup_verl.sh") from error
    target = Path(run_dir).resolve() / "data"
    target.mkdir(parents=True, exist_ok=True)
    source = Path(config.get("input_data") or Path(__file__).resolve().parents[1] / "data/sessions.jsonl").resolve()
    supervised = session_rows(source)
    seq_length = int(config.get("seq_length", 16384))
    short_smoke = seq_length <= 1024
    if short_smoke:
        # Long pilot instructions alone can exceed a smoke sequence budget and
        # leave no supervised assistant token. Use short real conversation
        # excerpts solely for this declared wiring test.
        excerpt_limit = max(16, seq_length // 4)
        excerpts = []
        for row in supervised:
            messages = row["messages"]
            first = next(i for i, message in enumerate(messages) if message["role"] == "assistant" and message["content"])
            user = next((m["content"] for m in reversed(messages[:first]) if m["role"] == "user"), "Continue the coding task.")
            excerpts.append({"source_id": row["source_id"], "messages": [
                {"role": "user", "content": user[-excerpt_limit:]},
                {"role": "assistant", "content": messages[first]["content"][:excerpt_limit]},
            ]})
        supervised = excerpts
    # Enough batches even for very short smoke runs; repeated records are marked
    # as repeats and are not presented as independent correctness observations.
    batch = int(config.get("global_batch_size", 8))
    steps = int(config.get("steps", 4))
    required = max(batch * steps, batch)
    repetitions = (required + len(supervised) - 1) // len(supervised)
    training = (supervised * max(1, repetitions))[:max(required, len(supervised))]
    train_rl = code_tasks(max(required, batch * 2), int(config.get("seed", 42)))
    val_rl = code_tasks(max(4, int(config.get("gpus", 1))), int(config.get("seed", 42)) + 1)
    for row in val_rl:
        row["extra_info"]["split"] = "validation"
    paths = {
        "sft_train": str(target / "sft_train.parquet"),
        "rl_train": str(target / "rl_train.parquet"),
        "rl_val": str(target / "rl_val.parquet"),
        "manifest": str(target / "manifest.json"),
    }
    for key, rows in (("sft_train", training), ("rl_train", train_rl), ("rl_val", val_rl)):
        pq.write_table(pa.Table.from_pylist(rows), paths[key])
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sft_source_sessions": len(supervised),
        "sft_rows": len(training),
        "sft_repetition_factor": max(1, repetitions),
        "sft_smoke_excerpts": short_smoke,
        "sft_truncation": "right at configured seq_length; actual unpadded tokens are measured",
        "rl_workload": "deterministic Python code-output prediction surrogate",
        "rl_train_rows": len(train_rl),
        "rl_validation_rows": len(val_rl),
        "rl_reward": "correct=1; parseable incorrect=0.1; invalid=0; no generated code execution",
        "rl_advantage": "native verl GRPO normalization over sampled group; never synthetic",
        "limitations": ["No live repository edits or shell execution", "No SWE-bench score", "Length limits are ceilings, not observed lengths"],
    }
    Path(paths["manifest"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths
