"""Verifiable code-output reward for the framework throughput workload.

This module never executes model output. It compares the model's final integer
with the answer of a generated, trusted Python task. This is a coding-workload
surrogate, not SWE-bench or a repository-editing agent evaluation.
"""

from __future__ import annotations

import re
import time
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_answer(text: str) -> int | None:
    """Accept a final ``<answer>integer</answer>`` or ``\\boxed{integer}``."""
    # Thinking text may contain proposed answers; only use the last final tag.
    matches = list(re.finditer(r"<answer>\s*([+-]?\d+)\s*</answer>|\\boxed\{\s*([+-]?\d+)\s*\}", text))
    if not matches:
        return None
    raw = matches[-1].group(1) or matches[-1].group(2)
    # Bound integer parsing to prevent pathological generated strings.
    if len(raw) > 32:
        return None
    return int(raw)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """verl custom reward API, shared by v0.8.0 and v0.9.1.

Reward is 1 for a correct answer, 0.1 for a parseable wrong answer and 0
otherwise. The small format component is explicit and deterministic; GRPO
advantages are computed by verl from each group of generated responses.
    """
    started = time.perf_counter()
    if data_source != "kisti_code_output_v1":
        raise ValueError(f"Unexpected reward data source: {data_source}")
    answer = extract_answer(solution_str)
    expected = int(ground_truth)
    correct = answer == expected if answer is not None else False
    extra_info = extra_info or {}
    return {
        "score": 1.0 if correct else (0.1 if answer is not None else 0.0),
        "acc": float(correct),
        "format_valid": float(answer is not None),
        "reward_seconds": time.perf_counter() - started,
        "task_id": str(extra_info.get("task_id", "unknown")),
    }


def audit_rollouts(run_dir: Path, warmup_steps: int = 0, rollout_n: int | None = None) -> dict[str, Any]:
    """Audit native sampled GRPO groups; missing/constant rewards are explicit.

The per-prompt score spread is evidence for a nonzero reward-derived GRPO
advantage. It does not establish convergence or repository-solving accuracy.
    """
    groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    rows = 0
    bad_rows = 0
    correct = 0.0
    valid_format = 0.0
    for path in sorted((Path(run_dir) / "rollouts").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                step = int(row.get("step", path.stem))
                if step <= warmup_steps:
                    continue
                score = float(row["score"])
                if not math.isfinite(score):
                    raise ValueError("nonfinite score")
                task = str(row.get("task_id") or row["input"])
                groups[(step, task)].append(score)
                rows += 1
                correct += float(row.get("acc", score == 1.0))
                valid_format += float(row.get("format_valid", score > 0.0))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                bad_rows += 1
    complete = [scores for scores in groups.values() if len(scores) >= 2 and (rollout_n is None or len(scores) == rollout_n)]
    nonzero = sum(max(scores) - min(scores) > 1e-8 for scores in complete)
    return {
        "reward/groups": len(groups),
        "reward/complete_groups": len(complete),
        "reward/incomplete_groups": len(groups) - len(complete),
        "reward/group_nonzero_std_fraction": nonzero / len(complete) if complete else None,
        "reward/nonzero_std_groups": nonzero,
        "reward/samples": rows,
        "reward/malformed_rows": bad_rows,
        "reward/proxy_accuracy": correct / rows if rows else None,
        "reward/format_valid_fraction": valid_format / rows if rows else None,
        "reward/learning_signal_detected": nonzero > 0 and bad_rows == 0,
        "reward/scope": "code-output surrogate; not SWE-bench",
    }
