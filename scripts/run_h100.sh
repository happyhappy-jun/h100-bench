#!/usr/bin/env bash
# Run both models and modes on one allocated 8-GPU H100 server.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_PYTHON=${BENCH_PYTHON:-$PWD/.venv-verl-h100/bin/python}
REPORT_PYTHON=${REPORT_PYTHON:-$PWD/.venv-report/bin/python}
RUN_ROOT=${1:-results/h100-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ $# -gt 0 ]]; then shift; fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to reuse existing run root: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
failed=0
for config in h100_llama8b h100_llama8b_grpo h100_qwen27b h100_qwen27b_grpo; do
  if ! "$BENCH_PYTHON" -m benchkit.cli run --config "configs/$config.json" --output "$RUN_ROOT/$config" "$@"; then
    echo "FAILED: $config (evidence retained)" >&2
    failed=1
  fi
done
"$REPORT_PYTHON" -m benchkit.cli report --input "$RUN_ROOT" --output "$RUN_ROOT/report"
exit "$failed"
