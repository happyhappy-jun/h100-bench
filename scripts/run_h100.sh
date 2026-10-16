#!/usr/bin/env bash
# Direct 8-GPU H100 campaign: 10, 100, then 1000 steps for both models/modes.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_PYTHON=${BENCH_PYTHON:-$PWD/.venv-verl-h100/bin/python}
REPORT_PYTHON=${REPORT_PYTHON:-$PWD/.venv-report/bin/python}
RUN_ROOT=${1:-results/h100-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ $# -gt 0 ]]; then shift; fi
exec "$BENCH_PYTHON" -m benchkit.campaign run --output "$RUN_ROOT" \
  --report-python "$REPORT_PYTHON" "$@"
