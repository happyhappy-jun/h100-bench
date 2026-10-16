#!/usr/bin/env bash
# Submit from repository root; set partition, account, QoS and GPU count at sbatch.
# Example: sbatch --partition=a6000 --qos=YOUR_QOS --gres=gpu:2 \
#   --cpus-per-task=16 --mem=96G --time=01:00:00 scripts/slurm_bench.sbatch \
#   configs/a6000_smoke.json results/attempt001
#SBATCH --job-name=framework-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=bench-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
CONFIG=${1:?config path required}
RUN_DIR=${2:?unique output directory required}
BENCH_PYTHON=${BENCH_PYTHON:-$PWD/.venv-verl-a6000-dev/bin/python}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Driver owns its spawned process group. No pkill or global ray stop is used.
exec "$BENCH_PYTHON" -m benchkit.cli run --config "$CONFIG" --output "$RUN_DIR"
