#!/bin/bash
# 키트 동작 검증(연구실 A6000 2장, 같은 구조의 소형 모델 Qwen3.5-2B). H100에서는 쓰지 않는다.
#SBATCH --job-name=h100bench-test
#SBATCH --partition=a6000
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0-01:30:00
#SBATCH --output=test_%j.out
set -x
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository directory}"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONUNBUFFERED=1
M=${M:-Qwen/Qwen3.5-2B}; O=results_test
python -c "from transformers.utils.import_utils import is_flash_linear_attention_available as f; print('fla', f())"
torchrun --standalone --nproc_per_node 2 bench_train.py --model $M --mode sft --seq 4096 --steps 4 --warmup 1 --out $O
torchrun --standalone --nproc_per_node 2 bench_train.py --model $M --mode grpo --ref --seq 4096 --steps 3 --warmup 1 --measure-sync --out $O
torchrun --standalone --nproc_per_node 2 bench_train.py --model $M --mode sft --init pretrained --seq 4096 --steps 3 --warmup 1 --out $O --tag _pre
MODEL=$M MAX_LEN=32768 GPU_UTIL=0.85 bash serve_vllm.sh 2 8100 && \
  python bench_rollout.py --model $M --ports 8100:2 --concurrency 16 --duration 90 --warmup 20 --out $O
bash serve_vllm.sh stop
python report.py $O
