#!/usr/bin/env bash
# 2노드 이상 학습 확장성 (SFT·GRPO). 모든 노드에서 같은 명령을 NODE_RANK만 바꿔 실행.
#   노드0: NNODES=2 NODE_RANK=0 MASTER_ADDR=<노드0 IP> bash run_multinode.sh
#   노드1: NNODES=2 NODE_RANK=1 MASTER_ADDR=<노드0 IP> bash run_multinode.sh
# 노드 간 InfiniBand/EFA가 있어야 의미 있는 수치가 나온다(NCCL_DEBUG=INFO 로그에서 NET/IB 확인).
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source "${VENV:-$PWD/.venv}/bin/activate"
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
NNODES=${NNODES:?}; NODE_RANK=${NODE_RANK:?}; MASTER_ADDR=${MASTER_ADDR:?}
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
mkdir -p results/logs
run() {
  torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port ${MASTER_PORT:-29500} \
    --nproc_per_node "$NPROC" bench_train.py "$@" 2>&1 | tee -a "results/logs/multinode_rank$NODE_RANK.log" | grep -E "결과|step |rror" || true
}
for MODE in ${MODES:-sft grpo}; do
  if [ "$MODE" = sft ]; then run --model "$MODEL" --mode sft --seq 16384 --steps 12 --warmup 3
  else run --model "$MODEL" --mode grpo --ref --seq 16384 --steps 8 --warmup 2 --measure-sync; fi
done
[ "$NODE_RANK" = 0 ] && echo "노드0 results/ 에 결과 저장됨 → python report.py"
