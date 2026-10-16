#!/usr/bin/env bash
# GPU마다 vLLM 서버 1개(TP1)를 띄운다. 6호기 계획과 같은 'GPU당 1 replica' 구성.
# usage: bash serve_vllm.sh <서버 수> [시작포트=8100]      중지: bash serve_vllm.sh stop
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
LOGDIR=${LOGDIR:-$PWD/results/logs}; mkdir -p "$LOGDIR"
if [ "${1:-}" = stop ]; then
  pkill -f "vllm serve" || true; sleep 10; pkill -9 -f "vllm serve" || true; exit 0
fi
N=${1:?서버 수}; PORT0=${2:-8100}
for ((i = 0; i < N; i++)); do
  CUDA_VISIBLE_DEVICES=$i nohup vllm serve "$MODEL" --port $((PORT0 + i)) --served-model-name bench \
    -tp 1 --max-model-len ${MAX_LEN:-65536} --gpu-memory-utilization ${GPU_UTIL:-0.92} \
    --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' ${VLLM_EXTRA:-} \
    > "$LOGDIR/vllm_gpu$i.log" 2>&1 &
done
for ((i = 0; i < N; i++)); do
  for _ in $(seq 1 180); do
    curl -sf -m 5 "http://127.0.0.1:$((PORT0 + i))/health" >/dev/null && break
    sleep 5
  done
  curl -sf -m 5 "http://127.0.0.1:$((PORT0 + i))/health" >/dev/null || { echo "서버 $i 기동 실패: $LOGDIR/vllm_gpu$i.log"; exit 1; }
done
echo "vLLM 서버 $N개 준비 (포트 $PORT0~$((PORT0 + N - 1)))"
