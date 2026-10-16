#!/usr/bin/env bash
# H100 단일 노드(8장) 전체 측정 → results/*.json → report.py 로 레포트 생성.
#   rollout : 롤아웃(에이전트 추론) GPU 1·2·4·8장, GPU당 vLLM 1개(TP1)       ~60분
#   sft     : CoHA-Coder-27B 기반 모델 SFT, 8장 FSDP2, 시퀀스 16K·32K          ~20분
#   grpo    : RL 정책 업데이트(이전 정책·참조 모델 포함), 8장 + 가중치 수집 시간  ~20분
#   scale   : 학습 확장성(9B 모델) GPU 2·4·8장                                ~20분
# usage: bash run_all.sh                 (일부만: PHASES="sft grpo" bash run_all.sh)
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source "${VENV:-$PWD/.venv}/bin/activate"
export MODEL=${MODEL:-Qwen/Qwen3.8-27B}
SCALE_MODEL=${SCALE_MODEL:-Qwen/Qwen3.5-9B}
PHASES=${PHASES:-"env rollout sft grpo scale"}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
DUR=${DUR:-300}; WARM=${WARM:-60}
mkdir -p results/logs
LOG=results/logs/run_all.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
train() {  # train <nproc> <args...>
  local n=$1; shift
  torchrun --standalone --nproc_per_node "$n" bench_train.py "$@" 2>&1 | tee -a "$LOG" | grep -E "결과|step |Error|error" || true
}

for ph in $PHASES; do
  case $ph in
  env)
    say "환경 기록"
    { nvidia-smi; nvidia-smi topo -m; python -c "import torch,vllm,transformers;print(torch.__version__,vllm.__version__,transformers.__version__)"; } \
      > results/env.txt 2>&1 ;;
  rollout)
    for G in 1 2 4 8; do
      [ "$G" -le "$NGPU" ] || continue
      say "롤아웃: GPU $G장"
      bash serve_vllm.sh "$G" 8100 >> "$LOG" 2>&1 || { say "서버 실패"; bash serve_vllm.sh stop; continue; }
      if [ "$G" = 1 ]; then CS="16 32 64"; else CS=$((G * ${PER_GPU_CONC:-32})); fi   # 1장에서 동시 수 포화점 확인
      for C in $CS; do
        python bench_rollout.py --model "$MODEL" --ports 8100:$G --concurrency "$C" --duration "$DUR" --warmup "$WARM" \
          2>&1 | tee -a "$LOG" | grep -E "결과|오류" || true
      done
      bash serve_vllm.sh stop >> "$LOG" 2>&1
    done ;;
  sft)
    say "SFT: $MODEL, $NGPU장"
    train "$NGPU" --model "$MODEL" --mode sft --seq 16384 --steps 12 --warmup 3
    train "$NGPU" --model "$MODEL" --mode sft --seq 32768 --steps 8 --warmup 2 ;;
  grpo)
    say "GRPO 정책 업데이트: $MODEL, $NGPU장"
    train "$NGPU" --model "$MODEL" --mode grpo --ref --seq 16384 --steps 8 --warmup 2 --measure-sync ;;
  scale)
    for G in 2 4 8; do
      [ "$G" -le "$NGPU" ] || continue
      say "학습 확장성: $SCALE_MODEL, $G장"
      train "$G" --model "$SCALE_MODEL" --mode sft --seq 16384 --steps 12 --warmup 3
    done ;;
  esac
done
say "완료 → python report.py"
python report.py | tee -a "$LOG"
