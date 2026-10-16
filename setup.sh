#!/usr/bin/env bash
# H100 노드에서 1회: 가상환경 + 모델 내려받기 (약 20~30분, 대부분 모델 다운로드)
# usage: bash setup.sh            (HF_TOKEN 필요 시 export HF_TOKEN=...)
set -euo pipefail
cd "$(dirname "$0")"
VENV=${VENV:-$PWD/.venv}
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
SCALE_MODEL=${SCALE_MODEL:-Qwen/Qwen3.5-9B}
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

[ -x "$VENV/bin/python" ] || uv venv -p 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# pilot(A6000)과 같은 버전: vLLM 0.19 / transformers 5.5. Qwen3.5 계열 linear attention 학습 커널(fla) 추가
# vllm 0.19 메타데이터는 transformers<5를 요구하지만 Qwen3.5/3.8은 transformers 5.x가 필요 → override (pilot과 동일 조합)
echo "transformers==5.5.0" > /tmp/h100_bench_override.txt
uv pip install "vllm==0.19.0" "transformers==5.5.0" "flash-linear-attention" aiohttp matplotlib huggingface_hub \
  --override /tmp/h100_bench_override.txt
# causal-conv1d는 소스 빌드가 필요할 수 있음 → 실패해도 진행(없으면 PyTorch 구현 사용, 처리량 영향 작음)
uv pip install --no-build-isolation causal-conv1d 2>/dev/null || echo "causal-conv1d 생략"

python - <<'EOF'
import torch, transformers, vllm
from transformers.utils.import_utils import is_flash_linear_attention_available as fla, is_causal_conv1d_available as cc
print(f"torch {torch.__version__} / transformers {transformers.__version__} / vllm {vllm.__version__}")
print(f"GPU {torch.cuda.device_count()} × {torch.cuda.get_device_name()}  fla={fla()} causal_conv1d={cc()}")
assert fla(), "flash-linear-attention 미설치: Qwen3.5 계열 학습이 매우 느려짐"
EOF
for m in "$MODEL" ${SCALE_MODEL:+"$SCALE_MODEL"}; do
  hf download "$m" --exclude "*.pth" >/dev/null && echo "모델 준비: $m"
done
echo "setup 완료 → bash run_all.sh"
