#!/usr/bin/env bash
# Pinned, isolated framework environments. No modification of the legacy .venv.
set -euo pipefail
cd "$(dirname "$0")/.."
LANE=${1:-h100}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
case "$LANE" in
  h100)
    REVISION=1876b06d0a3e4e71e06230be10af14492ca8a75b
    ;;
  a6000|a6000-dev)
    LANE=a6000-dev
    REVISION=7aed6b230776f963fa09509c10d9c3a767d1102c
    ;;
  *) echo "Usage: bash scripts/setup_verl.sh {h100|a6000-dev}" >&2; exit 2 ;;
esac
command -v uv >/dev/null || { echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ first" >&2; exit 2; }
command -v git >/dev/null || { echo "git is required" >&2; exit 2; }
VENDOR_DIR=${VERL_SOURCE_DIR:-$PWD/vendor/verl-$LANE}
VENV=${VERL_VENV:-$PWD/.venv-verl-$LANE}
REPORT_VENV=${BENCH_REPORT_VENV:-$PWD/.venv-report}
python3 - "$VENV" "$REPORT_VENV" <<'PY'
from pathlib import Path
import sys
if Path(sys.argv[1]).resolve() == Path(sys.argv[2]).resolve():
    raise SystemExit('Training and report environments must use different directories')
PY
if [[ ! -d "$VENDOR_DIR/.git" ]]; then
  git clone https://github.com/verl-project/verl.git "$VENDOR_DIR"
fi
if [[ -n "$(git -C "$VENDOR_DIR" status --porcelain)" ]]; then
  echo "Refusing to replace a modified framework checkout: $VENDOR_DIR" >&2
  exit 2
fi
git -C "$VENDOR_DIR" fetch origin "$REVISION"
git -C "$VENDOR_DIR" checkout --detach "$REVISION"
if [[ "$LANE" == h100 ]]; then
  # The official lock targets CUDA 13. NVIDIA's CUDA-13 minimum Linux driver
  # is 580.65.06. Check before downloading a multi-GB environment.
  python3 - <<'PY'
import subprocess
text = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True)
versions = [tuple(int(part) for part in line.strip().split('.')) for line in text.splitlines() if line.strip()]
if not versions or min(versions) < (580, 65, 6):
    raise SystemExit('Canonical verl v0.9.1 requires a CUDA-13-compatible driver >=580.65.06. Use a6000-dev only for declared compatibility testing on CUDA12.8.')
PY
  UV_PROJECT_ENVIRONMENT="$VENV" uv sync --project "$VENDOR_DIR" --python "$PYTHON_VERSION" --frozen --extra fsdp --extra vllm
else
  # Development compatibility lane, validated separately from H100. Qwen3.8
  # requires newer model support and is NOT supported by this old vLLM stack.
  [[ -x "$VENV/bin/python" ]] || uv venv --python "$PYTHON_VERSION" "$VENV"
  uv pip install --python "$VENV/bin/python" \
    'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0' \
    --index https://download.pytorch.org/whl/cu128
  uv pip install --python "$VENV/bin/python" \
    'vllm==0.10.2' 'transformers==4.56.2' 'ray[default]==2.49.2' \
    'tensordict==0.10.0' 'transferqueue==0.1.7' 'accelerate==1.10.1' \
    'peft==0.17.1' 'liger-kernel==0.6.2' 'numpy<2' \
    'pyarrow>=19' pandas datasets torchdata codetiming hydra-core \
    pylatexenc qwen-vl-utils wandb dill pybind11 tensorboard \
    latex2sympy2_extended math_verify orjson psutil
  # Prefer an explicitly supplied prebuilt wheel on development clusters.
  # A source build needs a matching CUDA toolkit and may take several minutes.
  if [[ -n "${FLASH_ATTN_WHEEL:-}" ]]; then
    uv pip install --python "$VENV/bin/python" "$FLASH_ATTN_WHEEL"
  else
    uv pip install --python "$VENV/bin/python" --no-build-isolation 'flash-attn==2.8.3'
  fi
  uv pip install --python "$VENV/bin/python" --no-deps -e "$VENDOR_DIR"
fi
"$VENV/bin/python" -m pip freeze > "$VENV/framework-freeze.txt" 2>/dev/null || uv pip freeze --python "$VENV/bin/python" > "$VENV/framework-freeze.txt"
"$VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
from verl.workers.engine_workers import TrainingWorker
for name in ['verl', 'torch', 'vllm', 'transformers', 'ray', 'flash-attn']:
    print(f'{name}=={metadata.version(name)}')
print('torch CUDA runtime:', torch.version.cuda)
print('GPU initialization must be tested inside the allocated Slurm job.')
PY
[[ -x "$REPORT_VENV/bin/python" ]] || uv venv --python "$PYTHON_VERSION" "$REPORT_VENV"
uv pip install --python "$REPORT_VENV/bin/python" -r requirements-report.txt
uv pip freeze --python "$REPORT_VENV/bin/python" > "$REPORT_VENV/report-freeze.txt"
printf 'Ready: source %q/bin/activate\n' "$VENV"
printf 'Report interpreter: %q/bin/python\n' "$REPORT_VENV"
