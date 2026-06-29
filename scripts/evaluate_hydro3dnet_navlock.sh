#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TORCH_LIB="$(python3 - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).parent / "lib")
PY
)"

CUDNN_LIB="$(python3 - <<'PY'
import nvidia.cudnn
from pathlib import Path
print(Path(nvidia.cudnn.__file__).parent / "lib")
PY
)"

cd "$REPO_ROOT/third_party/Hydro3DNet/tools"

export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDNN_LIB:$TORCH_LIB:$CUDA_HOME/lib64:${CONDA_PREFIX:-}/lib"

CKPT="../output/navlock_models/hydro3dnet_navlock/default/ckpt/latest_model.pth"

python3 test.py \
  --cfg_file cfgs/navlock_models/hydro3dnet_navlock.yaml \
  --batch_size 1 \
  --ckpt "$CKPT" \
  "$@"
