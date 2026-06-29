#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_ROOT="${CONDA_PREFIX:-/opt/conda/envs/NavLock-World}"
CUDNN_LIB="${CONDA_ENV_ROOT}/lib/python3.9/site-packages/nvidia/cudnn/lib"

export LD_LIBRARY_PATH="${CUDNN_LIB}:${CONDA_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "${PROJECT_ROOT}"
python third_party/mmdetection/tools/train.py configs/mmdet2d/navlock_rtmdet_s_8xb16.py "$@"
