#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_ROOT="${CONDA_PREFIX:-/opt/conda/envs/NavLock-World}"
CUDNN_LIB="${CONDA_ENV_ROOT}/lib/python3.9/site-packages/nvidia/cudnn/lib"
SPLIT="${1:-test}"
CHECKPOINT="${2:-${PROJECT_ROOT}/outputs/mmdet2d/navlock_rtmdet_s/epoch_20.pth}"
OUT_FILE="${3:-${PROJECT_ROOT}/outputs/mmdet2d/navlock_rtmdet_s/${SPLIT}_predictions.pkl}"

export LD_LIBRARY_PATH="${CUDNN_LIB}:${CONDA_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/mmdetection:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "${PROJECT_ROOT}"
python third_party/mmdetection/tools/test.py \
    configs/mmdet2d/navlock_rtmdet_s_8xb16.py \
    "${CHECKPOINT}" \
    --work-dir outputs/mmdet2d/navlock_rtmdet_s_eval \
    --out "${OUT_FILE}" \
    --cfg-options \
    test_dataloader.dataset.ann_file="2d_annotations/instances_${SPLIT}.json" \
    test_evaluator.ann_file="data/2d_annotations/instances_${SPLIT}.json"
