#!/usr/bin/env python3
"""Validate the NavLock RTMDet OpenMMLab config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mmcv
import mmdet
import numpy as np
import torch
from mmcv.ops import nms  # noqa: F401
from mmengine.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_2d(config_path: str) -> None:
    from mmdet.registry import DATASETS, MODELS
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(config_path)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    item = dataset[0]
    model = MODELS.build(cfg.model)
    print(f"2d_dataset_len={len(dataset)}")
    print(f"2d_first_input_shape={tuple(item['inputs'].shape)}")
    print(f"2d_first_gt_count={len(item['data_samples'].gt_instances)}")
    print(f"2d_model={type(model).__name__}")
    print(f"2d_num_classes={model.bbox_head.num_classes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/mmdet2d/navlock_rtmdet_s_8xb16.py",
    )
    args = parser.parse_args()

    print(f"torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
    print(f"numpy={np.__version__}")
    print(f"opencv={cv2.__version__}")
    print(f"mmcv={mmcv.__version__}")
    print(f"mmdet={mmdet.__version__}, path={mmdet.__file__}")
    print("mmcv_ops=ok")
    check_2d(args.config)


if __name__ == "__main__":
    main()
