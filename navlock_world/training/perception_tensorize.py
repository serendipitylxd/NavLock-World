"""Tensorization for perception-fused future prediction baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from navlock_world.datasets.navlock_scene_dataset import SHIP_INTENTION_LABELS
from navlock_world.training.tensorize import NavLockTensorizer


PERCEPTION_FLAT_FEATURES = (
    "camera_num_detections",
    "camera_num_ship_detections",
    "camera_top_score",
    "camera_mean_ship_score",
    "lidar_num_detections",
    "lidar_num_ship_detections",
    "lidar_top_score",
    "lidar_mean_ship_score",
)

PERCEPTION_FEATURE_NAMES = (
    "camera_num_detections_norm",
    "camera_num_ship_detections_norm",
    "camera_top_score",
    "camera_mean_ship_score",
    "lidar_num_detections_norm",
    "lidar_num_ship_detections_norm",
    "lidar_top_score",
    "lidar_mean_ship_score",
)


class PerceptionFeatureStore:
    """Lookup table over frame-level detector feature cache."""

    def __init__(self, cache_file: str | Path) -> None:
        self.cache_file = Path(cache_file)
        with self.cache_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.metadata = payload["metadata"]
        self.by_sample_token = {
            item["sample_token"]: item["flat_features"] for item in payload["frames"]
        }

    def get(self, sample_token: str) -> dict[str, float]:
        try:
            return self.by_sample_token[sample_token]
        except KeyError as exc:
            raise KeyError(
                f"missing perception features for sample_token={sample_token!r} "
                f"in {self.cache_file}"
            ) from exc


class PerceptionTemporalTensorizer:
    """Convert prediction samples into perception-fused temporal tensors."""

    def __init__(self, perception_store: PerceptionFeatureStore) -> None:
        self.structured_tensorizer = NavLockTensorizer(include_observed_lock_state=True)
        self.perception_store = perception_store

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            *self.structured_tensorizer.feature_names,
            "water_level_norm",
            *PERCEPTION_FEATURE_NAMES,
        )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def tensorize_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        input_frames = sample["prediction"]["input_frames"]
        target_frames = sample["prediction"]["target_frames"]
        if not target_frames:
            raise ValueError(f"sample has no prediction target: {sample['scene_token']}")

        target_frame = target_frames[-1]
        features = torch.stack([self.tensorize_input_frame(frame) for frame in input_frames])
        target = self.tensorize_target_frame(target_frame)
        return {
            "scene_token": sample["scene_token"],
            "scene_name": sample["scene_name"],
            "features": features,
            "num_frames": len(input_frames),
            **target,
        }

    def tensorize_input_frame(self, frame: dict[str, Any]) -> torch.Tensor:
        structured = self.structured_tensorizer.tensorize_frame(frame)["features"]
        water_level = frame.get("water_level")
        water_level_norm = 0.0 if water_level is None else float(water_level) / 10.0
        perception = self._perception_vector(frame["sample_token"])
        return torch.cat(
            [
                structured,
                torch.tensor([water_level_norm], dtype=torch.float32),
                perception,
            ]
        )

    def tensorize_target_frame(self, frame: dict[str, Any]) -> dict[str, torch.Tensor]:
        labels = frame["lock_state_labels"]
        water_level = frame.get("water_level")
        ship_intentions = torch.zeros(len(SHIP_INTENTION_LABELS), dtype=torch.float32)
        for instance in frame["instances_3d"]:
            for label in instance.get("ship_intention_labels", []):
                if 0 <= label < len(SHIP_INTENTION_LABELS):
                    ship_intentions[label] = 1.0
        return {
            "upper_gate_target": torch.tensor(labels["upper_gate_state"], dtype=torch.long),
            "lower_gate_target": torch.tensor(labels["lower_gate_state"], dtype=torch.long),
            "water_state_target": torch.tensor(labels["water_state"], dtype=torch.long),
            "water_level_target": torch.tensor(float(water_level), dtype=torch.float32),
            "ship_intention_target": ship_intentions,
        }

    def _perception_vector(self, sample_token: str) -> torch.Tensor:
        features = self.perception_store.get(sample_token)
        values = []
        for name in PERCEPTION_FLAT_FEATURES:
            value = float(features.get(name, 0.0))
            if name.startswith("camera_num"):
                value /= 100.0
            elif name.startswith("lidar_num"):
                value /= 20.0
            values.append(value)
        return torch.tensor(values, dtype=torch.float32)


def perception_temporal_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length future-prediction samples."""

    max_len = max(item["num_frames"] for item in batch)
    batch_size = len(batch)
    feature_dim = batch[0]["features"].shape[-1]

    features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for batch_index, item in enumerate(batch):
        length = item["num_frames"]
        features[batch_index, :length] = item["features"]
        frame_mask[batch_index, :length] = True

    return {
        "scene_tokens": [item["scene_token"] for item in batch],
        "features": features,
        "frame_mask": frame_mask,
        "upper_gate_targets": torch.stack([item["upper_gate_target"] for item in batch]),
        "lower_gate_targets": torch.stack([item["lower_gate_target"] for item in batch]),
        "water_state_targets": torch.stack([item["water_state_target"] for item in batch]),
        "water_level_targets": torch.stack([item["water_level_target"] for item in batch]),
        "ship_intention_targets": torch.stack(
            [item["ship_intention_target"] for item in batch]
        ),
    }
