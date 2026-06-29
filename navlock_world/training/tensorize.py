"""Tensorization helpers for structured NavLock scene samples."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch


SHIP_CATEGORIES = {
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Unladen_container_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
    "Tugboat",
    "Unknown_vessel",
}

FEATURE_NAMES = (
    "relative_time_norm",
    "num_instances_norm",
    "num_ship_instances_norm",
    "num_static_instances_norm",
    "mean_x_norm",
    "mean_y_norm",
    "mean_z_norm",
    "mean_length_norm",
    "mean_width_norm",
    "mean_height_norm",
    "mean_speed_norm",
    "max_speed_norm",
    "upper_gate_open",
    "upper_gate_closed",
    "upper_gate_opening",
    "upper_gate_closing",
    "lower_gate_open",
    "lower_gate_closed",
    "lower_gate_opening",
    "lower_gate_closing",
    "water_idle",
    "water_filling",
    "water_emptying",
)


class NavLockTensorizer:
    """Convert scene dictionaries into padded temporal tensors."""

    feature_names = FEATURE_NAMES

    def __init__(self, include_observed_lock_state: bool = True) -> None:
        self.include_observed_lock_state = include_observed_lock_state

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def tensorize_frame(self, frame: dict[str, Any]) -> dict[str, torch.Tensor]:
        features = self._frame_features(frame)
        upper = frame["lock_state_labels"]["upper_gate_state"]
        lower = frame["lock_state_labels"]["lower_gate_state"]
        water = frame["lock_state_labels"]["water_state"]
        ship_intentions = torch.zeros(3, dtype=torch.float32)
        for instance in frame["instances_3d"]:
            for label in instance.get("ship_intention_labels", []):
                if 0 <= label < 3:
                    ship_intentions[label] = 1.0
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "upper_gate_target": torch.tensor(upper, dtype=torch.long),
            "lower_gate_target": torch.tensor(lower, dtype=torch.long),
            "water_target": torch.tensor(water, dtype=torch.long),
            "ship_intention_target": ship_intentions,
        }

    def tensorize_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        frames = sample["frames"]
        tensor_frames = [self.tensorize_frame(frame) for frame in frames]
        return {
            "scene_token": sample["scene_token"],
            "scene_name": sample["scene_name"],
            "features": torch.stack([item["features"] for item in tensor_frames]),
            "upper_gate_targets": torch.stack(
                [item["upper_gate_target"] for item in tensor_frames]
            ),
            "lower_gate_targets": torch.stack(
                [item["lower_gate_target"] for item in tensor_frames]
            ),
            "water_targets": torch.stack([item["water_target"] for item in tensor_frames]),
            "ship_intention_targets": torch.stack(
                [item["ship_intention_target"] for item in tensor_frames]
            ),
            "has_prediction_target": sample["has_prediction_target"],
            "num_frames": len(frames),
        }

    def _frame_features(self, frame: dict[str, Any]) -> list[float]:
        instances = frame["instances_3d"]
        categories = Counter(item["category"] for item in instances)
        ship_instances = [item for item in instances if item["category"] in SHIP_CATEGORIES]
        static_instances = [item for item in instances if item["category"] not in SHIP_CATEGORIES]

        translations = [item["translation"] for item in ship_instances]
        sizes = [item["size"] for item in ship_instances]
        speeds = [
            (item["velocity"][0] ** 2 + item["velocity"][1] ** 2) ** 0.5
            for item in ship_instances
        ]

        mean_xyz = self._mean_vector(translations, 3)
        mean_size = self._mean_vector(sizes, 3)
        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0

        lock_state = frame["lock_state_labels"]
        upper_one_hot = self._one_hot(lock_state["upper_gate_state"], 4)
        lower_one_hot = self._one_hot(lock_state["lower_gate_state"], 4)
        water_one_hot = self._one_hot(lock_state["water_state"], 3)
        if not self.include_observed_lock_state:
            upper_one_hot = [0.0] * 4
            lower_one_hot = [0.0] * 4
            water_one_hot = [0.0] * 3

        return [
            frame["relative_time_sec"] / 60.0,
            len(instances) / 10.0,
            len(ship_instances) / 10.0,
            len(static_instances) / 10.0,
            mean_xyz[0] / 100.0,
            mean_xyz[1] / 350.0,
            mean_xyz[2] / 20.0,
            mean_size[0] / 100.0,
            mean_size[1] / 100.0,
            mean_size[2] / 20.0,
            mean_speed / 5.0,
            max_speed / 5.0,
            *upper_one_hot,
            *lower_one_hot,
            *water_one_hot,
        ]

    @staticmethod
    def _mean_vector(values: list[list[float]], dim: int) -> list[float]:
        if not values:
            return [0.0] * dim
        return [sum(item[i] for item in values) / len(values) for i in range(dim)]

    @staticmethod
    def _one_hot(label: int, num_classes: int) -> list[float]:
        values = [0.0] * num_classes
        if 0 <= label < num_classes:
            values[label] = 1.0
        return values


def navlock_tensor_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad tensorized variable-length samples into a batch."""

    max_len = max(item["num_frames"] for item in batch)
    batch_size = len(batch)
    feature_dim = batch[0]["features"].shape[-1]

    features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    upper_targets = torch.full((batch_size, max_len), -1, dtype=torch.long)
    lower_targets = torch.full((batch_size, max_len), -1, dtype=torch.long)
    water_targets = torch.full((batch_size, max_len), -1, dtype=torch.long)
    ship_targets = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        length = item["num_frames"]
        features[batch_index, :length] = item["features"]
        upper_targets[batch_index, :length] = item["upper_gate_targets"]
        lower_targets[batch_index, :length] = item["lower_gate_targets"]
        water_targets[batch_index, :length] = item["water_targets"]
        ship_targets[batch_index, :length] = item["ship_intention_targets"]
        frame_mask[batch_index, :length] = True

    return {
        "scene_tokens": [item["scene_token"] for item in batch],
        "features": features,
        "frame_mask": frame_mask,
        "upper_gate_targets": upper_targets,
        "lower_gate_targets": lower_targets,
        "water_targets": water_targets,
        "ship_intention_targets": ship_targets,
        "has_prediction_target": [item["has_prediction_target"] for item in batch],
    }

