#!/usr/bin/env python3
"""Build scene-level temporal indexes for the NavLock dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


CAMERA_CHANNELS = (
    "CAM_1",
    "CAM_2",
    "CAM_3",
    "CAM_4",
    "CAM_5",
    "CAM_6",
    "CAM_7",
    "CAM_8",
)

GEOMETRIC_CAMERA_CHANNELS = ("CAM_1", "CAM_2", "CAM_4", "CAM_5", "CAM_6", "CAM_7")
STATE_CAMERA_CHANNELS = ("CAM_3", "CAM_8")

SHIP_INTENTION_BY_ATTRIBUTE = {
    "attribute_ship_entering_lock": "ship_entering_lock",
    "attribute_ship_leaving_lock": "ship_leaving_lock",
    "attribute_ship_berthed": "ship_berthed",
}

GATE_LABELS = {
    "open": "open",
    "closed": "closed",
    "opening": "opening",
    "closing": "closing",
}

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)
        f.write("\n")


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_split_scene_map(data_root: Path) -> dict[str, str]:
    split_by_scene = {}
    for split in ("train", "val", "test"):
        for scene_token in read_lines(data_root / "splits" / f"{split}_scenes.txt"):
            split_by_scene[scene_token] = split
    return split_by_scene


def build_sensor_maps(version_root: Path):
    sensors = {item["token"]: item for item in load_json(version_root / "sensor.json")}
    calibrated = {
        item["token"]: item for item in load_json(version_root / "calibrated_sensor.json")
    }
    channel_by_calib = {}
    calibrated_by_channel = {}
    for calib_token, calib in calibrated.items():
        sensor = sensors[calib["sensor_token"]]
        channel = sensor["channel"]
        channel_by_calib[calib_token] = channel
        calibrated_by_channel[channel] = {
            "calibrated_sensor_token": calib_token,
            "sensor_token": calib["sensor_token"],
            "translation": calib["translation"],
            "rotation": calib["rotation"],
            "camera_intrinsic": calib.get("camera_intrinsic", []),
            "is_calibrated": True,
        }
    return channel_by_calib, calibrated_by_channel


def build_sample_data_by_sample(version_root: Path, channel_by_calib: dict[str, str]):
    sample_data_by_sample = defaultdict(dict)
    for item in load_json(version_root / "sample_data.json"):
        channel = channel_by_calib[item["calibrated_sensor_token"]]
        sample_data_by_sample[item["sample_token"]][channel] = item
    return sample_data_by_sample


def build_2d_index(data_root: Path):
    image_index_by_split = {}
    category_index_by_split = {}
    for split in ("train", "val", "test"):
        coco = load_json(data_root / "2d_annotations" / f"instances_{split}.json")
        ann_ids_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            ann_ids_by_image[ann["image_id"]].append(ann["id"])

        image_index = {}
        for image in coco["images"]:
            image_index[image["file_name"]] = {
                "image_id": image["id"],
                "width": image["width"],
                "height": image["height"],
                "annotation_ids": ann_ids_by_image.get(image["id"], []),
            }
        image_index_by_split[split] = image_index
        category_index_by_split[split] = {
            item["id"]: item["name"] for item in coco["categories"]
        }
    return image_index_by_split, category_index_by_split


def build_annotations(version_root: Path):
    categories = {item["token"]: item["name"] for item in load_json(version_root / "category.json")}
    instances = {item["token"]: item for item in load_json(version_root / "instance.json")}
    attributes = {item["token"]: item for item in load_json(version_root / "attribute.json")}

    annotations_by_sample = defaultdict(list)
    for ann in load_json(version_root / "sample_annotation.json"):
        instance = instances[ann["instance_token"]]
        category_name = categories[instance["category_token"]]
        attribute_tokens = ann.get("attribute_tokens", [])
        attribute_names = [
            attributes[token]["name"] for token in attribute_tokens if token in attributes
        ]
        ship_intentions = [
            SHIP_INTENTION_BY_ATTRIBUTE[token]
            for token in attribute_tokens
            if token in SHIP_INTENTION_BY_ATTRIBUTE
        ]
        annotations_by_sample[ann["sample_token"]].append(
            {
                "annotation_token": ann["token"],
                "instance_token": ann["instance_token"],
                "category": category_name,
                "translation": ann["translation"],
                "size": ann["size"],
                "rotation": ann["rotation"],
                "velocity": ann.get("velocity", [0.0, 0.0]),
                "num_lidar_points": ann.get("num_lidar_pts", 0),
                "num_radar_points": ann.get("num_radar_pts", 0),
                "visibility_token": ann.get("visibility_token", ""),
                "visibility_level": ann.get("visibility_level", ""),
                "occlusion_state": ann.get("occlusion_state", ""),
                "assigned_berth_slot": ann.get("assigned_berth_slot"),
                "attribute_tokens": attribute_tokens,
                "attribute_names": attribute_names,
                "ship_intentions": ship_intentions,
            }
        )
    return annotations_by_sample


def manual_camera_path(channel: str, timestamp_str: str) -> str:
    camera_id = channel.split("_")[1]
    return f"samples/{channel}/camera{camera_id}_{timestamp_str}.png"


def build_image_entry(
    channel: str,
    sample_token: str,
    timestamp: int,
    timestamp_str: str,
    split: str,
    sample_data_by_sample,
    calibrated_by_channel,
    image_index_by_split,
):
    sample_data = sample_data_by_sample.get(sample_token, {}).get(channel)
    if sample_data:
        file_name = sample_data["filename"]
        width = sample_data.get("width")
        height = sample_data.get("height")
        sample_data_token = sample_data["token"]
    else:
        file_name = manual_camera_path(channel, timestamp_str)
        image_meta = image_index_by_split.get(split, {}).get(file_name, {})
        width = image_meta.get("width")
        height = image_meta.get("height")
        sample_data_token = None

    image_meta = image_index_by_split.get(split, {}).get(file_name, {})
    calibrated = calibrated_by_channel.get(channel)
    return {
        "channel": channel,
        "file_name": file_name,
        "sample_data_token": sample_data_token,
        "timestamp": timestamp,
        "width": width,
        "height": height,
        "image_id_2d": image_meta.get("image_id"),
        "annotation_ids_2d": image_meta.get("annotation_ids", []),
        "camera_role": "state" if channel in STATE_CAMERA_CHANNELS else "geometric",
        "is_calibrated": bool(calibrated),
        "calibration": calibrated,
    }


def build_lidar_entry(sample_token: str, sample_data_by_sample):
    sample_data = sample_data_by_sample.get(sample_token, {}).get("LIDAR_TOP")
    if not sample_data:
        return None
    return {
        "channel": "LIDAR_TOP",
        "file_name": sample_data["filename"],
        "sample_data_token": sample_data["token"],
        "timestamp": sample_data["timestamp"],
        "num_point_features": 5,
    }


def lock_labels(sample: dict) -> dict:
    upper_state = sample.get("upper_gate_state", "unknown")
    lower_state = sample.get("lower_gate_state", "unknown")
    water_state = sample.get("lock_water_state", "unknown")
    water_level = sample.get("water_level")
    labels = {
        "upper_gate_state": upper_state,
        "lower_gate_state": lower_state,
        "water_state": water_state,
        "water_level": water_level,
        "upper_gate_label": f"upper_gate_{GATE_LABELS.get(upper_state, 'unknown')}",
        "lower_gate_label": f"lower_gate_{GATE_LABELS.get(lower_state, 'unknown')}",
    }
    for key in (
        "upstream_water_level",
        "downstream_water_level",
        "observed_action",
        "action_start_time",
        "action_end_time",
        "action_target",
        "action_source",
        "action_confidence",
        "operation_phase",
        "phase_start_time",
        "phase_end_time",
        "ship_operation_phase",
        "ship_phase_start_time",
        "ship_phase_end_time",
        "no_ship_in_upper_gate_zone",
        "no_ship_in_lower_gate_zone",
        "entry_path_clear",
        "exit_path_clear",
        "chamber_capacity_available",
        "available_berth_slots",
        "occupied_berth_slots",
        "num_occupied_berths",
        "num_ships_in_chamber",
        "all_in_chamber_ships_berthed_or_static",
        "no_ship_entering_or_leaving_inside_chamber",
        "queue_rank",
        "next_ship_to_enter_weak",
        "next_ship_to_leave_weak",
        "max_parallel_entries",
        "max_parallel_departures",
        "valid_actions",
        "invalid_actions",
        "violation_reason",
        "state_t_plus_10s",
        "state_t_plus_20s",
        "state_t_plus_30s",
        "phase_t_plus_10s",
        "phase_t_plus_20s",
        "phase_t_plus_30s",
        "future_state_after_observed_action",
        "future_phase_after_observed_action",
        "ship_dispatch_action",
        "ship_dispatch_targets",
        "ship_dispatch_target_count",
        "ship_dispatch_source",
        "ship_dispatch_confidence",
        "ship_dispatch_conflict",
    ):
        if key in sample:
            labels[key] = sample[key]
    return labels


def build_sequences(data_root: Path):
    version_root = data_root / "v1.0-trainval"
    scenes = load_json(version_root / "scene.json")
    samples = load_json(version_root / "sample.json")
    summary = {
        item["scene_token"]: item
        for item in load_json(version_root / "scene_frame_summary_direction_fixed.json")
    }

    split_by_scene = build_split_scene_map(data_root)
    channel_by_calib, calibrated_by_channel = build_sensor_maps(version_root)
    sample_data_by_sample = build_sample_data_by_sample(version_root, channel_by_calib)
    image_index_by_split, category_index_by_split = build_2d_index(data_root)
    annotations_by_sample = build_annotations(version_root)

    samples_by_scene = defaultdict(list)
    for sample in samples:
        samples_by_scene[sample["scene_token"]].append(sample)
    for scene_samples in samples_by_scene.values():
        scene_samples.sort(key=lambda item: item["timestamp"])

    sequences = []
    for scene in scenes:
        scene_token = scene["token"]
        split = split_by_scene.get(scene_token, "unknown")
        frame_samples = samples_by_scene.get(scene_token, [])
        if not frame_samples:
            continue

        start_time = frame_samples[0]["timestamp"]
        frames = []
        input_indices = []
        target_indices = []

        for index, sample in enumerate(frame_samples):
            rel_time_sec = (sample["timestamp"] - start_time) / 1_000_000.0
            timestamp_str = sample.get("timestamp_str") or sample["token"].replace("sample_", "")
            images = {
                channel: build_image_entry(
                    channel=channel,
                    sample_token=sample["token"],
                    timestamp=sample["timestamp"],
                    timestamp_str=timestamp_str,
                    split=split,
                    sample_data_by_sample=sample_data_by_sample,
                    calibrated_by_channel=calibrated_by_channel,
                    image_index_by_split=image_index_by_split,
                )
                for channel in CAMERA_CHANNELS
            }
            frame = {
                "frame_index": index,
                "sample_token": sample["token"],
                "sample_idx": timestamp_str,
                "frame_id": sample.get("frame_id"),
                "timestamp": sample["timestamp"],
                "timestamp_str": timestamp_str,
                "relative_time_sec": rel_time_sec,
                "images": images,
                "lidar": build_lidar_entry(sample["token"], sample_data_by_sample),
                "lock_state": lock_labels(sample),
                "instances_3d": annotations_by_sample.get(sample["token"], []),
            }
            frames.append(frame)
            if rel_time_sec <= 50.0:
                input_indices.append(index)
            elif rel_time_sec <= 60.0:
                target_indices.append(index)

        scene_summary = summary.get(scene_token, {})
        duration_sec = (
            (frame_samples[-1]["timestamp"] - frame_samples[0]["timestamp"]) / 1_000_000.0
            if len(frame_samples) > 1
            else 0.0
        )
        sequence = {
            "scene_token": scene_token,
            "scene_name": scene["name"],
            "split": split,
            "log_token": scene["log_token"],
            "direction": scene_summary.get("direction"),
            "operation_date": scene_summary.get("operation_date"),
            "operation_index": scene_summary.get("operation_index"),
            "line_index": scene_summary.get("line_index"),
            "segment_index": scene_summary.get("segment_index"),
            "nominal_duration_sec": 60.0,
            "actual_duration_sec": duration_sec,
            "num_frames": len(frames),
            "camera_channels": list(CAMERA_CHANNELS),
            "geometric_camera_channels": list(GEOMETRIC_CAMERA_CHANNELS),
            "state_camera_channels": list(STATE_CAMERA_CHANNELS),
            "recognition_frame_indices": list(range(len(frames))),
            "prediction_input_frame_indices": input_indices,
            "prediction_target_frame_indices": target_indices,
            "has_prediction_target": bool(input_indices and target_indices),
            "frames": frames,
        }
        sequences.append(sequence)

    metadata = {
        "dataset": "NavLock-World",
        "source_dataset": "NavLock-HY",
        "schema_version": "navlock_sequence_v1",
        "language": "en",
        "nominal_scene_duration_sec": 60.0,
        "prediction_input_duration_sec": 50.0,
        "prediction_target_duration_sec": 10.0,
        "camera_channels": list(CAMERA_CHANNELS),
        "geometric_camera_channels": list(GEOMETRIC_CAMERA_CHANNELS),
        "state_camera_channels": list(STATE_CAMERA_CHANNELS),
        "lidar": {
            "channel": "LIDAR_TOP",
            "modality": "lidar",
            "num_point_features": 5,
            "is_calibrated": "LIDAR_TOP" in calibrated_by_channel,
            "calibration": calibrated_by_channel.get("LIDAR_TOP"),
        },
        "label_schema": {
            "ship_intentions": sorted(set(SHIP_INTENTION_BY_ATTRIBUTE.values())),
            "upper_gate_labels": [
                "upper_gate_open",
                "upper_gate_closed",
                "upper_gate_opening",
                "upper_gate_closing",
            ],
            "lower_gate_labels": [
                "lower_gate_open",
                "lower_gate_closed",
                "lower_gate_opening",
                "lower_gate_closing",
            ],
            "water_state_labels": ["idle", "filling", "emptying"],
            "observed_actions": [
                "hold",
                "open_upper_gate",
                "close_upper_gate",
                "open_lower_gate",
                "close_lower_gate",
                "start_filling",
                "start_emptying",
                "stop_filling_emptying",
            ],
            "operation_phases": [
                "all_gates_closed_idle",
                "upper_gate_open_idle",
                "lower_gate_open_idle",
                "gate_opening",
                "gate_closing",
                "filling",
                "emptying",
                "hold_uncertain",
            ],
            "ship_operation_phases": [
                "waiting_for_entry",
                "ship_entering",
                "all_ships_berthed",
                "ship_leaving",
                "lock_clear",
                "ship_phase_uncertain",
            ],
            "ship_operation_phase_fields": [
                "ship_operation_phase",
                "ship_phase_start_time",
                "ship_phase_end_time",
            ],
            "gate_zone_clear_labels": [
                "no_ship_in_upper_gate_zone",
                "no_ship_in_lower_gate_zone",
            ],
            "path_clear_labels": [
                "entry_path_clear",
                "exit_path_clear",
            ],
            "chamber_queue_labels": [
                "chamber_capacity_available",
                "available_berth_slots",
                "occupied_berth_slots",
                "num_occupied_berths",
                "num_ships_in_chamber",
                "all_in_chamber_ships_berthed_or_static",
                "no_ship_entering_or_leaving_inside_chamber",
                "queue_rank",
                "next_ship_to_enter_weak",
                "next_ship_to_leave_weak",
                "max_parallel_entries",
                "max_parallel_departures",
            ],
            "planner_actions": [
                "hold",
                "open_upper_gate",
                "close_upper_gate",
                "open_lower_gate",
                "close_lower_gate",
                "start_filling",
                "start_emptying",
                "stop_filling_emptying",
                "dispatch_enter",
                "dispatch_exit",
            ],
            "planner_action_mask_fields": [
                "valid_actions",
                "invalid_actions",
                "violation_reason",
            ],
            "action_future_fields": [
                "state_t_plus_10s",
                "state_t_plus_20s",
                "state_t_plus_30s",
                "phase_t_plus_10s",
                "phase_t_plus_20s",
                "phase_t_plus_30s",
                "future_state_after_observed_action",
                "future_phase_after_observed_action",
            ],
            "ship_context_fields": [
                "assigned_berth_slot",
                "occlusion_state",
                "visibility_level",
            ],
            "ship_dispatch_fields": [
                "ship_dispatch_action",
                "ship_dispatch_targets",
                "ship_dispatch_target_count",
                "ship_dispatch_source",
                "ship_dispatch_confidence",
                "ship_dispatch_conflict",
            ],
            "water_level": {
                "field": "water_level",
                "type": "continuous",
                "unit": "meter",
                "source": "v1.0-trainval/sample.json",
            },
        },
        "annotation_sources": {
            "3d": "v1.0-trainval/sample_annotation.json",
            "2d_train": "2d_annotations/instances_train.json",
            "2d_val": "2d_annotations/instances_val.json",
            "2d_test": "2d_annotations/instances_test.json",
        },
        "category_index_2d": category_index_by_split,
    }
    return metadata, sequences


def summarize(sequences):
    by_split = defaultdict(lambda: {"num_scenes": 0, "num_frames": 0, "num_prediction_scenes": 0})
    for sequence in sequences:
        item = by_split[sequence["split"]]
        item["num_scenes"] += 1
        item["num_frames"] += sequence["num_frames"]
        item["num_prediction_scenes"] += int(sequence["has_prediction_target"])
    return dict(sorted(by_split.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--out-dir", default="data/navlock_sequences", type=Path)
    args = parser.parse_args()

    metadata, sequences = build_sequences(args.data_root)
    summary = summarize(sequences)

    dump_json(args.out_dir / "metadata.json", metadata)
    dump_json(args.out_dir / "scene_sequences_all.json", {"metadata": metadata, "sequences": sequences})
    for split in ("train", "val", "test"):
        split_sequences = [item for item in sequences if item["split"] == split]
        dump_json(
            args.out_dir / f"scene_sequences_{split}.json",
            {"metadata": metadata, "sequences": split_sequences},
        )
    dump_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
