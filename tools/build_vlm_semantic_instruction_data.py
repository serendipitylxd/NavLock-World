#!/usr/bin/env python3
"""Build generic VLM semantic branch instruction data for NavLock scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


CALIBRATED_CAMERA_CHANNELS = ("CAM_1", "CAM_2", "CAM_4", "CAM_5", "CAM_6", "CAM_7")
STATE_CAMERA_CHANNELS = ("CAM_3", "CAM_8")
STATE_CAMERA_ROLES = {
    "CAM_3": "upper_gate_and_upper_gate_near_water_surface_state",
    "CAM_8": "lower_gate_and_lower_gate_near_water_surface_state",
}
WAVE_ANNOTATION_RULES = {
    "filling": {
        "camera": "CAM_3",
        "region_id": "upper_gate_left_in_chamber",
        "region_description": "left side of the upper gate, inside the lock chamber",
    },
    "emptying": {
        "camera": "CAM_8",
        "region_id": "lower_gate_right_outside_chamber",
        "region_description": "right side of the lower gate, outside the lock chamber",
    },
}
SHIP_2D_CLASSES = (
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Unladen_container_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
)
SHIP_3D_CLASSES = (
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--perception-cache", default=None)
    parser.add_argument("--max-keyframes", type=int, default=6)
    parser.add_argument(
        "--task-mode",
        choices=("prediction", "recognition", "all"),
        default="prediction",
        help=(
            "prediction keeps only scenes with a 50-60s future target; "
            "recognition exports every scene for current-state recognition; "
            "all writes both tasks into one JSONL."
        ),
    )
    parser.add_argument(
        "--recognition-granularity",
        choices=("scene", "frame"),
        default="scene",
        help=(
            "scene exports one recognition item per scene, matching the historical "
            "VLM semantic protocol. frame exports one recognition item per recognition "
            "frame, using earlier selected frames as temporal context and the "
            "current frame as the answer target."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to outputs/vlm_semantic/navlock_vlm_semantic_<task_mode>_<split>.jsonl.",
    )
    parser.add_argument(
        "--wave-label-file",
        default=None,
        help=(
            "Optional targeted wave-label JSONL. Defaults to "
            "outputs/wave_labels/navlock_wave_labels_<split>.jsonl when present."
        ),
    )
    parser.add_argument(
        "--lidar-view-root",
        default="outputs/vlm_semantic/lidar_views",
        help=(
            "Root containing rendered LiDAR BEV/range-view PNGs. Existing files "
            "are linked into each frame as lidar.rendered_views."
        ),
    )
    parser.add_argument(
        "--no-lidar-views",
        action="store_true",
        help="Do not attach rendered LiDAR view paths even when they exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    sequence_file = data_root / "navlock_sequences" / f"scene_sequences_{args.split}.json"
    perception_file = (
        Path(args.perception_cache)
        if args.perception_cache
        else Path("outputs")
        / "perception_features"
        / f"perception_features_{args.split}.json"
    )
    output = (
        Path(args.output)
        if args.output
        else Path("outputs")
        / "vlm_semantic"
        / _default_output_name(args.task_mode, args.split)
    )

    sequences_payload = _load_json(sequence_file)
    perception_by_sample, detector_sources = _load_perception_cache(perception_file)
    wave_label_file = (
        Path(args.wave_label_file)
        if args.wave_label_file
        else Path("outputs") / "wave_labels" / f"navlock_wave_labels_{args.split}.jsonl"
    )
    wave_labels_by_sample_camera = _load_wave_labels(wave_label_file)
    lidar_view_root = None if args.no_lidar_views else Path(args.lidar_view_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0
    with output.open("w", encoding="utf-8") as f:
        for sequence in sequences_payload["sequences"]:
            for item in build_items_for_mode(
                sequence=sequence,
                data_root=data_root,
                perception_by_sample=perception_by_sample,
                detector_sources=detector_sources,
                wave_labels_by_sample_camera=wave_labels_by_sample_camera,
                lidar_view_root=lidar_view_root,
                max_keyframes=args.max_keyframes,
                task_mode=args.task_mode,
                recognition_granularity=args.recognition_granularity,
            ):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                num_written += 1

    print(f"wrote={output}")
    print(f"split={args.split}")
    print(f"task_mode={args.task_mode}")
    print(f"recognition_granularity={args.recognition_granularity}")
    print(f"num_items={num_written}")
    print(f"max_keyframes={args.max_keyframes}")


def build_items_for_mode(
    sequence: dict[str, Any],
    data_root: Path,
    perception_by_sample: dict[str, dict[str, Any]],
    detector_sources: dict[str, str],
    max_keyframes: int,
    task_mode: str,
    wave_labels_by_sample_camera: Optional[dict[tuple[str, str], dict[str, Any]]] = None,
    lidar_view_root: Optional[Path] = None,
    recognition_granularity: str = "scene",
) -> list[dict[str, Any]]:
    items = []
    if task_mode in {"prediction", "all"} and sequence["has_prediction_target"]:
        items.append(
            build_prediction_item(
                sequence=sequence,
                data_root=data_root,
                perception_by_sample=perception_by_sample,
                detector_sources=detector_sources,
                wave_labels_by_sample_camera=wave_labels_by_sample_camera or {},
                lidar_view_root=lidar_view_root,
                max_keyframes=max_keyframes,
            )
        )
    if task_mode in {"recognition", "all"}:
        if recognition_granularity == "frame":
            for current_index in sequence["recognition_frame_indices"]:
                items.append(
                    build_recognition_item(
                        sequence=sequence,
                        data_root=data_root,
                        perception_by_sample=perception_by_sample,
                        detector_sources=detector_sources,
                        wave_labels_by_sample_camera=wave_labels_by_sample_camera or {},
                        lidar_view_root=lidar_view_root,
                        max_keyframes=max_keyframes,
                        current_frame_index=current_index,
                    )
                )
        else:
            items.append(
                build_recognition_item(
                    sequence=sequence,
                    data_root=data_root,
                    perception_by_sample=perception_by_sample,
                    detector_sources=detector_sources,
                    wave_labels_by_sample_camera=wave_labels_by_sample_camera or {},
                    lidar_view_root=lidar_view_root,
                    max_keyframes=max_keyframes,
                )
            )
    return items


def build_prediction_item(
    sequence: dict[str, Any],
    data_root: Path,
    perception_by_sample: dict[str, dict[str, Any]],
    detector_sources: dict[str, str],
    max_keyframes: int,
    wave_labels_by_sample_camera: Optional[dict[tuple[str, str], dict[str, Any]]] = None,
    lidar_view_root: Optional[Path] = None,
) -> dict[str, Any]:
    input_indices = sequence["prediction_input_frame_indices"]
    target_indices = sequence["prediction_target_frame_indices"]
    selected_indices = _select_keyframe_indices(input_indices, max_keyframes)
    all_input_frames = [sequence["frames"][index] for index in input_indices]
    input_frames = [sequence["frames"][index] for index in selected_indices]
    target_frame = sequence["frames"][target_indices[-1]]
    last_input_frame = sequence["frames"][input_indices[-1]]

    frame_summaries = [
        _frame_input_summary(
            frame,
            data_root,
            perception_by_sample,
            lidar_view_root=lidar_view_root,
            split=sequence["split"],
        )
        for frame in input_frames
    ]
    images = [
        image["path"]
        for frame in frame_summaries
        for image in frame["images"].values()
    ]
    lidar_images = _lidar_image_paths(frame_summaries)

    current_state = _state_summary(last_input_frame)
    future_state = _state_summary(target_frame)
    water_level_delta = _safe_float(future_state["water_level"]) - _safe_float(
        current_state["water_level"]
    )
    water_surface_task = _water_surface_task(
        sequence,
        last_input_frame,
        target_frame,
        wave_labels_by_sample_camera or {},
    )
    mooring_evidence = _mooring_confidence_evidence(frame_summaries)

    return {
        "id": f"{sequence['split']}:prediction:{sequence['scene_token']}",
        "split": sequence["split"],
        "scene_token": sequence["scene_token"],
        "scene_name": sequence["scene_name"],
        "task": "navlock_vlm_semantic_multimodal_temporal_reasoning",
        "instruction": _instruction(),
        "images": images,
        "lidar_images": lidar_images,
        "input": {
            "temporal_setup": {
                "input_duration_sec": 50,
                "future_horizon_sec": 10,
                "selected_frame_indices": selected_indices,
                "target_frame_index": target_frame["frame_index"],
            },
            "camera_layout": {
                "calibrated_geometry_cameras": list(CALIBRATED_CAMERA_CHANNELS),
                "uncalibrated_state_cameras": list(STATE_CAMERA_CHANNELS),
                "state_camera_roles": STATE_CAMERA_ROLES,
                "fusion_note": (
                    "Use calibrated cameras with LiDAR/3D boxes for geometric fusion; "
                    "use CAM_3 for upper-gate/water-surface evidence and CAM_8 for "
                    "lower-gate/water-surface evidence."
                ),
            },
            "frames": frame_summaries,
            "lidar_visualization": _lidar_visualization_summary(lidar_images),
            "current_state_from_last_input_frame": current_state,
            "gate_transition_context": _gate_transition_context(
                all_input_frames,
                current_state,
                ship_context_frames=frame_summaries,
            ),
            "detector_sources": detector_sources,
        },
        "answer": {
            "current_state": current_state,
            "future_state_10s": future_state,
            "future_water_level_delta": water_level_delta,
            "water_surface_dynamics": water_surface_task,
            "ship_behavior": {
                "ship_intentions": _ship_intentions(target_frame),
                "mooring_or_berthing_confidence_evidence": mooring_evidence,
            },
            "fusion_reasoning": {
                "use_calibrated_2d_3d_fusion": True,
                "calibrated_cameras": list(CALIBRATED_CAMERA_CHANNELS),
                "state_cameras_without_geometry": list(STATE_CAMERA_CHANNELS),
            },
        },
    }


def build_recognition_item(
    sequence: dict[str, Any],
    data_root: Path,
    perception_by_sample: dict[str, dict[str, Any]],
    detector_sources: dict[str, str],
    max_keyframes: int,
    wave_labels_by_sample_camera: Optional[dict[tuple[str, str], dict[str, Any]]] = None,
    lidar_view_root: Optional[Path] = None,
    current_frame_index: Optional[int] = None,
) -> dict[str, Any]:
    recognition_indices = sequence["recognition_frame_indices"]
    frame_level_mode = current_frame_index is not None
    if current_frame_index is None:
        current_frame_index = recognition_indices[-1]
    if current_frame_index not in recognition_indices:
        raise ValueError(
            f"current_frame_index={current_frame_index} is not a recognition frame "
            f"for scene {sequence['scene_token']}"
        )
    context_indices = [index for index in recognition_indices if index <= current_frame_index]
    selected_indices = _select_keyframe_indices(
        context_indices, max_keyframes
    )
    if selected_indices[-1] != current_frame_index:
        selected_indices = selected_indices[:-1] + [current_frame_index]
    current_frame = sequence["frames"][current_frame_index]
    first_frame = sequence["frames"][selected_indices[0]]

    frame_summaries = [
        _frame_input_summary(
            frame,
            data_root,
            perception_by_sample,
            lidar_view_root=lidar_view_root,
            split=sequence["split"],
        )
        for frame in (sequence["frames"][index] for index in selected_indices)
    ]
    images = [
        image["path"]
        for frame in frame_summaries
        for image in frame["images"].values()
    ]
    lidar_images = _lidar_image_paths(frame_summaries)

    current_state = _state_summary(current_frame)
    mooring_evidence = _mooring_confidence_evidence(frame_summaries)

    item_id = (
        f"{sequence['split']}:recognition_frame:{sequence['scene_token']}:"
        f"{current_frame['sample_token']}"
        if frame_level_mode
        else f"{sequence['split']}:recognition:{sequence['scene_token']}"
    )

    return {
        "id": item_id,
        "split": sequence["split"],
        "scene_token": sequence["scene_token"],
        "scene_name": sequence["scene_name"],
        "sample_token": current_frame["sample_token"],
        "timestamp": current_frame.get("timestamp"),
        "timestamp_str": current_frame.get("timestamp_str"),
        "current_frame_index": current_frame_index,
        "task": "navlock_vlm_semantic_current_multimodal_recognition",
        "instruction": _recognition_instruction(),
        "images": images,
        "lidar_images": lidar_images,
        "input": {
            "temporal_setup": {
                "recognition_duration_sec": sequence["actual_duration_sec"],
                "selected_frame_indices": selected_indices,
                "current_frame_index": current_frame["frame_index"],
                "has_future_prediction_target": sequence["has_prediction_target"],
            },
            "camera_layout": _camera_layout(),
            "frames": frame_summaries,
            "lidar_visualization": _lidar_visualization_summary(lidar_images),
            "detector_sources": detector_sources,
        },
        "answer": {
            "current_state": current_state,
            "current_water_level_delta_from_first_selected_frame": _safe_float(
                current_state["water_level"]
            )
            - _safe_float(first_frame["lock_state"].get("water_level")),
            "water_surface_dynamics": _current_water_surface_task(
                first_frame, current_frame, wave_labels_by_sample_camera or {}
            ),
            "ship_behavior": {
                "ship_intentions": _ship_intentions(current_frame),
                "mooring_or_berthing_confidence_evidence": mooring_evidence,
            },
            "fusion_reasoning": {
                "use_calibrated_2d_3d_fusion": True,
                "calibrated_cameras": list(CALIBRATED_CAMERA_CHANNELS),
                "state_cameras_without_geometry": list(STATE_CAMERA_CHANNELS),
            },
        },
    }


def _instruction() -> str:
    return (
        "You are a multimodal temporal world model specialized for ship-lock "
        "scenes. Use the 8-camera long temporal context, LiDAR/3D detections, "
        "2D detections, numeric water level, water state, and gate state to "
        "produce structured JSON. Focus on: 1) whether filling or emptying "
        "may cause waves or surface disturbance in the target water-surface "
        "region; 2) using CAM_3 for the upper gate and upper-gate-near "
        "water-surface state, and CAM_8 for the lower gate and lower-gate-near "
        "water-surface state; 3) whether "
        "crew, mooring lines, and ship position increase confidence in a "
        "moored/berthed interpretation, while remembering that invisible "
        "mooring lines must not rule out berthing; 4) whether the six "
        "calibrated cameras and LiDAR/3D detections support 2D/3D fusion."
    )


def _recognition_instruction() -> str:
    return (
        "You are a multimodal temporal world model specialized for ship-lock "
        "scenes. Use the selected 8-camera temporal context, LiDAR/3D "
        "detections, 2D detections, numeric water level, water state, and gate "
        "state to recognize the current scene state and output structured JSON. "
        "Use CAM_3 for upper-gate and upper-gate-near water-surface evidence, "
        "CAM_8 for lower-gate and lower-gate-near water-surface evidence, "
        "and the six calibrated cameras with LiDAR/3D detections for geometric "
        "2D/3D fusion when available. Treat visible crew and mooring lines as "
        "confidence evidence for berthing, but do not rule out berthing when "
        "mooring lines are occluded or invisible."
    )


def _camera_layout() -> dict[str, Any]:
    return {
        "calibrated_geometry_cameras": list(CALIBRATED_CAMERA_CHANNELS),
        "uncalibrated_state_cameras": list(STATE_CAMERA_CHANNELS),
        "state_camera_roles": STATE_CAMERA_ROLES,
        "fusion_note": (
            "Use calibrated cameras with LiDAR/3D boxes for geometric fusion; "
            "use CAM_3 for upper-gate/water-surface evidence and CAM_8 for "
            "lower-gate/water-surface evidence."
        ),
    }


def _detector_sources(metadata: Optional[dict[str, Any]] = None) -> dict[str, str]:
    if metadata:
        detector_sources = metadata.get("detector_sources")
        if isinstance(detector_sources, dict):
            return {str(key): str(value) for key, value in detector_sources.items()}
        det3d_backend = metadata.get("det3d_backend")
        if det3d_backend:
            det3d_name = {
                "hydro3dnet": "Hydro3DNet",
            }.get(str(det3d_backend), str(det3d_backend))
            return {
                "2d": "RTMDet",
                "3d": det3d_name,
                "fusion": "Structured fusion of RTMDet image summaries and Hydro3DNet LiDAR geometry.",
            }
    return {
        "2d": "RTMDet",
        "3d": "Hydro3DNet",
        "fusion": "Structured fusion of RTMDet image summaries and Hydro3DNet LiDAR geometry.",
    }


def _frame_input_summary(
    frame: dict[str, Any],
    data_root: Path,
    perception_by_sample: dict[str, dict[str, Any]],
    lidar_view_root: Optional[Path] = None,
    split: Optional[str] = None,
) -> dict[str, Any]:
    perception = perception_by_sample[frame["sample_token"]]
    images = {}
    for channel, image in frame["images"].items():
        images[channel] = {
            "path": str(data_root / image["file_name"]),
            "camera_role": image.get("camera_role"),
            "is_calibrated": bool(image.get("is_calibrated")),
            "calibration_available_for_2d_3d_fusion": channel in CALIBRATED_CAMERA_CHANNELS,
            "state_camera_role": STATE_CAMERA_ROLES.get(channel),
            "perception_2d_summary": perception["image_features"].get(channel),
        }

    lidar_entry = {
        "path": str(data_root / frame["lidar"]["file_name"]),
        "channel": frame["lidar"]["channel"],
        "num_point_features": frame["lidar"]["num_point_features"],
        "perception_3d_summary": perception["lidar_3d_features"],
    }
    rendered_views = _lidar_rendered_views(frame, lidar_view_root, split)
    if rendered_views:
        lidar_entry["rendered_views"] = rendered_views
        lidar_entry["rendered_view_descriptions"] = {
            "bev": (
                "Bird's-eye-view LiDAR raster: point density, height, and "
                "range cues in the configured NavLock point-cloud range."
            ),
            "range_view": (
                "Cylindrical range-view LiDAR raster: yaw/pitch projection "
                "with distance, height, and density cues."
            ),
        }

    return {
        "frame_index": frame["frame_index"],
        "sample_token": frame["sample_token"],
        "timestamp": frame["timestamp"],
        "relative_time_sec": frame["relative_time_sec"],
        "images": images,
        "lidar": lidar_entry,
        "lock_state": {
            "upper_gate_state": frame["lock_state"]["upper_gate_state"],
            "lower_gate_state": frame["lock_state"]["lower_gate_state"],
            "water_state": frame["lock_state"]["water_state"],
            "water_level": frame["lock_state"].get("water_level"),
        },
        "ship_instances": _ship_instance_context(frame),
        "flat_perception_features": perception["flat_features"],
    }


def _lidar_rendered_views(
    frame: dict[str, Any],
    lidar_view_root: Optional[Path],
    split: Optional[str],
) -> dict[str, str]:
    if lidar_view_root is None or not split:
        return {}
    sample_token = frame["sample_token"]
    split_root = lidar_view_root / split
    expected = {
        "bev": split_root / f"{sample_token}_bev.png",
        "range_view": split_root / f"{sample_token}_range.png",
    }
    return {name: str(path) for name, path in expected.items() if path.exists()}


def _lidar_image_paths(frame_summaries: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for frame in frame_summaries:
        rendered = frame.get("lidar", {}).get("rendered_views") or {}
        for view_name in ("bev", "range_view"):
            path = rendered.get(view_name)
            if path:
                paths.append(path)
    return paths


def _lidar_visualization_summary(lidar_images: list[str]) -> dict[str, Any]:
    return {
        "rendered_views_available": bool(lidar_images),
        "view_types": ["bev", "range_view"] if lidar_images else [],
        "num_rendered_lidar_images": len(lidar_images),
        "note": (
            "Rendered LiDAR views are VLM-friendly visualizations of the raw "
            "LIDAR_TOP point cloud. They complement, but do not replace, the "
            "structured 3D detection summaries."
            if lidar_images
            else "No rendered LiDAR views were linked for this sample."
        ),
    }


def _state_summary(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "upper_gate_state": frame["lock_state"]["upper_gate_state"],
        "lower_gate_state": frame["lock_state"]["lower_gate_state"],
        "water_state": frame["lock_state"]["water_state"],
        "water_level": frame["lock_state"].get("water_level"),
    }


def _gate_transition_context(
    input_frames: list[dict[str, Any]],
    current_state: dict[str, Any],
    ship_context_frames: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    sequence = [
        {
            "frame_index": frame.get("frame_index"),
            "relative_time_sec": frame.get("relative_time_sec"),
            "upper_gate_state": frame.get("lock_state", {}).get("upper_gate_state"),
            "lower_gate_state": frame.get("lock_state", {}).get("lower_gate_state"),
            "water_state": frame.get("lock_state", {}).get("water_state"),
            "water_level": frame.get("lock_state", {}).get("water_level"),
        }
        for frame in input_frames
    ]
    observed_transitions = _observed_gate_transitions(sequence)
    ship_status = _ship_berthing_status(ship_context_frames or input_frames)
    opening_hold_rules = _opening_completed_hold_rules(
        current_state,
        observed_transitions,
        ship_status,
    )
    return {
        "state_camera_mapping": {
            "upper_gate_state": "CAM_3",
            "lower_gate_state": "CAM_8",
        },
        "observed_input_gate_transitions": observed_transitions,
        "ship_berthing_status": ship_status,
        "candidate_future_gate_checks": _candidate_future_gate_checks(
            current_state,
            ship_status,
        ),
        "future_gate_domain_rules": _future_gate_domain_rules(
            current_state,
            observed_transitions,
            ship_status,
            opening_hold_rules,
        ),
        "opening_completed_hold_rules": opening_hold_rules,
        "critical_label_pairs": [
            ["open", "closing"],
            ["closed", "opening"],
            ["opening", "open"],
            ["closing", "closed"],
        ],
        "retention_rule": (
            "Retain current gate labels only when the state camera shows no motion "
            "cue; after an input opening_to_open transition, keep open in the short "
            "future unless all labeled ships are ship_berthed."
        ),
    }


def _observed_gate_transitions(
    sequence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    transitions = []
    previous: Optional[dict[str, Any]] = None
    for state in sequence:
        if previous is None:
            previous = state
            continue
        for gate_key in ("upper_gate_state", "lower_gate_state"):
            before = previous.get(gate_key)
            after = state.get(gate_key)
            if before == after:
                continue
            transitions.append(
                {
                    "gate": gate_key,
                    "from": before,
                    "to": after,
                    "from_frame_index": previous.get("frame_index"),
                    "to_frame_index": state.get("frame_index"),
                    "to_relative_time_sec": state.get("relative_time_sec"),
                }
            )
        previous = state
    return transitions


def _ship_berthing_status(frames: list[dict[str, Any]]) -> dict[str, Any]:
    latest = frames[-1] if frames else {}
    instances = _ship_instances_from_frame(latest)
    num_instances = len(instances)
    num_berthed = sum(1 for instance in instances if _is_berthed_ship(instance))
    flat_features = latest.get("flat_perception_features") or {}
    detected_ship_count_2d = _safe_float(flat_features.get("camera_num_ship_detections"))
    detected_ship_count_3d = _safe_float(flat_features.get("lidar_num_ship_detections"))
    return {
        "berth_label": "ship_berthed",
        "num_labeled_ship_instances": num_instances,
        "num_labeled_berthed_ship_instances": num_berthed,
        "ship_berthing_labels_available": bool(num_instances),
        "all_labeled_ship_instances_berthed": bool(
            num_instances and num_berthed == num_instances
        ),
        "detected_ship_count_2d": detected_ship_count_2d,
        "detected_ship_count_3d": detected_ship_count_3d,
        "gate_closing_precondition": (
            "An open gate may transition to closing only when all ships are berthed."
        ),
    }


def _ship_instances_from_frame(frame: dict[str, Any]) -> list[dict[str, Any]]:
    instances = frame.get("ship_instances")
    if isinstance(instances, list):
        return [instance for instance in instances if isinstance(instance, dict)]
    raw_instances = frame.get("instances_3d")
    if not isinstance(raw_instances, list):
        return []
    return [
        instance
        for instance in raw_instances
        if isinstance(instance, dict)
        and (
            bool(instance.get("ship_intentions"))
            or instance.get("category") in SHIP_2D_CLASSES
            or instance.get("category") in SHIP_3D_CLASSES
        )
    ]


def _is_berthed_ship(instance: dict[str, Any]) -> bool:
    return "ship_berthed" in {
        str(label)
        for label in (instance.get("ship_intentions") or [])
        if label is not None
    }


def _future_gate_domain_rules(
    current_state: dict[str, Any],
    observed_transitions: list[dict[str, Any]],
    ship_status: dict[str, Any],
    opening_hold_rules: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    rules = []
    if ship_status.get("all_labeled_ship_instances_berthed", False):
        for gate_key in ("upper_gate_state", "lower_gate_state"):
            if current_state.get(gate_key) == "closing" or _observed_open_to_closing(
                observed_transitions,
                gate_key,
            ):
                rules.append(
                    {
                        "gate": gate_key,
                        "forced_future_label": "closing",
                        "condition": (
                            "input already shows open_to_closing and all labeled ships "
                            "are ship_berthed"
                        ),
                    }
                )
    rules.extend(opening_hold_rules or [])
    return rules


def _opening_completed_hold_rules(
    current_state: dict[str, Any],
    observed_transitions: list[dict[str, Any]],
    ship_status: dict[str, Any],
) -> list[dict[str, Any]]:
    if ship_status.get("all_labeled_ship_instances_berthed", False):
        return []
    rules = []
    for gate_key in ("upper_gate_state", "lower_gate_state"):
        if current_state.get(gate_key) != "open":
            continue
        if not _observed_opening_to_open_completed(observed_transitions, gate_key):
            continue
        rules.append(
            {
                "gate": gate_key,
                "forced_future_label": "open",
                "condition": (
                    "input shows opening_to_open completed; short horizon remains "
                    "open unless all labeled ships are ship_berthed"
                ),
                "exception": (
                    "If all labeled ships are ship_berthed, the open gate may start "
                    "closing."
                ),
            }
        )
    return rules


def _observed_opening_to_open_completed(
    observed_transitions: list[dict[str, Any]],
    gate_key: str,
) -> bool:
    gate_transitions = [
        transition
        for transition in observed_transitions
        if transition.get("gate") == gate_key
    ]
    if not gate_transitions:
        return False
    last_transition = gate_transitions[-1]
    return (
        last_transition.get("from") == "opening"
        and last_transition.get("to") == "open"
    )


def _observed_open_to_closing(
    observed_transitions: list[dict[str, Any]],
    gate_key: str,
) -> bool:
    return any(
        transition.get("gate") == gate_key
        and transition.get("from") == "open"
        and transition.get("to") == "closing"
        for transition in observed_transitions
    )


def _candidate_future_gate_checks(
    current_state: dict[str, Any],
    ship_status: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = []
    for gate_key, camera in (
        ("upper_gate_state", "CAM_3"),
        ("lower_gate_state", "CAM_8"),
    ):
        current_label = current_state.get(gate_key)
        competing_label = {
            "open": "closing",
            "closed": "opening",
            "opening": "open",
            "closing": "closed",
        }.get(current_label)
        if competing_label is None:
            continue
        checks.append(
            {
                "gate": gate_key,
                "state_camera": camera,
                "current_label": current_label,
                "competing_future_label": competing_label,
                "confusing_pair": [current_label, competing_label],
                "open_to_closing_requires_all_ships_berthed": bool(
                    current_label == "open" and competing_label == "closing"
                ),
                "all_labeled_ship_instances_berthed": bool(
                    ship_status.get("all_labeled_ship_instances_berthed", False)
                ),
            }
        )
    return checks


def _water_surface_task(
    sequence: dict[str, Any],
    last_input_frame: dict[str, Any],
    target_frame: dict[str, Any],
    wave_labels_by_sample_camera: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    target_state = target_frame["lock_state"]
    wave_rule = _wave_annotation_rule(target_state["water_state"])
    wave_label = _matching_wave_label(target_frame, wave_rule, wave_labels_by_sample_camera)
    water_delta = _safe_float(target_state.get("water_level")) - _safe_float(
        last_input_frame["lock_state"].get("water_level")
    )
    surface_task = _water_surface_fields(
        water_state=target_state["water_state"],
        wave_rule=wave_rule,
        wave_label=wave_label,
    )
    surface_task.update(
        {
            "numeric_water_level_available": target_state.get("water_level") is not None,
            "water_level_delta_from_last_input_to_target": water_delta,
            "target_water_state": target_state["water_state"],
        }
    )
    return surface_task


def _current_water_surface_task(
    first_frame: dict[str, Any],
    current_frame: dict[str, Any],
    wave_labels_by_sample_camera: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    current_state = current_frame["lock_state"]
    wave_rule = _wave_annotation_rule(current_state["water_state"])
    wave_label = _matching_wave_label(current_frame, wave_rule, wave_labels_by_sample_camera)
    water_delta = _safe_float(current_state.get("water_level")) - _safe_float(
        first_frame["lock_state"].get("water_level")
    )
    surface_task = _water_surface_fields(
        water_state=current_state["water_state"],
        wave_rule=wave_rule,
        wave_label=wave_label,
    )
    surface_task.update(
        {
            "numeric_water_level_available": current_state.get("water_level") is not None,
            "water_level_delta_from_first_selected_to_current": water_delta,
            "current_water_state": current_state["water_state"],
        }
    )
    return surface_task


def _water_surface_fields(
    water_state: str,
    wave_rule: dict[str, str],
    wave_label: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if wave_label is not None:
        camera = wave_label.get("camera") or wave_rule.get("camera")
        region_id = wave_label.get("region_id") or wave_rule.get("region_id")
        region_description = (
            wave_label.get("region_description") or wave_rule.get("region_description")
        )
        image_verified = bool(wave_label.get("image_verified", False))
        return {
            "visual_check_required": True,
            "water_surface_wave_expected": bool(wave_label.get("wave_expected", False)),
            "wave_annotation_source": wave_label.get("label_source", "wave_label_file"),
            "target_wave_camera": camera,
            "target_wave_region_id": region_id,
            "target_wave_region_description": region_description,
            "image_level_waterline_annotation_required": bool(
                wave_label.get("image_level_waterline_annotation_required", False)
            ),
            "wave_label_image_verified": image_verified,
            "reason": _wave_label_reason(water_state, wave_label, image_verified),
            "scene_has_manual_wave_label": image_verified,
        }
    return {
        "visual_check_required": bool(wave_rule),
        "water_surface_wave_expected": bool(wave_rule),
        "wave_annotation_source": (
            "derived_from_water_state_target_region_rule" if wave_rule else "none"
        ),
        "target_wave_camera": wave_rule.get("camera") if wave_rule else None,
        "target_wave_region_id": wave_rule.get("region_id") if wave_rule else None,
        "target_wave_region_description": (
            wave_rule.get("region_description") if wave_rule else None
        ),
        "image_level_waterline_annotation_required": False,
        "wave_label_image_verified": False,
        "reason": _wave_reason(water_state, wave_rule),
        "scene_has_manual_wave_label": False,
    }


def _wave_annotation_rule(water_state: str) -> dict[str, str]:
    return WAVE_ANNOTATION_RULES.get(water_state, {})


def _wave_reason(water_state: str, wave_rule: dict[str, str]) -> str:
    if not wave_rule:
        return (
            "water_state is not filling or emptying, so no targeted wave annotation "
            "is derived from the current rule. Numeric water_level remains available."
        )
    return (
        f"water_state is {water_state}; inspect {wave_rule['camera']} "
        f"{wave_rule['region_description']} for water-surface waves. Do not create "
        "image-level waterline labels; use existing numeric water_level."
    )


def _wave_label_reason(
    water_state: str,
    wave_label: dict[str, Any],
    image_verified: bool,
) -> str:
    verification = "image-verified" if image_verified else "unverified"
    return (
        f"External targeted wave label says wave_expected="
        f"{bool(wave_label.get('wave_expected', False))} for water_state={water_state}; "
        f"label is {verification}. Numeric water_level remains available."
    )


def _matching_wave_label(
    frame: dict[str, Any],
    wave_rule: dict[str, str],
    wave_labels_by_sample_camera: dict[tuple[str, str], dict[str, Any]],
) -> Optional[dict[str, Any]]:
    sample_token = frame["sample_token"]
    if wave_rule:
        label = wave_labels_by_sample_camera.get((sample_token, wave_rule["camera"]))
        if label is not None:
            return label
    for channel in STATE_CAMERA_CHANNELS:
        label = wave_labels_by_sample_camera.get((sample_token, channel))
        if label is not None:
            return label
    return None


def _mooring_confidence_evidence(frame_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    crew_count = 0
    mooring_line_count = 0
    ship_2d_count = 0
    ship_3d_count = 0
    for frame in frame_summaries:
        for image in frame["images"].values():
            summary = image.get("perception_2d_summary") or {}
            counts = summary.get("counts_by_class") or {}
            crew_count += int(counts.get("Crew_member", 0))
            mooring_line_count += int(counts.get("Mooring_line", 0))
            ship_2d_count += sum(int(counts.get(name, 0)) for name in SHIP_2D_CLASSES)

        lidar_summary = frame["lidar"].get("perception_3d_summary") or {}
        counts_3d = lidar_summary.get("counts_by_class") or {}
        ship_3d_count += sum(int(counts_3d.get(name, 0)) for name in SHIP_3D_CLASSES)

    return {
        "crew_count_2d": crew_count,
        "mooring_line_count_2d": mooring_line_count,
        "ship_count_2d": ship_2d_count,
        "ship_count_3d": ship_3d_count,
        "mooring_confidence_boost_present": bool(
            crew_count > 0 and mooring_line_count > 0 and (ship_2d_count > 0 or ship_3d_count > 0)
        ),
        "weak_rule": (
            "Crew_member + Mooring_line + ship detection should increase confidence in "
            "berthed/moored behavior, but missing mooring lines must not rule it out "
            "because occlusion is common."
        ),
    }


def _ship_intentions(frame: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for instance in frame["instances_3d"]:
        intentions = instance.get("ship_intentions", [])
        if not intentions:
            continue
        items.append(
            {
                "instance_token": instance.get("instance_token"),
                "category": instance.get("category"),
                "ship_intentions": intentions,
            }
        )
    return items


def _ship_instance_context(frame: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for instance in frame.get("instances_3d", []):
        intentions = [
            str(label)
            for label in (instance.get("ship_intentions") or [])
            if label is not None
        ]
        category = instance.get("category")
        if (
            not intentions
            and category not in SHIP_2D_CLASSES
            and category not in SHIP_3D_CLASSES
        ):
            continue
        item = {
            "instance_token": instance.get("instance_token"),
            "category": category,
            "ship_intentions": intentions,
            "translation_xy": _rounded_xy(instance.get("translation")),
            "velocity_xy": _rounded_xy(instance.get("velocity")),
            "num_lidar_points": _optional_int(instance.get("num_lidar_points")),
        }
        items.append({key: value for key, value in item.items() if value is not None})
    return items


def _rounded_xy(value: Any) -> Optional[list[float]]:
    if not isinstance(value, list) or len(value) < 2:
        return None
    return [round(float(value[0]), 3), round(float(value[1]), 3)]


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _select_keyframe_indices(indices: list[int], max_keyframes: int) -> list[int]:
    if len(indices) <= max_keyframes:
        return list(indices)
    if max_keyframes <= 1:
        return [indices[-1]]
    selected = []
    last = len(indices) - 1
    for i in range(max_keyframes):
        selected.append(indices[round(i * last / (max_keyframes - 1))])
    return sorted(set(selected))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_perception_cache(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    payload = _load_json(path)
    metadata = payload["metadata"]
    if metadata["missing_2d_camera_predictions"] != 0:
        raise ValueError(f"2D perception cache has missing predictions: {path}")
    if metadata["missing_3d_frame_predictions"] != 0:
        raise ValueError(f"3D perception cache has missing predictions: {path}")
    return {item["sample_token"]: item for item in payload["frames"]}, _detector_sources(metadata)


def _load_wave_labels(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        sample_token = item.get("sample_token")
        camera = item.get("camera")
        if not sample_token or not camera:
            continue
        key = (sample_token, camera)
        previous = labels.get(key)
        if previous is None or (
            bool(item.get("image_verified", False))
            and not bool(previous.get("image_verified", False))
        ):
            labels[key] = item
    return labels


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _default_output_name(task_mode: str, split: str) -> str:
    if task_mode == "prediction":
        return f"navlock_vlm_semantic_{split}.jsonl"
    return f"navlock_vlm_semantic_{task_mode}_{split}.jsonl"


if __name__ == "__main__":
    main()
