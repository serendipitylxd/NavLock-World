#!/usr/bin/env python3
"""Recover missing lock-chamber ship detections from multi-camera RTMDet boxes.

The recovery is intentionally conservative: only RTMDet boxes unmatched by
projected Hydro3DNet boxes are considered; at least two calibrated cameras must
support a candidate; and the triangulated 3D point must fall inside the lock
chamber before it can become a ship detection for occupancy or intention rules.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navlock_world.lock_world_state import (  # noqa: E402
    _chamber_bounds,
    load_lock_chamber_bounds,
    load_scene_berths,
)
from navlock_world.projection import (  # noqa: E402
    bbox_iou,
    camera_ray_to_lidar,
    project_lidar_box_to_image,
    triangulate_lidar_rays,
)
from tools.analyze_rtmdet_hydro_2d_support import (  # noqa: E402
    CALIBRATED_CAMERAS,
    load_rtmdet_ship_boxes,
)


DEFAULT_SHIP_SIZE = [60.0, 12.0, 6.0]
RECOVERABLE_2D_LABELS = {
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Unladen_container_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
}
OPEN_GATE_STATES = {"open", "opening"}
CALIBRATED_GATE_CAMERA_LAYOUT = {
    # Calibration/geometry check: CAM_1/2/4/5 see the lower-gate side well.
    "lower_gate_state": ("CAM_1", "CAM_2", "CAM_4", "CAM_5"),
    # CAM_6/7 primarily cover the upper-gate side.
    "upper_gate_state": ("CAM_6", "CAM_7"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--scene-json", type=Path, default=None)
    parser.add_argument("--lock-boundary-map", type=Path, default=Path("data/maps/huaiyin_lock_boundary.json"))
    parser.add_argument("--hydro-predictions", type=Path, default=None)
    parser.add_argument("--rtmdet-predictions", type=Path, default=None)
    parser.add_argument("--hydro-score-threshold", type=float, default=0.05)
    parser.add_argument("--rtmdet-score-threshold", type=float, default=0.30)
    parser.add_argument(
        "--allow-unknown-vessel-recovery",
        action="store_true",
        help=(
            "Deprecated no-op. RTMDet-only categories such as Unknown_vessel "
            "and Tugboat are not used for 3D count correction or recovery."
        ),
    )
    parser.add_argument("--support-iou-threshold", type=float, default=0.30)
    parser.add_argument("--min-cameras", type=int, default=4)
    parser.add_argument("--max-ray-residual-m", type=float, default=10.0)
    parser.add_argument("--cluster-distance-m", type=float, default=20.0)
    parser.add_argument("--existing-distance-m", type=float, default=20.0)
    parser.add_argument("--chamber-margin-m", type=float, default=0.0)
    parser.add_argument("--recover-open-gate-new-ships", action="store_true")
    parser.add_argument("--open-gate-min-cameras", type=int, default=3)
    parser.add_argument("--open-gate-zone-length-m", type=float, default=70.0)
    parser.add_argument("--open-gate-max-candidates", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rtmdet_hydro_support/rtmdet_multicamera_recovery_test.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_file = args.sequence_file or (
        args.data_root / "navlock_sequences" / f"scene_sequences_{args.split}.json"
    )
    scene_json = args.scene_json or (args.data_root / "v1.0-trainval" / "scene.json")
    lock_boundary_map = args.lock_boundary_map
    hydro_predictions = args.hydro_predictions or (
        Path("outputs") / "hydro3dnet_navlock" / f"{args.split}_predictions.json"
    )
    rtmdet_predictions = args.rtmdet_predictions or (
        Path("outputs") / "mmdet2d" / "navlock_rtmdet_s" / f"{args.split}_predictions.pkl"
    )

    sequences = json.loads(sequence_file.read_text(encoding="utf-8"))["sequences"]
    scene_berths = load_scene_berths(scene_json)
    lock_chamber_bounds = load_lock_chamber_bounds(lock_boundary_map)
    from tools.derive_world_state_from_hydro3dnet_tracks import load_hydro_predictions

    hydro_by_token = load_hydro_predictions(hydro_predictions)
    rtmdet_by_path = load_rtmdet_ship_boxes(rtmdet_predictions, args.rtmdet_score_threshold)
    summary = analyze_recovery(
        sequences,
        scene_berths,
        hydro_by_token,
        rtmdet_by_path,
        data_root=args.data_root,
        lock_chamber_bounds=lock_chamber_bounds,
        hydro_score_threshold=args.hydro_score_threshold,
        support_iou_threshold=args.support_iou_threshold,
        min_cameras=args.min_cameras,
        max_ray_residual_m=args.max_ray_residual_m,
        cluster_distance_m=args.cluster_distance_m,
        existing_distance_m=args.existing_distance_m,
        chamber_margin_m=args.chamber_margin_m,
        allow_unknown_vessel_recovery=args.allow_unknown_vessel_recovery,
        recover_open_gate_new_ships=args.recover_open_gate_new_ships,
        open_gate_min_cameras=args.open_gate_min_cameras,
        open_gate_zone_length_m=args.open_gate_zone_length_m,
        open_gate_max_candidates=args.open_gate_max_candidates,
    )
    summary["settings"] = {
        "split": args.split,
        "sequence_file": str(sequence_file),
        "scene_json": str(scene_json),
        "lock_boundary_map": str(lock_boundary_map),
        "hydro_predictions": str(hydro_predictions),
        "rtmdet_predictions": str(rtmdet_predictions),
        "hydro_score_threshold": args.hydro_score_threshold,
        "rtmdet_score_threshold": args.rtmdet_score_threshold,
        "allow_unknown_vessel_recovery": False,
        "recoverable_2d_labels": sorted(RECOVERABLE_2D_LABELS),
        "support_iou_threshold": args.support_iou_threshold,
        "min_cameras": args.min_cameras,
        "max_ray_residual_m": args.max_ray_residual_m,
        "cluster_distance_m": args.cluster_distance_m,
        "existing_distance_m": args.existing_distance_m,
        "chamber_margin_m": args.chamber_margin_m,
        "recover_open_gate_new_ships": args.recover_open_gate_new_ships,
        "open_gate_min_cameras": args.open_gate_min_cameras,
        "open_gate_zone_length_m": args.open_gate_zone_length_m,
        "open_gate_max_candidates": args.open_gate_max_candidates,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


def analyze_recovery(
    sequences: list[dict[str, Any]],
    scene_berths: dict[str, list[dict[str, Any]]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    *,
    data_root: Path,
    hydro_score_threshold: float,
    support_iou_threshold: float,
    min_cameras: int,
    max_ray_residual_m: float,
    cluster_distance_m: float,
    existing_distance_m: float,
    chamber_margin_m: float,
    lock_chamber_bounds: Optional[dict[str, float]] = None,
    allow_unknown_vessel_recovery: bool = False,
    recover_open_gate_new_ships: bool = False,
    open_gate_min_cameras: int = 3,
    open_gate_zone_length_m: float = 70.0,
    open_gate_max_candidates: int = 1,
) -> dict[str, Any]:
    from tools.derive_world_state_from_hydro3dnet_tracks import detections_for_frame

    summary = new_summary()
    examples = []
    for sequence in sequences:
        if not sequence.get("has_prediction_target"):
            continue
        chamber = lock_chamber_bounds or _chamber_bounds(
            scene_berths.get(sequence.get("scene_token"), [])
        )
        frames = sequence.get("frames") or []
        for frame_index in sequence.get("prediction_input_frame_indices") or []:
            frame = frames[frame_index]
            hydro = detections_for_frame(frame, hydro_predictions, hydro_score_threshold)
            candidates = recover_frame_detections(
                frame,
                hydro,
                rtmdet_by_path,
                data_root=data_root,
                chamber=chamber,
                support_iou_threshold=support_iou_threshold,
                min_cameras=min_cameras,
                max_ray_residual_m=max_ray_residual_m,
                cluster_distance_m=cluster_distance_m,
                existing_distance_m=existing_distance_m,
                chamber_margin_m=chamber_margin_m,
                allow_unknown_vessel_recovery=allow_unknown_vessel_recovery,
            )
            if recover_open_gate_new_ships:
                candidates.extend(
                    recover_open_gate_frame_detections(
                        frame,
                        hydro,
                        rtmdet_by_path,
                        data_root=data_root,
                        chamber=chamber,
                        support_iou_threshold=support_iou_threshold,
                        min_cameras=open_gate_min_cameras,
                        max_ray_residual_m=max_ray_residual_m,
                        cluster_distance_m=cluster_distance_m,
                        existing_distance_m=existing_distance_m,
                        chamber_margin_m=chamber_margin_m,
                        gate_zone_length_m=open_gate_zone_length_m,
                        max_candidates=open_gate_max_candidates,
                        allow_unknown_vessel_recovery=allow_unknown_vessel_recovery,
                    )
                )
            summary["frames"] += 1
            summary["hydro_detections"] += len(hydro)
            summary["recovered_candidates"] += len(candidates)
            for candidate in candidates:
                summary["support_camera_hist"][str(len(candidate.get("support_cameras") or []))] += 1
                summary["ray_residuals"].append(candidate.get("ray_residual_m", 0.0))
                if len(examples) < 20:
                    examples.append(
                        {
                            "scene_token": sequence.get("scene_token"),
                            "sample_token": frame.get("sample_token"),
                            "candidate": candidate,
                        }
                    )
    finalize_summary(summary)
    summary["examples"] = examples
    return summary


def recover_frame_detections(
    frame: dict[str, Any],
    hydro_detections: list[dict[str, Any]],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    *,
    data_root: Path,
    chamber: Optional[dict[str, float]],
    support_iou_threshold: float = 0.30,
    min_cameras: int = 2,
    max_ray_residual_m: float = 20.0,
    cluster_distance_m: float = 20.0,
    existing_distance_m: float = 20.0,
    chamber_margin_m: float = 0.0,
    allow_unknown_vessel_recovery: bool = False,
) -> list[dict[str, Any]]:
    all_observations = rtmdet_observations(
        frame,
        hydro_detections,
        rtmdet_by_path,
        data_root=data_root,
        support_iou_threshold=support_iou_threshold,
        unmatched_only=False,
        allow_unknown_vessel_recovery=allow_unknown_vessel_recovery,
    )
    rtmdet_lock_candidates = dedupe_candidates(
        candidate_clusters_from_observations(
            all_observations,
            chamber=chamber,
            min_cameras=min_cameras,
            max_ray_residual_m=max_ray_residual_m,
            cluster_distance_m=cluster_distance_m,
            chamber_margin_m=chamber_margin_m,
        ),
        cluster_distance_m,
    )
    hydro_lock_count = count_hydro_detections_in_chamber(
        hydro_detections,
        chamber,
        chamber_margin_m,
    )
    missing_count = max(0, len(rtmdet_lock_candidates) - hydro_lock_count)
    if missing_count <= 0:
        return []

    observations = rtmdet_observations(
        frame,
        hydro_detections,
        rtmdet_by_path,
        data_root=data_root,
        support_iou_threshold=support_iou_threshold,
        unmatched_only=True,
        allow_unknown_vessel_recovery=allow_unknown_vessel_recovery,
    )
    candidates = candidate_clusters_from_observations(
        observations,
        chamber=chamber,
        min_cameras=min_cameras,
        max_ray_residual_m=max_ray_residual_m,
        cluster_distance_m=cluster_distance_m,
        chamber_margin_m=chamber_margin_m,
    )
    candidates = [
        candidate
        for candidate in candidates
        if not near_existing_detection(candidate, hydro_detections, existing_distance_m)
    ]
    candidates = dedupe_candidates(candidates, cluster_distance_m)[:missing_count]
    detections = []
    for index, candidate in enumerate(candidates, start=1):
        detection = candidate_to_detection(candidate, index)
        detection["rtmdet_lock_ship_count"] = len(rtmdet_lock_candidates)
        detection["hydro_lock_ship_count"] = hydro_lock_count
        detection["recovery_missing_count"] = missing_count
        detections.append(detection)
    return detections


def rtmdet_in_chamber_camera_consensus_count(
    frame: dict[str, Any],
    hydro_detections: list[dict[str, Any]],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    *,
    data_root: Path,
    chamber: Optional[dict[str, float]],
    support_iou_threshold: float = 0.30,
    min_cameras: int = 6,
    candidate_min_cameras: int = 4,
    max_ray_residual_m: float = 10.0,
    cluster_distance_m: float = 20.0,
    chamber_margin_m: float = 0.0,
    allow_unknown_vessel_recovery: bool = False,
) -> Optional[int]:
    if chamber is None or min_cameras <= 0:
        return None
    observations = rtmdet_observations(
        frame,
        hydro_detections,
        rtmdet_by_path,
        data_root=data_root,
        support_iou_threshold=support_iou_threshold,
        unmatched_only=False,
        allow_unknown_vessel_recovery=allow_unknown_vessel_recovery,
    )
    candidates = dedupe_candidates(
        candidate_clusters_from_observations(
            observations,
            chamber=chamber,
            min_cameras=candidate_min_cameras,
            max_ray_residual_m=max_ray_residual_m,
            cluster_distance_m=cluster_distance_m,
            chamber_margin_m=chamber_margin_m,
        ),
        cluster_distance_m,
    )
    counts = unique_observation_counts_by_camera(candidates)
    agreeing = [
        counts.get(camera, 0)
        for camera in CALIBRATED_CAMERAS
        if counts.get(camera, 0) > 0
    ]
    if len(agreeing) < min_cameras:
        return None
    values = set(agreeing)
    if len(values) != 1:
        return None
    count = values.pop()
    return count if count > 0 else None


def unique_observation_counts_by_camera(
    candidates: list[dict[str, Any]],
) -> dict[str, int]:
    by_camera: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        for observation in candidate.get("observations") or []:
            channel = observation.get("channel")
            observation_id = observation.get("observation_id")
            if channel is None or observation_id is None:
                continue
            by_camera[str(channel)].add(str(observation_id))
    return {camera: len(observation_ids) for camera, observation_ids in by_camera.items()}


def recover_open_gate_frame_detections(
    frame: dict[str, Any],
    hydro_detections: list[dict[str, Any]],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    *,
    data_root: Path,
    chamber: Optional[dict[str, float]],
    support_iou_threshold: float = 0.30,
    min_cameras: int = 3,
    max_ray_residual_m: float = 5.0,
    cluster_distance_m: float = 20.0,
    existing_distance_m: float = 20.0,
    chamber_margin_m: float = 0.0,
    gate_zone_length_m: float = 70.0,
    max_candidates: int = 1,
    allow_unknown_vessel_recovery: bool = False,
) -> list[dict[str, Any]]:
    """Recover new ships visible near the currently open gate only."""
    open_gate = selected_open_gate(frame)
    if open_gate is None or chamber is None or max_candidates <= 0:
        return []
    camera_group = CALIBRATED_GATE_CAMERA_LAYOUT[open_gate]
    required_cameras = min(max(1, int(min_cameras)), len(camera_group))

    all_observations = observations_for_channels(
        rtmdet_observations(
            frame,
            hydro_detections,
            rtmdet_by_path,
            data_root=data_root,
            support_iou_threshold=support_iou_threshold,
            unmatched_only=False,
            allow_unknown_vessel_recovery=allow_unknown_vessel_recovery,
        ),
        camera_group,
    )
    all_candidates = gate_region_candidates(
        all_observations,
        chamber=chamber,
        open_gate=open_gate,
        min_cameras=required_cameras,
        max_ray_residual_m=max_ray_residual_m,
        cluster_distance_m=cluster_distance_m,
        chamber_margin_m=chamber_margin_m,
        gate_zone_length_m=gate_zone_length_m,
    )
    gate_candidate_count = len(dedupe_candidates(all_candidates, cluster_distance_m))
    hydro_gate_count = count_hydro_detections_in_open_gate_region(
        hydro_detections,
        chamber,
        open_gate,
        gate_zone_length_m,
        chamber_margin_m,
    )
    missing_count = min(int(max_candidates), max(0, gate_candidate_count - hydro_gate_count))
    if missing_count <= 0:
        return []

    observations = observations_for_channels(
        rtmdet_observations(
            frame,
            hydro_detections,
            rtmdet_by_path,
            data_root=data_root,
            support_iou_threshold=support_iou_threshold,
            unmatched_only=True,
            allow_unknown_vessel_recovery=allow_unknown_vessel_recovery,
        ),
        camera_group,
    )
    candidates = gate_region_candidates(
        observations,
        chamber=chamber,
        open_gate=open_gate,
        min_cameras=required_cameras,
        max_ray_residual_m=max_ray_residual_m,
        cluster_distance_m=cluster_distance_m,
        chamber_margin_m=chamber_margin_m,
        gate_zone_length_m=gate_zone_length_m,
    )
    candidates = [
        candidate
        for candidate in dedupe_candidates(candidates, cluster_distance_m)
        if not near_existing_detection(candidate, hydro_detections, existing_distance_m)
    ][:missing_count]

    detections = []
    for index, candidate in enumerate(candidates, start=1):
        detection = candidate_to_detection(
            candidate,
            index,
            detection_source="rtmdet_open_gate_recovery",
        )
        detection["open_gate_state"] = open_gate
        detection["open_gate_camera_group"] = list(camera_group)
        detection["rtmdet_open_gate_candidate_count"] = gate_candidate_count
        detection["hydro_open_gate_ship_count"] = hydro_gate_count
        detection["open_gate_recovery_missing_count"] = missing_count
        detections.append(detection)
    return detections


def selected_open_gate(frame: dict[str, Any]) -> Optional[str]:
    lock_state = frame.get("lock_state") or {}
    upper_open = lock_state.get("upper_gate_state") in OPEN_GATE_STATES
    lower_open = lock_state.get("lower_gate_state") in OPEN_GATE_STATES
    if lower_open and not upper_open:
        return "lower_gate_state"
    if upper_open and not lower_open:
        return "upper_gate_state"
    return None


def observations_for_channels(
    observations: list[dict[str, Any]],
    channels: tuple[str, ...],
) -> list[dict[str, Any]]:
    allowed = set(channels)
    return [item for item in observations if item.get("channel") in allowed]


def gate_region_candidates(
    observations: list[dict[str, Any]],
    *,
    chamber: dict[str, float],
    open_gate: str,
    min_cameras: int,
    max_ray_residual_m: float,
    cluster_distance_m: float,
    chamber_margin_m: float,
    gate_zone_length_m: float,
) -> list[dict[str, Any]]:
    candidates = candidate_clusters_from_observations(
        observations,
        chamber=chamber,
        min_cameras=min_cameras,
        max_ray_residual_m=max_ray_residual_m,
        cluster_distance_m=cluster_distance_m,
        chamber_margin_m=chamber_margin_m,
    )
    return [
        candidate
        for candidate in candidates
        if point_in_open_gate_region(
            candidate["point"],
            chamber,
            open_gate,
            gate_zone_length_m,
            chamber_margin_m,
        )
    ]


def rtmdet_observations(
    frame: dict[str, Any],
    hydro_detections: list[dict[str, Any]],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    *,
    data_root: Path,
    support_iou_threshold: float,
    unmatched_only: bool,
    allow_unknown_vessel_recovery: bool = False,
) -> list[dict[str, Any]]:
    observations = []
    for channel in CALIBRATED_CAMERAS:
        image = (frame.get("images") or {}).get(channel)
        if not image or not image.get("is_calibrated"):
            continue
        rtmdet_boxes = rtmdet_by_path.get(str(data_root / image["file_name"]), [])
        matched = matched_rtmdet_for_hydro(
            hydro_detections,
            image,
            rtmdet_boxes,
            support_iou_threshold,
        )
        for index, box in enumerate(rtmdet_boxes):
            if unmatched_only and index in matched:
                continue
            if not recoverable_label(
                box.get("label_name", "Unknown_vessel"),
                allow_unknown_vessel_recovery,
            ):
                continue
            ray = camera_ray_to_lidar(bbox_center(box["bbox"]), image["calibration"])
            if ray is None:
                continue
            observations.append(
                {
                    "observation_id": f"{channel}:{index}",
                    "channel": channel,
                    "bbox": box["bbox"],
                    "score": float(box.get("score", 0.0)),
                    "label_name": box.get("label_name", "Unknown_vessel"),
                    "image_width": int(image["width"]),
                    "image_height": int(image["height"]),
                    "ray": ray,
                }
            )
    return observations


def unmatched_rtmdet_observations(
    frame: dict[str, Any],
    hydro_detections: list[dict[str, Any]],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    *,
    data_root: Path,
    support_iou_threshold: float,
    allow_unknown_vessel_recovery: bool = False,
) -> list[dict[str, Any]]:
    return rtmdet_observations(
        frame,
        hydro_detections,
        rtmdet_by_path,
        data_root=data_root,
        support_iou_threshold=support_iou_threshold,
        unmatched_only=True,
        allow_unknown_vessel_recovery=allow_unknown_vessel_recovery,
    )


def recoverable_label(label_name: str, allow_unknown_vessel_recovery: bool = False) -> bool:
    return label_name in RECOVERABLE_2D_LABELS


def matched_rtmdet_for_hydro(
    hydro_detections: list[dict[str, Any]],
    image: dict[str, Any],
    rtmdet_boxes: list[dict[str, Any]],
    support_iou_threshold: float,
) -> set[int]:
    candidates = []
    for hydro_index, detection in enumerate(hydro_detections):
        box = detection_lidar_box(detection)
        if box is None:
            continue
        projected = project_lidar_box_to_image(
            box,
            image["calibration"],
            image["width"],
            image["height"],
        )
        if projected is None:
            continue
        for rtmdet_index, rtmdet in enumerate(rtmdet_boxes):
            iou = bbox_iou(projected, rtmdet["bbox"])
            if iou >= support_iou_threshold:
                candidates.append((iou, hydro_index, rtmdet_index))
    used_hydro = set()
    used_rtmdet = set()
    for _, hydro_index, rtmdet_index in sorted(candidates, reverse=True):
        if hydro_index in used_hydro or rtmdet_index in used_rtmdet:
            continue
        used_hydro.add(hydro_index)
        used_rtmdet.add(rtmdet_index)
    return used_rtmdet


def detection_lidar_box(detection: dict[str, Any]) -> Optional[list[float]]:
    if detection.get("box") is not None:
        box = [float(value) for value in detection["box"]]
        return box if len(box) >= 7 else None
    size = detection.get("size") or []
    if len(size) < 3:
        return None
    return [
        float(detection["x"]),
        float(detection["y"]),
        float(detection.get("z", 0.0)),
        float(size[0]),
        float(size[1]),
        float(size[2]),
        float(detection.get("yaw", 0.0)),
    ]


def bbox_center(box: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def candidate_clusters_from_observations(
    observations: list[dict[str, Any]],
    *,
    chamber: Optional[dict[str, float]],
    min_cameras: int,
    max_ray_residual_m: float,
    cluster_distance_m: float,
    chamber_margin_m: float,
) -> list[dict[str, Any]]:
    pair_candidates = []
    for left_index, left in enumerate(observations):
        for right_index in range(left_index + 1, len(observations)):
            right = observations[right_index]
            if left["channel"] == right["channel"]:
                continue
            triangulated = triangulate_lidar_rays([left["ray"], right["ray"]])
            if triangulated is None:
                continue
            point, residual = triangulated
            if residual > max_ray_residual_m:
                continue
            if not point_in_chamber(point, chamber, chamber_margin_m):
                continue
            pair_candidates.append(
                {
                    "point": point,
                    "residual": residual,
                    "observation_indices": {left_index, right_index},
                }
            )

    clusters: list[dict[str, Any]] = []
    for candidate in pair_candidates:
        matched_cluster = None
        for cluster in clusters:
            if np.linalg.norm(candidate["point"] - cluster["point_sum"] / cluster["count"]) <= cluster_distance_m:
                matched_cluster = cluster
                break
        if matched_cluster is None:
            matched_cluster = {
                "point_sum": np.zeros(3, dtype=float),
                "count": 0,
                "observation_indices": set(),
            }
            clusters.append(matched_cluster)
        matched_cluster["point_sum"] += candidate["point"]
        matched_cluster["count"] += 1
        matched_cluster["observation_indices"].update(candidate["observation_indices"])

    out = []
    for cluster in clusters:
        selected = select_one_observation_per_camera(
            [observations[index] for index in cluster["observation_indices"]]
        )
        if len({obs["channel"] for obs in selected}) < min_cameras:
            continue
        triangulated = triangulate_lidar_rays([obs["ray"] for obs in selected])
        if triangulated is None:
            continue
        point, residual = triangulated
        if residual > max_ray_residual_m:
            continue
        if not point_in_chamber(point, chamber, chamber_margin_m):
            continue
        out.append(
            {
                "point": point,
                "ray_residual_m": float(residual),
                "observations": selected,
            }
        )
    return out


def select_one_observation_per_camera(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_camera: dict[str, dict[str, Any]] = {}
    for observation in observations:
        channel = observation["channel"]
        current = by_camera.get(channel)
        if current is None or observation["score"] > current["score"]:
            by_camera[channel] = observation
    return list(by_camera.values())


def point_in_chamber(
    point: np.ndarray,
    chamber: Optional[dict[str, float]],
    margin_m: float = 0.0,
) -> bool:
    if chamber is None:
        return False
    x, y = float(point[0]), float(point[1])
    margin = float(margin_m)
    return (
        chamber["x_min"] - margin <= x <= chamber["x_max"] + margin
        and chamber["y_min"] - margin <= y <= chamber["y_max"] + margin
    )


def count_hydro_detections_in_chamber(
    hydro_detections: list[dict[str, Any]],
    chamber: Optional[dict[str, float]],
    margin_m: float = 0.0,
) -> int:
    count = 0
    for detection in hydro_detections:
        if detection.get("x") is None or detection.get("y") is None:
            continue
        point = np.array([float(detection["x"]), float(detection["y"]), float(detection.get("z", 0.0))])
        if point_in_chamber(point, chamber, margin_m):
            count += 1
    return count


def count_hydro_detections_in_open_gate_region(
    hydro_detections: list[dict[str, Any]],
    chamber: dict[str, float],
    open_gate: str,
    gate_zone_length_m: float,
    margin_m: float = 0.0,
) -> int:
    count = 0
    for detection in hydro_detections:
        if detection.get("x") is None or detection.get("y") is None:
            continue
        point = np.array([float(detection["x"]), float(detection["y"]), float(detection.get("z", 0.0))])
        if point_in_open_gate_region(point, chamber, open_gate, gate_zone_length_m, margin_m):
            count += 1
    return count


def point_in_open_gate_region(
    point: np.ndarray,
    chamber: dict[str, float],
    open_gate: str,
    gate_zone_length_m: float,
    margin_m: float = 0.0,
) -> bool:
    x, y = float(point[0]), float(point[1])
    margin = float(margin_m)
    if not (chamber["x_min"] - margin <= x <= chamber["x_max"] + margin):
        return False
    if open_gate == "lower_gate_state":
        return chamber["y_min"] - margin <= y <= chamber["y_min"] + float(gate_zone_length_m)
    if open_gate == "upper_gate_state":
        return chamber["y_max"] - float(gate_zone_length_m) <= y <= chamber["y_max"] + margin
    return False


def near_existing_detection(
    candidate: dict[str, Any],
    hydro_detections: list[dict[str, Any]],
    existing_distance_m: float,
) -> bool:
    point = candidate["point"]
    for detection in hydro_detections:
        if detection.get("x") is None or detection.get("y") is None:
            continue
        distance = math.hypot(float(point[0]) - float(detection["x"]), float(point[1]) - float(detection["y"]))
        if distance <= existing_distance_m:
            return True
    return False


def dedupe_candidates(
    candidates: list[dict[str, Any]],
    cluster_distance_m: float,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -len({obs["channel"] for obs in item["observations"]}),
            item["ray_residual_m"],
            -sum(obs["score"] for obs in item["observations"]),
        ),
    )
    kept = []
    for candidate in ordered:
        if any(
            np.linalg.norm(candidate["point"] - existing["point"]) <= cluster_distance_m
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def candidate_to_detection(
    candidate: dict[str, Any],
    index: int,
    *,
    detection_source: str = "rtmdet_multicamera_recovery",
) -> dict[str, Any]:
    observations = candidate["observations"]
    labels = [obs["label_name"] for obs in observations]
    label_name = Counter(labels).most_common(1)[0][0] if labels else "Unknown_vessel"
    category = recovered_3d_category(label_name)
    point = candidate["point"]
    cameras = sorted({obs["channel"] for obs in observations})
    return {
        "detection_id": f"rtmdet_recovery_{index:03d}",
        "category": category,
        "x": float(point[0]),
        "y": float(point[1]),
        "z": float(point[2]),
        "size": list(DEFAULT_SHIP_SIZE),
        "yaw": 0.0,
        "score": float(sum(obs["score"] for obs in observations) / len(observations)),
        "detection_source": detection_source,
        "support_cameras": cameras,
        "support_camera_count": len(cameras),
        "ray_residual_m": round(float(candidate["ray_residual_m"]), 4),
        "rtmdet_labels": sorted(set(labels)),
    }


def recovered_3d_category(label_name: str) -> str:
    if label_name in {
        "Fully_loaded_cargo_ship",
        "Fully_loaded_container_ship",
        "Unladen_cargo_ship",
        "Fully_loaded_cargo_fleet",
        "Unladen_cargo_fleet",
    }:
        return label_name
    if label_name == "Unladen_container_ship":
        return "Unladen_cargo_ship"
    if label_name == "Fully_loaded_cargo_fleet":
        return "Fully_loaded_cargo_fleet"
    return "Unladen_cargo_ship"


def new_summary() -> dict[str, Any]:
    return {
        "frames": 0,
        "hydro_detections": 0,
        "recovered_candidates": 0,
        "support_camera_hist": Counter(),
        "ray_residuals": [],
    }


def finalize_summary(summary: dict[str, Any]) -> None:
    residuals = summary.pop("ray_residuals")
    summary["support_camera_hist"] = dict(sorted(summary["support_camera_hist"].items()))
    summary["ray_residual_m_mean"] = (
        round(sum(residuals) / len(residuals), 4) if residuals else 0.0
    )
    summary["ray_residual_m_max"] = round(max(residuals), 4) if residuals else 0.0


if __name__ == "__main__":
    main()
