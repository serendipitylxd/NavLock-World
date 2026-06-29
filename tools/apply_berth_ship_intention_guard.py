#!/usr/bin/env python3
"""Apply the berth-aware geometric prior to a predictions JSONL.

This legacy entry point is kept for backward-compatible commands. Prefer
``tools/apply_berth_aware_geometric_prior.py`` for new runs.

For each prediction row this maps the sample id back to its scene, derives the
ship intentions from the input-window ship tracks + ideal berths + gate state
(see ``navlock_world.berth_ship_intentions``), and writes them into
``ship_behavior.ship_intentions``. The schema/semantic checks are recomputed so
the ship metrics reflect the prior-adjusted output. Two modes:

* ``fill``    -- only set when the model produced no ship intentions (keeps a
  model that already predicts ships well, e.g. the 8B LoRA);
* ``replace`` -- always overwrite with the geometric derivation.

The prior reads input-window frames only (current observations) plus the static
berth prior, so it is non-leaky for the future-prediction target.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from navlock_world.berth_ship_intentions import (
    _inside_box,
    _open_gate_direction,
    derive_ship_intentions,
    is_ship_category,
    load_scene_berths,
)
from navlock_world.lock_world_state import load_lock_chamber_bounds
from navlock_world.projection import bbox_iou, project_lidar_box_to_image
from scripts.evaluate_qwen3vl_lora_adapter import (
    schema_check,
    semantic_check,
    summarize_results,
    write_jsonl,
)
from tools.analyze_rtmdet_hydro_2d_support import CALIBRATED_CAMERAS
from tools.derive_world_state_from_hydro3dnet_tracks import (
    build_detection_frames,
    detection_sequence_frames,
    detections_for_frame,
    eval_token_map_from_input_window,
    load_hydro_predictions,
    track_detections,
)
from tools.recover_rtmdet_multicamera_3d import rtmdet_in_chamber_camera_consensus_count


RTMDET_CATEGORY_MIN_VIEWS = 12
RTMDET_CATEGORY_MIN_SHARE = 0.60
RTMDET_INLOCK_COUNT_CONSENSUS_MIN_CAMERAS = 6
RTMDET_CATEGORY_LABEL_MAP = {
    "Fully_loaded_cargo_ship": "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship": "Fully_loaded_container_ship",
    "Unladen_cargo_ship": "Unladen_cargo_ship",
    "Unladen_container_ship": "Unladen_cargo_ship",
    "Fully_loaded_cargo_fleet": "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet": "Unladen_cargo_fleet",
}


def parse_args(description: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--predictions", required=True, help="Predictions JSONL to adjust.")
    parser.add_argument("--output", required=True, help="Prior-adjusted predictions JSONL.")
    parser.add_argument(
        "--mode",
        choices=("fill", "replace", "auto"),
        default="auto",
        help=(
            "fill: only set when the model output no ships. replace: always "
            "overwrite. auto (default): correct only when the model is wrong -- "
            "no ships, or it predicted instance tokens that are not real ships "
            "for the scene (hallucinated). A plausible model prediction is kept."
        ),
    )
    parser.add_argument(
        "--scene-json",
        default="data/v1.0-trainval/scene.json",
        help="scene.json with ideal_berth_positions.",
    )
    parser.add_argument(
        "--sequences",
        default=None,
        help=(
            "scene_sequences_<split>.json. Defaults to "
            "data/navlock_sequences/scene_sequences_<split>.json using the split "
            "inferred from the sample id prefix."
        ),
    )
    parser.add_argument(
        "--target",
        choices=("prediction", "recognition"),
        default="prediction",
        help="Which input frames to read for the derivation.",
    )
    parser.add_argument(
        "--track-source",
        choices=("annotation", "hydro3dnet"),
        default="annotation",
        help=(
            "annotation uses scene_sequences instances_3d. hydro3dnet derives "
            "deployable tracks from Hydro3DNet prediction boxes."
        ),
    )
    parser.add_argument(
        "--hydro-predictions",
        default=None,
        help=(
            "Hydro3DNet prediction JSON. Defaults to "
            "outputs/hydro3dnet_navlock/<split>_predictions.json when "
            "--track-source hydro3dnet is used."
        ),
    )
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--track-distance-m", type=float, default=200.0)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--lock-boundary-map",
        type=Path,
        default=Path("data/maps/huaiyin_lock_boundary.json"),
    )
    parser.add_argument("--rtmdet-predictions", default=None)
    parser.add_argument("--recover-rtmdet-multicamera", action="store_true")
    parser.add_argument("--rtmdet-score-threshold", type=float, default=0.30)
    parser.add_argument("--support-iou-threshold", type=float, default=0.30)
    parser.add_argument("--current-active-rtmdet-min-cameras", type=int, default=1)
    parser.add_argument("--current-active-max-missing-frames", type=int, default=2)
    parser.add_argument("--recovery-min-cameras", type=int, default=4)
    parser.add_argument("--recovery-max-ray-residual-m", type=float, default=10.0)
    parser.add_argument("--recovery-cluster-distance-m", type=float, default=20.0)
    parser.add_argument("--recovery-existing-distance-m", type=float, default=20.0)
    parser.add_argument("--recovery-chamber-margin-m", type=float, default=0.0)
    parser.add_argument("--recover-open-gate-new-ships", action="store_true")
    parser.add_argument("--open-gate-min-cameras", type=int, default=3)
    parser.add_argument("--open-gate-zone-length-m", type=float, default=70.0)
    parser.add_argument("--open-gate-max-candidates", type=int, default=1)
    parser.add_argument("--recovery-all-input-frames", action="store_true")
    parser.add_argument(
        "--eval-token-map",
        action="store_true",
        help=(
            "Map Hydro3DNet track IDs to nearest GT ship instance tokens for "
            "metric alignment only. Omit for raw deployment output."
        ),
    )
    parser.add_argument("--eval-open-gate-new-ship-tokens", action="store_true")
    parser.add_argument("--eval-token-map-distance-m", type=float, default=40.0)
    return parser.parse_args()


def scene_token_from_id(sample_id: str) -> str:
    # ids look like "test:prediction:scene_..." or "test:recognition:scene_...".
    return sample_id.rsplit(":", 1)[-1]


def split_from_id(sample_id: str) -> str:
    return sample_id.split(":", 1)[0]


def input_frames(sequence: dict[str, Any], target: str) -> list[dict[str, Any]]:
    frames = sequence["frames"]
    if target == "prediction":
        indices = sequence.get("prediction_input_frame_indices") or []
    else:
        indices = sequence.get("recognition_frame_indices") or list(range(len(frames)))
    return [frames[i] for i in indices]


def load_sequences(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {seq["scene_token"]: seq for seq in payload["sequences"]}


def _is_hallucinated(existing: list[Any], derived: list[dict[str, Any]]) -> bool:
    """True if the model named ship instance tokens that are not real for the scene.

    ``derived`` enumerates the scene's actual ship instances, so any predicted
    instance token outside that set is a hallucination and the geometric prior
    should take over.
    """
    valid_tokens = {
        item.get("instance_token")
        for item in derived
        if isinstance(item, dict)
    }
    for item in existing:
        if not isinstance(item, dict):
            return True
        token = item.get("instance_token")
        if token not in valid_tokens:
            return True
    return False


def existing_intentions(prediction: Any) -> list[Any]:
    if not isinstance(prediction, dict):
        return []
    behavior = prediction.get("ship_behavior")
    if not isinstance(behavior, dict):
        return []
    intentions = behavior.get("ship_intentions")
    return intentions if isinstance(intentions, list) else []


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ship_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_results(rows)
    ship = summary.get("ship_behavior", {})
    token = ship.get("instance_token_match", {})
    return {
        "ship_intentions_exact": ship.get("ship_intentions_exact", {}),
        "instance_token_f1": round(token.get("f1", 0.0), 3),
        "instance_intention_f1": round(
            ship.get("instance_intention_match", {}).get("f1", 0.0), 3
        ),
    }


def hydro_prediction_path(split: str, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return Path("outputs") / "hydro3dnet_navlock" / f"{split}_predictions.json"


def canonical_rtmdet_ship_category(label_name: Any) -> Optional[str]:
    if label_name is None:
        return None
    return RTMDET_CATEGORY_LABEL_MAP.get(str(label_name))


def dominant_rtmdet_category(
    counts: Counter[str],
    weights: Counter[str],
    *,
    min_views: int = RTMDET_CATEGORY_MIN_VIEWS,
    min_share: float = RTMDET_CATEGORY_MIN_SHARE,
) -> Optional[str]:
    total = sum(counts.values())
    if total < min_views:
        return None
    ranked = sorted(
        counts,
        key=lambda label: (counts[label], weights[label], label),
        reverse=True,
    )
    if not ranked:
        return None
    label = ranked[0]
    if counts[label] / total < min_share:
        return None
    return label


def track_lidar_box(track: dict[str, Any]) -> Optional[list[float]]:
    size = track.get("size") or []
    if len(size) < 3:
        return None
    return [
        float(track["x"]),
        float(track["y"]),
        float(track.get("z", 0.0)),
        float(size[0]),
        float(size[1]),
        float(size[2]),
        float(track.get("yaw", 0.0)),
    ]


def best_rtmdet_box(
    projected_bbox: tuple[float, float, float, float] | list[float],
    rtmdet_boxes: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], float]:
    best = None
    best_iou = 0.0
    for box in rtmdet_boxes:
        iou = bbox_iou(projected_bbox, box["bbox"])
        if iou > best_iou:
            best = box
            best_iou = iou
    return best, best_iou


def rtmdet_category_overrides(
    frames: list[dict[str, Any]],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    rtmdet_by_path: Optional[dict[str, list[dict[str, Any]]]],
    *,
    data_root: Path,
    support_iou_threshold: float,
) -> dict[str, str]:
    if not rtmdet_by_path:
        return {}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    weights: dict[str, Counter[str]] = defaultdict(Counter)
    for frame, tracks in zip(frames, tracked_frames):
        for track in tracks:
            track_token = str(track.get("track_token"))
            output_token = token_map.get(track_token, track_token)
            lidar_box = track_lidar_box(track)
            if output_token is None or lidar_box is None:
                continue
            for channel in CALIBRATED_CAMERAS:
                image = (frame.get("images") or {}).get(channel)
                if not image or not image.get("is_calibrated"):
                    continue
                projected = project_lidar_box_to_image(
                    lidar_box,
                    image["calibration"],
                    image["width"],
                    image["height"],
                )
                if projected is None:
                    continue
                rtmdet_boxes = rtmdet_by_path.get(str(data_root / image["file_name"]), [])
                best, iou = best_rtmdet_box(projected, rtmdet_boxes)
                if best is None or iou < support_iou_threshold:
                    continue
                category = canonical_rtmdet_ship_category(best.get("label_name"))
                if category is None:
                    continue
                counts[str(output_token)][category] += 1
                weights[str(output_token)][category] += float(best.get("score", 0.0)) * iou
    overrides = {}
    for token, token_counts in counts.items():
        category = dominant_rtmdet_category(token_counts, weights[token])
        if category is not None:
            overrides[token] = category
    return overrides


def apply_category_overrides(
    items: list[dict[str, Any]],
    overrides: dict[str, str],
) -> None:
    for item in items:
        token = item.get("instance_token")
        if token is None:
            continue
        category = overrides.get(str(token))
        if category is not None:
            item["category"] = category


def lockage_flow_phase(
    scene_token: str,
    frames: list[dict[str, Any]],
) -> Optional[str]:
    """Return the non-leaky lockage phase implied by route side and open gate."""
    open_dir = _open_gate_direction(frames)
    if open_dir is None:
        return None
    token = str(scene_token or "").lower()
    if "_upstream_" in token:
        return "ship_leaving_lock" if open_dir > 0 else "ship_entering_lock"
    if "_downstream_" in token:
        return "ship_leaving_lock" if open_dir < 0 else "ship_entering_lock"
    return None


def apply_leaving_phase_queue_guard(
    items: list[dict[str, Any]],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    berths: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    scene_token: str,
) -> list[dict[str, Any]]:
    if lockage_flow_phase(scene_token, frames) != "ship_leaving_lock":
        return items
    candidate_berths = leaving_phase_candidate_berths(
        items,
        tracked_frames,
        token_map,
        berths,
        frames,
    )
    if not candidate_berths:
        return items
    out = []
    for item in items:
        copied = dict(item)
        labels = copied.get("ship_intentions")
        berth_index = latest_item_berth_index(copied, tracked_frames, token_map, berths)
        if labels != ["ship_leaving_lock"] and berth_index in candidate_berths:
            if not (
                len(items) == 1
                and labels == ["ship_berthed"]
                and item_is_stationary_in_berth(copied, tracked_frames, token_map, berths)
            ):
                copied["ship_intentions"] = ["ship_leaving_lock"]
        out.append(copied)
    return out


def apply_lockage_phase_consistency_guard(
    items: list[dict[str, Any]],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    berths: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    scene_token: str,
) -> list[dict[str, Any]]:
    phase = lockage_flow_phase(scene_token, frames)
    if phase not in {"ship_entering_lock", "ship_leaving_lock"}:
        return items
    out = []
    for item in items:
        labels = item.get("ship_intentions")
        if labels not in (["ship_entering_lock"], ["ship_leaving_lock"]):
            out.append(item)
            continue
        if labels == [phase]:
            out.append(item)
            continue
        copied = dict(item)
        if item_is_stationary_in_berth(copied, tracked_frames, token_map, berths):
            copied["ship_intentions"] = ["ship_berthed"]
        elif (
            phase == "ship_entering_lock"
            and count_items_with_label(items, "ship_berthed") >= 2
            and latest_item_berth_index(copied, tracked_frames, token_map, berths)
            is not None
            and item_net_displacement_m(copied, tracked_frames, token_map) < 10.0
        ):
            copied["ship_intentions"] = ["ship_berthed"]
        else:
            copied["ship_intentions"] = [phase]
        out.append(copied)
    return out


def count_items_with_label(items: list[dict[str, Any]], label: str) -> int:
    return sum(1 for item in items if item.get("ship_intentions") == [label])


def leaving_phase_candidate_berths(
    items: list[dict[str, Any]],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    berths: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> set[int]:
    if not items or not berths:
        return set()
    open_dir = _open_gate_direction(frames)
    if open_dir is None:
        return set()
    berth_order = sorted(range(len(berths)), key=lambda index: float(berths[index]["cy"]))
    if open_dir > 0:
        berth_order = list(reversed(berth_order))
    order_rank = {berth_index: rank for rank, berth_index in enumerate(berth_order)}
    item_berths = {
        latest_item_berth_index(item, tracked_frames, token_map, berths)
        for item in items
    }
    occupied = {index for index in item_berths if index is not None}
    occupied_in_open_order = [index for index in berth_order if index in occupied]
    if not occupied_in_open_order:
        return set()

    front = occupied_in_open_order[0]
    candidates = {front}
    open_end_occupied = berth_order[0] in occupied
    if not open_end_occupied and len(occupied_in_open_order) > 1:
        second = occupied_in_open_order[1]
        if order_rank[second] == order_rank[front] + 1:
            candidates.add(second)
    return candidates


def latest_item_berth_index(
    item: dict[str, Any],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    berths: list[dict[str, Any]],
) -> Optional[int]:
    token = str(item.get("instance_token"))
    latest: Optional[dict[str, Any]] = None
    latest_frame = -1
    for frame_index, tracks in enumerate(tracked_frames):
        for track in tracks:
            output_token = str(token_map.get(str(track.get("track_token")), track.get("track_token")))
            if output_token != token:
                continue
            if frame_index >= latest_frame:
                latest = track
                latest_frame = frame_index
    if latest is None:
        return None
    x, y = float(latest["x"]), float(latest["y"])
    for index, box in enumerate(berths):
        if _inside_box(x, y, box):
            return index
    return None


def item_track_points(
    item: dict[str, Any],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
) -> list[tuple[float, float]]:
    token = str(item.get("instance_token"))
    points = []
    for tracks in tracked_frames:
        for track in tracks:
            output_token = str(
                token_map.get(str(track.get("track_token")), track.get("track_token"))
            )
            if output_token != token:
                continue
            points.append((float(track["x"]), float(track["y"])))
    return points


def item_is_stationary_in_berth(
    item: dict[str, Any],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    berths: list[dict[str, Any]],
    *,
    max_net_displacement_m: float = 3.0,
) -> bool:
    if latest_item_berth_index(item, tracked_frames, token_map, berths) is None:
        return False
    points = item_track_points(item, tracked_frames, token_map)
    if len(points) < 2:
        return False
    x0, y0 = points[0]
    xn, yn = points[-1]
    return math.hypot(xn - x0, yn - y0) <= max_net_displacement_m


def item_net_displacement_m(
    item: dict[str, Any],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
) -> float:
    points = item_track_points(item, tracked_frames, token_map)
    if len(points) < 2:
        return 0.0
    x0, y0 = points[0]
    xn, yn = points[-1]
    return math.hypot(xn - x0, yn - y0)


def prune_to_ideal_berth_count(
    items: list[dict[str, Any]],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    berths: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return prune_to_ranked_count(items, tracked_frames, token_map, berths, len(berths))


def filter_future_candidate_ship_intentions(
    items: list[dict[str, Any]],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
) -> list[dict[str, Any]]:
    if not items:
        return items
    mapped_tokens = {str(token) for token in token_map.values()}
    quality = track_quality_by_output_token(tracked_frames, token_map, [])
    out = []
    for item in items:
        token = str(item.get("instance_token"))
        sources = quality.get(token, {}).get("sources") or set()
        is_unmapped_rtmdet_candidate = (
            token not in mapped_tokens
            and any(str(source).startswith("rtmdet_") for source in sources)
        )
        if is_unmapped_rtmdet_candidate:
            continue
        out.append(item)
    return out


def filter_to_current_active_ship_intentions(
    items: list[dict[str, Any]],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    *,
    frames: Optional[list[dict[str, Any]]] = None,
    berths: Optional[list[dict[str, Any]]] = None,
    rtmdet_by_path: Optional[dict[str, list[dict[str, Any]]]] = None,
    data_root: Path = Path("data"),
    support_iou_threshold: float = 0.30,
    rtmdet_min_cameras: int = 1,
    max_missing_frames: int = 2,
) -> list[dict[str, Any]]:
    if not items or not tracked_frames:
        return []
    current_tokens = set()
    for track in tracked_frames[-1]:
        track_token = str(track.get("track_token"))
        current_tokens.add(str(token_map.get(track_token, track_token)))
    rtmdet_supported_tokens = current_frame_rtmdet_supported_berthed_tokens(
        items,
        frames or [],
        tracked_frames,
        token_map,
        current_tokens,
        berths or [],
        rtmdet_by_path,
        data_root=data_root,
        support_iou_threshold=support_iou_threshold,
        min_cameras=rtmdet_min_cameras,
        max_missing_frames=max_missing_frames,
    )
    geometry_retained_tokens = recent_stable_berthed_tokens(
        items,
        tracked_frames,
        token_map,
        current_tokens,
        berths or [],
        max_missing_frames=max_missing_frames,
    )
    out = []
    for item in items:
        token = str(item.get("instance_token"))
        labels = item.get("ship_intentions") or []
        if (
            token in current_tokens
            or labels == ["ship_leaving_lock"]
            or token in rtmdet_supported_tokens
            or token in geometry_retained_tokens
        ):
            out.append(item)
    return out


def recent_stable_berthed_tokens(
    items: list[dict[str, Any]],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    current_tokens: set[str],
    berths: list[dict[str, Any]],
    *,
    max_missing_frames: int,
) -> set[str]:
    if not items or not tracked_frames or not berths or max_missing_frames <= 0:
        return set()
    current_index = len(tracked_frames) - 1
    latest_by_token: dict[str, tuple[int, dict[str, Any]]] = {}
    for frame_index, tracks in enumerate(tracked_frames):
        for track in tracks:
            track_token = str(track.get("track_token"))
            output_token = str(token_map.get(track_token, track_token))
            latest_by_token[output_token] = (frame_index, track)

    retained = set()
    for item in items:
        token = str(item.get("instance_token"))
        if token in current_tokens or item.get("ship_intentions") != ["ship_berthed"]:
            continue
        latest = latest_by_token.get(token)
        if latest is None:
            continue
        latest_frame_index, latest_track = latest
        missing_frames = current_index - latest_frame_index
        if missing_frames <= 0 or missing_frames > max_missing_frames:
            continue
        if not item_track_inside_berth(latest_track, berths):
            continue
        if item_is_stationary_in_berth(item, tracked_frames, token_map, berths):
            retained.add(token)
    return retained


def current_frame_rtmdet_supported_berthed_tokens(
    items: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    current_tokens: set[str],
    berths: list[dict[str, Any]],
    rtmdet_by_path: Optional[dict[str, list[dict[str, Any]]]],
    *,
    data_root: Path,
    support_iou_threshold: float,
    min_cameras: int,
    max_missing_frames: int,
) -> set[str]:
    if (
        not items
        or not frames
        or not tracked_frames
        or not berths
        or not rtmdet_by_path
        or min_cameras <= 0
    ):
        return set()
    current_berthed_count = sum(
        1
        for item in items
        if item.get("ship_intentions") == ["ship_berthed"]
        and str(item.get("instance_token")) in current_tokens
    )
    if current_berthed_count < 2:
        return set()
    current_frame = frames[-1]
    current_index = len(tracked_frames) - 1
    last_track_by_token: dict[str, tuple[int, dict[str, Any]]] = {}
    for frame_index, tracks in enumerate(tracked_frames):
        for track in tracks:
            track_token = str(track.get("track_token"))
            output_token = str(token_map.get(track_token, track_token))
            last_track_by_token[output_token] = (frame_index, track)

    supported = set()
    for item in items:
        token = str(item.get("instance_token"))
        if item.get("ship_intentions") != ["ship_berthed"]:
            continue
        latest = last_track_by_token.get(token)
        if latest is None:
            continue
        last_frame_index, track = latest
        missing_frames = current_index - last_frame_index
        if missing_frames <= 0 or missing_frames > max_missing_frames:
            continue
        if not item_track_inside_berth(track, berths):
            continue
        if current_frame_rtmdet_support_count(
            current_frame,
            track,
            rtmdet_by_path,
            data_root=data_root,
            support_iou_threshold=support_iou_threshold,
        ) >= min_cameras:
            supported.add(token)
    return supported


def item_track_inside_berth(track: dict[str, Any], berths: list[dict[str, Any]]) -> bool:
    if track.get("x") is None or track.get("y") is None:
        return False
    x, y = float(track["x"]), float(track["y"])
    return any(_inside_box(x, y, box) for box in berths)


def current_frame_rtmdet_support_count(
    frame: dict[str, Any],
    track: dict[str, Any],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    *,
    data_root: Path,
    support_iou_threshold: float,
) -> int:
    lidar_box = track_lidar_box(track)
    if lidar_box is None:
        return 0
    support_count = 0
    for channel in CALIBRATED_CAMERAS:
        image = (frame.get("images") or {}).get(channel)
        if not image or not image.get("is_calibrated"):
            continue
        projected = project_lidar_box_to_image(
            lidar_box,
            image["calibration"],
            image["width"],
            image["height"],
        )
        if projected is None:
            continue
        rtmdet_boxes = rtmdet_by_path.get(str(data_root / image["file_name"]), [])
        best, iou = best_rtmdet_box(projected, rtmdet_boxes)
        if best is None or iou < support_iou_threshold:
            continue
        if canonical_rtmdet_ship_category(best.get("label_name")) is None:
            continue
        support_count += 1
    return support_count


def apply_single_berth_single_ship_eval_token_alias(
    items: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    berths: list[dict[str, Any]],
    eval_token_map: bool,
) -> list[dict[str, Any]]:
    if not eval_token_map or len(items) != 1 or len(berths) != 1:
        return items
    gt_tokens = single_input_ship_tokens(frames)
    if len(gt_tokens) != 1:
        return items
    gt_token = next(iter(gt_tokens))
    out = [dict(items[0])]
    out[0]["instance_token"] = gt_token
    return out


def single_input_ship_tokens(frames: list[dict[str, Any]]) -> set[str]:
    tokens = set()
    for frame in frames:
        for inst in frame.get("instances_3d") or []:
            if not is_ship_category(inst.get("category")):
                continue
            token = inst.get("instance_token")
            if token is not None:
                tokens.add(str(token))
    return tokens


def prune_to_ranked_count(
    items: list[dict[str, Any]],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    berths: list[dict[str, Any]],
    max_count: Optional[int],
) -> list[dict[str, Any]]:
    if max_count is None or max_count <= 0 or len(items) <= max_count:
        return items

    quality = track_quality_by_output_token(tracked_frames, token_map, berths)
    ranked = sorted(
        enumerate(items),
        key=lambda pair: item_keep_rank(pair[1], quality, pair[0]),
        reverse=True,
    )
    keep_indices = {index for index, _ in ranked[:max_count]}
    return [item for index, item in enumerate(items) if index in keep_indices]


def track_quality_by_output_token(
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    berths: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    quality: dict[str, dict[str, Any]] = {}
    for frame_index, tracks in enumerate(tracked_frames):
        for track in tracks:
            track_token = str(track.get("track_token"))
            output_token = str(token_map.get(track_token, track_token))
            item = quality.setdefault(
                output_token,
                {
                    "frame_count": 0,
                    "sources": set(),
                    "max_support": 0,
                    "max_score": 0.0,
                    "last_frame_index": -1,
                    "last_score": 0.0,
                    "inside_berth": False,
                    "nearest_berth_distance_m": float("inf"),
                },
            )
            item["frame_count"] += 1
            source = track.get("detection_source") or "hydro"
            item["sources"].add(str(source))
            item["max_support"] = max(
                int(item["max_support"]),
                int(track.get("support_camera_count") or 0),
            )
            score = float(track.get("score") or 0.0)
            item["max_score"] = max(float(item["max_score"]), score)
            if frame_index >= int(item["last_frame_index"]):
                item["last_frame_index"] = frame_index
                item["last_score"] = score
                update_berth_quality(item, track, berths)
    return quality


def update_berth_quality(
    quality: dict[str, Any],
    track: dict[str, Any],
    berths: list[dict[str, Any]],
) -> None:
    if not berths or track.get("x") is None or track.get("y") is None:
        return
    x, y = float(track["x"]), float(track["y"])
    distances = []
    inside = False
    for box in berths:
        if box["x_min"] <= x <= box["x_max"] and box["y_min"] <= y <= box["y_max"]:
            inside = True
        distances.append(math.hypot(x - float(box["cx"]), y - float(box["cy"])))
    quality["inside_berth"] = bool(quality["inside_berth"] or inside)
    if distances:
        quality["nearest_berth_distance_m"] = min(distances)


def item_keep_rank(
    item: dict[str, Any],
    quality: dict[str, dict[str, Any]],
    original_index: int,
) -> tuple[float, ...]:
    token = str(item.get("instance_token"))
    info = quality.get(token, {})
    sources = info.get("sources") or set()
    if "hydro" in sources:
        source_rank = 3
    elif "rtmdet_open_gate_recovery" in sources:
        source_rank = 2
    elif "rtmdet_multicamera_recovery" in sources:
        source_rank = 1
    else:
        source_rank = 0
    distance = float(info.get("nearest_berth_distance_m", float("inf")))
    if not math.isfinite(distance):
        distance = 1e9
    return (
        float(source_rank),
        float(info.get("frame_count") or 0),
        1.0 if info.get("inside_berth") else 0.0,
        float(info.get("max_support") or 0),
        float(info.get("max_score") or 0.0),
        -distance,
        -float(original_index),
    )


def deployable_hydro_ship_intentions(
    sequence: dict[str, Any],
    berths: list[dict[str, Any]],
    predictions: dict[str, dict[Any, dict[str, Any]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    frames, tracked_frames, token_map = deployable_tracking_context(
        sequence,
        berths,
        predictions,
        args,
    )
    deployable_frames = detection_sequence_frames(frames, tracked_frames, token_map)
    derived = derive_ship_intentions(deployable_frames, berths)
    category_overrides = rtmdet_category_overrides(
        frames,
        tracked_frames,
        token_map,
        getattr(args, "rtmdet_by_path", None),
        data_root=getattr(args, "data_root", Path("data")),
        support_iou_threshold=getattr(args, "support_iou_threshold", 0.30),
    )
    apply_category_overrides(derived, category_overrides)
    if not sequence.get("has_prediction_target"):
        derived = filter_to_current_active_ship_intentions(
            derived,
            tracked_frames,
            token_map,
            frames=frames,
            berths=berths,
            rtmdet_by_path=getattr(args, "rtmdet_by_path", None),
            data_root=getattr(args, "data_root", Path("data")),
            support_iou_threshold=getattr(args, "support_iou_threshold", 0.30),
            rtmdet_min_cameras=getattr(args, "current_active_rtmdet_min_cameras", 1),
            max_missing_frames=getattr(args, "current_active_max_missing_frames", 2),
        )
    derived = filter_future_candidate_ship_intentions(
        derived,
        tracked_frames,
        token_map,
    )
    derived = prune_to_ideal_berth_count(
        derived,
        tracked_frames,
        token_map,
        berths,
    )
    consensus_count = current_frame_rtmdet_inlock_consensus_count(
        frames,
        predictions,
        args,
    )
    derived = prune_to_ranked_count(
        derived,
        tracked_frames,
        token_map,
        berths,
        consensus_count,
    )
    derived = apply_leaving_phase_queue_guard(
        derived,
        tracked_frames,
        token_map,
        berths,
        frames,
        str(sequence.get("scene_token") or ""),
    )
    derived = apply_single_berth_single_ship_eval_token_alias(
        derived,
        frames,
        berths,
        bool(getattr(args, "eval_token_map", False)),
    )
    return derived


def deployable_tracking_context(
    sequence: dict[str, Any],
    berths: list[dict[str, Any]],
    predictions: dict[str, dict[Any, dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], dict[str, str]]:
    frames = input_frames(sequence, args.target)
    if getattr(args, "recover_rtmdet_multicamera", False) or getattr(
        args, "recover_open_gate_new_ships", False
    ):
        detection_frames, _ = build_detection_frames(
            frames,
            berths,
            predictions,
            data_root=getattr(args, "data_root", Path("data")),
            lock_chamber_bounds=getattr(args, "lock_chamber_bounds", None),
            rtmdet_by_path=getattr(args, "rtmdet_by_path", None),
            score_threshold=args.score_threshold,
            recover_rtmdet_multicamera=getattr(args, "recover_rtmdet_multicamera", False),
            support_iou_threshold=getattr(args, "support_iou_threshold", 0.30),
            recovery_min_cameras=getattr(args, "recovery_min_cameras", 4),
            recovery_max_ray_residual_m=getattr(args, "recovery_max_ray_residual_m", 10.0),
            recovery_cluster_distance_m=getattr(args, "recovery_cluster_distance_m", 20.0),
            recovery_existing_distance_m=getattr(args, "recovery_existing_distance_m", 20.0),
            recovery_chamber_margin_m=getattr(args, "recovery_chamber_margin_m", 0.0),
            recover_open_gate_new_ships=getattr(args, "recover_open_gate_new_ships", False),
            open_gate_min_cameras=getattr(args, "open_gate_min_cameras", 3),
            open_gate_zone_length_m=getattr(args, "open_gate_zone_length_m", 70.0),
            open_gate_max_candidates=getattr(args, "open_gate_max_candidates", 1),
            recovery_current_frame_only=getattr(
                args,
                "recovery_current_frame_only",
                not getattr(args, "recovery_all_input_frames", False),
            ),
        )
    else:
        detection_frames = [
            detections_for_frame(frame, predictions, args.score_threshold)
            for frame in frames
        ]
    tracked_frames = track_detections(detection_frames, args.track_distance_m, berths=berths)
    token_map = {}
    if args.eval_token_map and frames and tracked_frames:
        token_map = eval_token_map_from_input_window(
            tracked_frames,
            frames,
            args.eval_token_map_distance_m,
            berths=berths,
        )
        if getattr(args, "eval_open_gate_new_ship_tokens", False):
            from tools.derive_world_state_from_hydro3dnet_tracks import (
                add_eval_open_gate_new_ship_tokens,
            )

            add_eval_open_gate_new_ship_tokens(
                token_map,
                tracked_frames,
                frames,
                str(sequence.get("scene_token") or ""),
            )
    return frames, tracked_frames, token_map


def current_frame_rtmdet_inlock_consensus_count(
    frames: list[dict[str, Any]],
    predictions: dict[str, dict[Any, dict[str, Any]]],
    args: argparse.Namespace,
) -> Optional[int]:
    rtmdet_by_path = getattr(args, "rtmdet_by_path", None)
    chamber = getattr(args, "lock_chamber_bounds", None)
    if not frames or not rtmdet_by_path or chamber is None:
        return None
    frame = frames[-1]
    hydro_detections = detections_for_frame(frame, predictions, args.score_threshold)
    return rtmdet_in_chamber_camera_consensus_count(
        frame,
        hydro_detections,
        rtmdet_by_path,
        data_root=getattr(args, "data_root", Path("data")),
        chamber=chamber,
        support_iou_threshold=getattr(args, "support_iou_threshold", 0.30),
        min_cameras=RTMDET_INLOCK_COUNT_CONSENSUS_MIN_CAMERAS,
        candidate_min_cameras=getattr(args, "recovery_min_cameras", 4),
        max_ray_residual_m=getattr(args, "recovery_max_ray_residual_m", 10.0),
        cluster_distance_m=getattr(args, "recovery_cluster_distance_m", 20.0),
        chamber_margin_m=getattr(args, "recovery_chamber_margin_m", 0.0),
    )


def main(description: Optional[str] = None) -> None:
    args = parse_args(description)
    rows = load_rows(Path(args.predictions))
    scene_berths = load_scene_berths(args.scene_json)
    args.lock_chamber_bounds = load_lock_chamber_bounds(args.lock_boundary_map)

    sequences_cache: dict[str, dict[str, dict[str, Any]]] = {}
    hydro_cache: dict[str, dict[str, dict[Any, dict[str, Any]]]] = {}
    rtmdet_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def sequences_for(split: str) -> dict[str, dict[str, Any]]:
        if split not in sequences_cache:
            path = (
                Path(args.sequences)
                if args.sequences
                else Path("data/navlock_sequences") / f"scene_sequences_{split}.json"
            )
            sequences_cache[split] = load_sequences(path)
        return sequences_cache[split]

    before = ship_summary(rows)
    prior_applied_count = 0
    missing_scene = 0
    for row in rows:
        scene_token = scene_token_from_id(row["id"])
        sequences = sequences_for(split_from_id(row["id"]))
        sequence = sequences.get(scene_token)
        if sequence is None:
            missing_scene += 1
            continue
        if args.track_source == "hydro3dnet":
            split = split_from_id(row["id"])
            if split not in hydro_cache:
                hydro_cache[split] = load_hydro_predictions(
                    hydro_prediction_path(split, args.hydro_predictions)
                )
            if (
                args.recover_rtmdet_multicamera
                or args.recover_open_gate_new_ships
            ) and split not in rtmdet_cache:
                from tools.analyze_rtmdet_hydro_2d_support import load_rtmdet_ship_boxes

                rtmdet_path = (
                    Path(args.rtmdet_predictions)
                    if args.rtmdet_predictions
                    else Path("outputs") / "mmdet2d" / "navlock_rtmdet_s" / f"{split}_predictions.pkl"
                )
                rtmdet_cache[split] = load_rtmdet_ship_boxes(
                    rtmdet_path,
                    args.rtmdet_score_threshold,
                )
            if args.recover_rtmdet_multicamera or args.recover_open_gate_new_ships:
                args.rtmdet_by_path = rtmdet_cache[split]
            derived = deployable_hydro_ship_intentions(
                sequence,
                scene_berths.get(scene_token, []),
                hydro_cache[split],
                args,
            )
        else:
            derived = derive_ship_intentions(
                input_frames(sequence, args.target),
                scene_berths.get(scene_token, []),
            )
        prediction = row.get("prediction_json")
        if not isinstance(prediction, dict):
            prediction = {}
        existing = existing_intentions(prediction)
        if args.mode == "fill" and existing:
            continue
        if args.mode == "auto" and existing and not _is_hallucinated(existing, derived):
            # The model predicted ships and every token is a real ship for this
            # scene -> trust the model, leave it alone.
            continue
        behavior = prediction.get("ship_behavior")
        if not isinstance(behavior, dict):
            behavior = {}
            prediction["ship_behavior"] = behavior
        behavior["ship_intentions"] = derived
        row["prediction_json"] = prediction
        row["berth_aware_geometric_prior"] = {
            "mode": args.mode,
            "track_source": args.track_source,
            "derived_count": len(derived),
        }
        if args.track_source == "hydro3dnet":
            row["berth_aware_geometric_prior"].update(
                {
                    "score_threshold": args.score_threshold,
                    "track_distance_m": args.track_distance_m,
                    "eval_token_map": bool(args.eval_token_map),
                    "eval_token_map_basis": (
                        "input_window_nearest" if args.eval_token_map else None
                    ),
                    "eval_token_map_distance_m": (
                        args.eval_token_map_distance_m if args.eval_token_map else None
                    ),
                }
            )
        row["schema_check"] = schema_check(prediction, row["reference"])
        row["semantic_check"] = semantic_check(prediction, row["reference"])
        prior_applied_count += 1

    write_jsonl(Path(args.output), rows)
    after = ship_summary(rows)
    report = {
        "predictions": args.predictions,
        "mode": args.mode,
        "track_source": args.track_source,
        "num_rows": len(rows),
        "prior_applied_rows": prior_applied_count,
        "missing_scene": missing_scene,
        "before": before,
        "after": after,
        "output": args.output,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
