#!/usr/bin/env python3
"""Diagnose RTMDet 2D support for Hydro-track ship-intention errors.

This script does not change the deployable fused baseline. It aligns the same
Hydro3DNet tracks used by the baseline with calibrated RTMDet ship detections,
then reports whether ship-intention errors look like detection/support failures
or berth-phase rule failures.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navlock_world.projection import bbox_iou, project_lidar_box_to_image  # noqa: E402
from navlock_world.berth_ship_intentions import load_scene_berths  # noqa: E402
from tools.analyze_rtmdet_hydro_2d_support import (  # noqa: E402
    CALIBRATED_CAMERAS,
    load_rtmdet_ship_boxes,
)
from tools.derive_world_state_from_hydro3dnet_tracks import (  # noqa: E402
    detections_for_frame,
    eval_token_map_from_input_window,
    load_hydro_predictions,
    track_detections,
)


DEFAULT_FUSED = Path(
    "outputs/fused_deployable_baseline/predictions_test24_fused_deployable.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--scene-json", type=Path, default=None)
    parser.add_argument("--fused-predictions", type=Path, default=DEFAULT_FUSED)
    parser.add_argument("--hydro-predictions", type=Path, default=None)
    parser.add_argument("--rtmdet-predictions", type=Path, default=None)
    parser.add_argument("--hydro-score-threshold", type=float, default=0.05)
    parser.add_argument("--track-distance-m", type=float, default=40.0)
    parser.add_argument("--eval-token-map-distance-m", type=float, default=40.0)
    parser.add_argument("--rtmdet-score-threshold", type=float, default=0.30)
    parser.add_argument("--support-iou-threshold", type=float, default=0.30)
    parser.add_argument(
        "--static-2d-motion-threshold",
        type=float,
        default=0.02,
        help="Normalized image-diagonal motion below this is treated as 2D-stable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rtmdet_hydro_support/ship_intention_error_support_test.json"),
    )
    parser.add_argument(
        "--print-scenes",
        action="store_true",
        help="Print full per-scene diagnostics to stdout instead of summary only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_file = args.sequence_file or (
        args.data_root / "navlock_sequences" / f"scene_sequences_{args.split}.json"
    )
    scene_json = args.scene_json or (args.data_root / "v1.0-trainval" / "scene.json")
    hydro_predictions = args.hydro_predictions or (
        Path("outputs") / "hydro3dnet_navlock" / f"{args.split}_predictions.json"
    )
    rtmdet_predictions = args.rtmdet_predictions or (
        Path("outputs") / "mmdet2d" / "navlock_rtmdet_s" / f"{args.split}_predictions.pkl"
    )

    sequences = {
        item["scene_token"]: item
        for item in json.loads(sequence_file.read_text(encoding="utf-8")).get("sequences", [])
        if item.get("scene_token")
    }
    rows = load_prediction_rows(args.fused_predictions)
    scene_berths = load_scene_berths(scene_json)
    hydro_by_token = load_hydro_predictions(hydro_predictions)
    rtmdet_by_path = load_rtmdet_ship_boxes(rtmdet_predictions, args.rtmdet_score_threshold)

    report = analyze_rows(
        rows,
        sequences,
        scene_berths,
        hydro_by_token,
        rtmdet_by_path,
        data_root=args.data_root,
        hydro_score_threshold=args.hydro_score_threshold,
        track_distance_m=args.track_distance_m,
        eval_token_map_distance_m=args.eval_token_map_distance_m,
        support_iou_threshold=args.support_iou_threshold,
        static_2d_motion_threshold=args.static_2d_motion_threshold,
    )
    report["settings"] = {
        "split": args.split,
        "sequence_file": str(sequence_file),
        "scene_json": str(scene_json),
        "fused_predictions": str(args.fused_predictions),
        "hydro_predictions": str(hydro_predictions),
        "rtmdet_predictions": str(rtmdet_predictions),
        "hydro_score_threshold": args.hydro_score_threshold,
        "track_distance_m": args.track_distance_m,
        "eval_token_map_distance_m": args.eval_token_map_distance_m,
        "rtmdet_score_threshold": args.rtmdet_score_threshold,
        "support_iou_threshold": args.support_iou_threshold,
        "static_2d_motion_threshold": args.static_2d_motion_threshold,
        "calibrated_cameras": list(CALIBRATED_CAMERAS),
    }

    stdout_payload = report if args.print_scenes else {
        "summary": report["summary"],
        "settings": report["settings"],
    }
    print(json.dumps(stdout_payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        args.output.write_text(text + "\n", encoding="utf-8")


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def analyze_rows(
    rows: list[dict[str, Any]],
    sequences: dict[str, dict[str, Any]],
    scene_berths: dict[str, list[dict[str, Any]]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    *,
    data_root: Path,
    hydro_score_threshold: float,
    track_distance_m: float,
    eval_token_map_distance_m: float,
    support_iou_threshold: float,
    static_2d_motion_threshold: float,
) -> dict[str, Any]:
    summary = new_summary()
    scene_reports = []
    for row in rows:
        scene_token = scene_token_from_row(row)
        if scene_token is None:
            summary["missing_scene_token_rows"] += 1
            continue
        sequence = sequences.get(scene_token)
        if sequence is None:
            summary["missing_sequence_rows"] += 1
            continue
        berths = scene_berths.get(scene_token, [])
        track_features = build_track_features(
            sequence,
            berths,
            hydro_predictions,
            rtmdet_by_path,
            data_root=data_root,
            hydro_score_threshold=hydro_score_threshold,
            track_distance_m=track_distance_m,
            eval_token_map_distance_m=eval_token_map_distance_m,
            support_iou_threshold=support_iou_threshold,
        )
        feature_index = index_features_by_output_token(track_features)
        errors = ship_intention_errors(row)
        scene_report = attach_features_to_errors(
            row,
            scene_token,
            errors,
            feature_index,
        )
        scene_reports.append(scene_report)
        update_summary(summary, scene_report, static_2d_motion_threshold)

    finalize_summary(summary)
    return {
        "summary": summary,
        "scenes": scene_reports,
    }


def scene_token_from_row(row: dict[str, Any]) -> Optional[str]:
    prediction = row.get("prediction_json")
    if isinstance(prediction, dict) and isinstance(prediction.get("scene_token"), str):
        return prediction["scene_token"]
    item_id = row.get("id")
    if isinstance(item_id, str) and ":" in item_id:
        return item_id.rsplit(":", 1)[-1]
    return None


def build_track_features(
    sequence: dict[str, Any],
    berths: list[dict[str, Any]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    *,
    data_root: Path,
    hydro_score_threshold: float,
    track_distance_m: float,
    eval_token_map_distance_m: float,
    support_iou_threshold: float,
) -> list[dict[str, Any]]:
    frames = sequence.get("frames") or []
    input_indices = sequence.get("prediction_input_frame_indices") or []
    input_frames = [frames[index] for index in input_indices]
    detection_frames = [
        detections_for_frame(frame, hydro_predictions, hydro_score_threshold)
        for frame in input_frames
    ]
    tracked_frames = track_detections(detection_frames, track_distance_m, berths=berths)
    token_map = {}
    if input_frames and tracked_frames:
        token_map = eval_token_map_from_input_window(
            tracked_frames,
            input_frames,
            eval_token_map_distance_m,
            berths=berths,
        )

    raw: dict[str, dict[str, Any]] = {}
    for ordinal, (frame, tracks) in enumerate(zip(input_frames, tracked_frames)):
        frame_index = input_indices[ordinal] if ordinal < len(input_indices) else ordinal
        for track in tracks:
            item = raw.setdefault(track["track_token"], new_track_feature(track, token_map))
            update_track_position(item, track, frame, frame_index)
            frame_support_count = 0
            frame_best_iou = 0.0
            for channel in CALIBRATED_CAMERAS:
                image = (frame.get("images") or {}).get(channel)
                if not image or not image.get("is_calibrated"):
                    continue
                box = track_lidar_box(track)
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
                rtmdet_boxes = rtmdet_by_path.get(str(data_root / image["file_name"]), [])
                best = best_rtmdet_match(projected, rtmdet_boxes, support_iou_threshold)
                best_iou = best["iou"] if best is not None else max(
                    (bbox_iou(projected, rtmdet["bbox"]) for rtmdet in rtmdet_boxes),
                    default=0.0,
                )
                update_track_projection(item, channel, projected, best_iou)
                frame_best_iou = max(frame_best_iou, best_iou)
                if best is None:
                    continue
                frame_support_count += 1
                update_track_match(item, channel, frame_index, image, best)
            item["max_supported_cameras_single_frame"] = max(
                item["max_supported_cameras_single_frame"],
                frame_support_count,
            )
            item["support_camera_count_hist"][str(frame_support_count)] += 1
            if frame_support_count > 0:
                item["supported_frame_indices"].add(frame_index)
            item["frame_best_ious"].append(frame_best_iou)

    return [finalize_track_feature(item, berths) for item in raw.values()]


def new_track_feature(track: dict[str, Any], token_map: dict[str, str]) -> dict[str, Any]:
    track_token = track["track_token"]
    return {
        "track_token": track_token,
        "output_instance_token": token_map.get(track_token, track_token),
        "category": track.get("category"),
        "positions": [],
        "scores": [],
        "frame_indices": [],
        "projected_views": 0,
        "supported_views": 0,
        "best_iou_sum": 0.0,
        "best_iou_count": 0,
        "frame_best_ious": [],
        "supported_frame_indices": set(),
        "cameras_supported": set(),
        "max_supported_cameras_single_frame": 0,
        "support_camera_count_hist": Counter(),
        "matched_centers_by_camera": defaultdict(list),
        "projected_centers_by_camera": defaultdict(list),
    }


def update_track_position(
    item: dict[str, Any],
    track: dict[str, Any],
    frame: dict[str, Any],
    frame_index: int,
) -> None:
    item["category"] = track.get("category") or item["category"]
    time = frame.get("relative_time_sec", frame_index)
    item["positions"].append(
        {
            "time": float(time),
            "x": float(track["x"]),
            "y": float(track["y"]),
            "z": float(track.get("z", 0.0)),
        }
    )
    item["scores"].append(float(track.get("score", 0.0)))
    item["frame_indices"].append(int(frame_index))


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


def best_rtmdet_match(
    projected_bbox: tuple[float, float, float, float] | list[float],
    rtmdet_boxes: list[dict[str, Any]],
    support_iou_threshold: float,
) -> Optional[dict[str, Any]]:
    best_index = None
    best_iou = 0.0
    for index, rtmdet in enumerate(rtmdet_boxes):
        iou = bbox_iou(projected_bbox, rtmdet["bbox"])
        if iou > best_iou:
            best_index = index
            best_iou = iou
    if best_index is None or best_iou < support_iou_threshold:
        return None
    return {
        "index": best_index,
        "iou": best_iou,
        "box": rtmdet_boxes[best_index],
    }


def update_track_projection(
    item: dict[str, Any],
    channel: str,
    projected_bbox: tuple[float, float, float, float],
    best_iou: float,
) -> None:
    item["projected_views"] += 1
    item["best_iou_sum"] += float(best_iou)
    item["best_iou_count"] += 1
    item["projected_centers_by_camera"][channel].append(
        {"center": bbox_center(projected_bbox)}
    )


def update_track_match(
    item: dict[str, Any],
    channel: str,
    frame_index: int,
    image: dict[str, Any],
    best: dict[str, Any],
) -> None:
    item["supported_views"] += 1
    item["cameras_supported"].add(channel)
    item["matched_centers_by_camera"][channel].append(
        {
            "frame_index": int(frame_index),
            "center": bbox_center(best["box"]["bbox"]),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "iou": float(best["iou"]),
            "score": float(best["box"].get("score", 0.0)),
        }
    )


def bbox_center(box: tuple[float, float, float, float] | list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def finalize_track_feature(item: dict[str, Any], berths: list[dict[str, Any]]) -> dict[str, Any]:
    positions = item["positions"]
    first = positions[0] if positions else {"x": 0.0, "y": 0.0, "z": 0.0}
    last = positions[-1] if positions else first
    dx = last["x"] - first["x"]
    dy = last["y"] - first["y"]
    net = math.hypot(dx, dy)
    scores = item["scores"]
    camera_motions = [
        motion_summary_for_camera(channel, centers)
        for channel, centers in sorted(item["matched_centers_by_camera"].items())
    ]
    camera_motions = [motion for motion in camera_motions if motion is not None]
    norm_motions = [motion["normalized_displacement"] for motion in camera_motions]
    pixel_motions = [motion["pixel_displacement"] for motion in camera_motions]
    return {
        "track_token": item["track_token"],
        "output_instance_token": item["output_instance_token"],
        "category": item["category"],
        "frame_count": len(positions),
        "first_frame_index": item["frame_indices"][0] if item["frame_indices"] else None,
        "last_frame_index": item["frame_indices"][-1] if item["frame_indices"] else None,
        "start_xy": [round_float(first["x"]), round_float(first["y"])],
        "end_xy": [round_float(last["x"]), round_float(last["y"])],
        "delta_xy_m": [round_float(dx), round_float(dy)],
        "net_displacement_m": round_float(net),
        "delta_y_m": round_float(dy),
        "end_inside_berth": any(inside_box(last["x"], last["y"], box) for box in berths),
        "end_nearest_berth_distance_m": round_float(nearest_box_distance(last["x"], last["y"], berths)),
        "score_min": round_float(min(scores)) if scores else 0.0,
        "score_mean": round_float(sum(scores) / len(scores)) if scores else 0.0,
        "score_last": round_float(scores[-1]) if scores else 0.0,
        "projected_views": item["projected_views"],
        "supported_views": item["supported_views"],
        "supported_view_rate": round_float(
            item["supported_views"] / item["projected_views"]
            if item["projected_views"]
            else 0.0
        ),
        "supported_frame_count": len(item["supported_frame_indices"]),
        "supported_frame_rate": round_float(
            len(item["supported_frame_indices"]) / len(positions) if positions else 0.0
        ),
        "cameras_supported": sorted(item["cameras_supported"]),
        "max_supported_cameras_single_frame": item["max_supported_cameras_single_frame"],
        "mean_best_iou": round_float(
            item["best_iou_sum"] / item["best_iou_count"]
            if item["best_iou_count"]
            else 0.0
        ),
        "max_frame_best_iou": round_float(max(item["frame_best_ious"], default=0.0)),
        "support_camera_count_hist": dict(sorted(item["support_camera_count_hist"].items())),
        "rtmdet_2d_motion": {
            "camera_count_with_motion": len(camera_motions),
            "max_normalized_displacement": round_float(max(norm_motions, default=0.0)),
            "median_normalized_displacement": round_float(
                statistics.median(norm_motions) if norm_motions else 0.0
            ),
            "max_pixel_displacement": round_float(max(pixel_motions, default=0.0)),
            "per_camera": camera_motions,
        },
    }


def motion_summary_for_camera(
    channel: str,
    centers: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if len(centers) < 2:
        return None
    first = centers[0]
    last = centers[-1]
    x0, y0 = first["center"]
    x1, y1 = last["center"]
    pixel = math.hypot(x1 - x0, y1 - y0)
    diagonal = math.hypot(float(last["width"]), float(last["height"]))
    return {
        "channel": channel,
        "matched_frame_count": len(centers),
        "first_frame_index": first["frame_index"],
        "last_frame_index": last["frame_index"],
        "pixel_displacement": round_float(pixel),
        "normalized_displacement": round_float(pixel / diagonal if diagonal else 0.0),
        "first_score": round_float(first["score"]),
        "last_score": round_float(last["score"]),
        "mean_iou": round_float(sum(item["iou"] for item in centers) / len(centers)),
    }


def inside_box(x: float, y: float, box: dict[str, Any]) -> bool:
    return box["x_min"] <= x <= box["x_max"] and box["y_min"] <= y <= box["y_max"]


def nearest_box_distance(x: float, y: float, boxes: list[dict[str, Any]]) -> float:
    if not boxes:
        return 0.0
    return min(point_box_distance(x, y, box) for box in boxes)


def point_box_distance(x: float, y: float, box: dict[str, Any]) -> float:
    dx = max(float(box["x_min"]) - x, 0.0, x - float(box["x_max"]))
    dy = max(float(box["y_min"]) - y, 0.0, y - float(box["y_max"]))
    return math.hypot(dx, dy)


def index_features_by_output_token(
    features: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in features:
        token = feature.get("output_instance_token")
        if token is not None:
            out[str(token)].append(feature)
    for items in out.values():
        items.sort(key=lambda item: (item["last_frame_index"] or -1, item["frame_count"]), reverse=True)
    return dict(out)


def ship_intention_errors(row: dict[str, Any]) -> dict[str, Any]:
    prediction = row.get("prediction_json") if isinstance(row.get("prediction_json"), dict) else {}
    reference = row.get("reference") if isinstance(row.get("reference"), dict) else {}
    pred_items = normalize_intention_items((prediction.get("ship_behavior") or {}).get("ship_intentions"))
    ref_items = normalize_intention_items((reference.get("ship_behavior") or {}).get("ship_intentions"))
    pred_by_token = {item["instance_token"]: item for item in pred_items if item.get("instance_token")}
    ref_by_token = {item["instance_token"]: item for item in ref_items if item.get("instance_token")}

    correct = []
    wrong = []
    for token in sorted(set(pred_by_token) & set(ref_by_token)):
        pred_label = first_intention(pred_by_token[token])
        ref_label = first_intention(ref_by_token[token])
        item = {
            "instance_token": token,
            "predicted": pred_label,
            "reference": ref_label,
            "predicted_category": pred_by_token[token].get("category"),
            "reference_category": ref_by_token[token].get("category"),
        }
        if pred_label == ref_label:
            correct.append(item)
        else:
            wrong.append(item)

    missed = [
        {
            "instance_token": token,
            "reference": first_intention(ref_by_token[token]),
            "reference_category": ref_by_token[token].get("category"),
        }
        for token in sorted(set(ref_by_token) - set(pred_by_token))
    ]
    extra = [
        {
            "instance_token": token,
            "predicted": first_intention(pred_by_token[token]),
            "predicted_category": pred_by_token[token].get("category"),
        }
        for token in sorted(set(pred_by_token) - set(ref_by_token))
    ]
    return {
        "correct": correct,
        "wrong": wrong,
        "missed": missed,
        "extra": extra,
        "prediction_count": len(pred_items),
        "reference_count": len(ref_items),
    }


def normalize_intention_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if not isinstance(item, dict):
            continue
        token = item.get("instance_token")
        if token is None:
            continue
        intentions = item.get("ship_intentions")
        if not isinstance(intentions, list):
            intentions = []
        items.append(
            {
                "instance_token": str(token),
                "category": item.get("category"),
                "ship_intentions": [str(label) for label in intentions if label is not None],
            }
        )
    return items


def first_intention(item: dict[str, Any]) -> Optional[str]:
    labels = item.get("ship_intentions") or []
    return str(labels[0]) if labels else None


def attach_features_to_errors(
    row: dict[str, Any],
    scene_token: str,
    errors: dict[str, Any],
    feature_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    out = {
        "id": row.get("id"),
        "scene_token": scene_token,
        "prediction_count": errors["prediction_count"],
        "reference_count": errors["reference_count"],
        "correct": attach_track_features(errors["correct"], feature_index),
        "wrong": attach_track_features(errors["wrong"], feature_index),
        "missed": attach_track_features(errors["missed"], feature_index),
        "extra": attach_track_features(errors["extra"], feature_index),
    }
    out["counts"] = {
        "correct": len(out["correct"]),
        "wrong": len(out["wrong"]),
        "missed": len(out["missed"]),
        "extra": len(out["extra"]),
    }
    return out


def attach_track_features(
    items: list[dict[str, Any]],
    feature_index: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out = []
    for item in items:
        enriched = dict(item)
        enriched["track_features"] = feature_index.get(str(item["instance_token"]), [])
        out.append(enriched)
    return out


def new_summary() -> dict[str, Any]:
    return {
        "rows": 0,
        "missing_scene_token_rows": 0,
        "missing_sequence_rows": 0,
        "prediction_items": 0,
        "reference_items": 0,
        "correct_matched_intentions": 0,
        "wrong_matched_intentions": 0,
        "missed_reference_items": 0,
        "extra_prediction_items": 0,
        "wrong_confusions": Counter(),
        "wrong_with_track_features": 0,
        "wrong_end_inside_berth": 0,
        "wrong_any_2d_support": 0,
        "wrong_multicamera_2d_support": 0,
        "wrong_2d_stable": 0,
        "wrong_3d_net_displacements_m": [],
        "wrong_max_2d_motion_norm": [],
        "wrong_supported_view_rates": [],
        "extra_with_track_features": 0,
        "extra_any_2d_support": 0,
        "missed_without_track_features": 0,
    }


def update_summary(
    summary: dict[str, Any],
    scene_report: dict[str, Any],
    static_2d_motion_threshold: float,
) -> None:
    summary["rows"] += 1
    summary["prediction_items"] += scene_report["prediction_count"]
    summary["reference_items"] += scene_report["reference_count"]
    summary["correct_matched_intentions"] += scene_report["counts"]["correct"]
    summary["wrong_matched_intentions"] += scene_report["counts"]["wrong"]
    summary["missed_reference_items"] += scene_report["counts"]["missed"]
    summary["extra_prediction_items"] += scene_report["counts"]["extra"]

    for item in scene_report["wrong"]:
        summary["wrong_confusions"][f"{item.get('reference')}->{item.get('predicted')}"] += 1
        feature = primary_feature(item)
        if feature is None:
            continue
        summary["wrong_with_track_features"] += 1
        summary["wrong_3d_net_displacements_m"].append(feature["net_displacement_m"])
        summary["wrong_max_2d_motion_norm"].append(
            feature["rtmdet_2d_motion"]["max_normalized_displacement"]
        )
        summary["wrong_supported_view_rates"].append(feature["supported_view_rate"])
        if feature["end_inside_berth"]:
            summary["wrong_end_inside_berth"] += 1
        if feature["supported_views"] > 0:
            summary["wrong_any_2d_support"] += 1
        if len(feature["cameras_supported"]) >= 2:
            summary["wrong_multicamera_2d_support"] += 1
        if (
            feature["rtmdet_2d_motion"]["camera_count_with_motion"] > 0
            and feature["rtmdet_2d_motion"]["max_normalized_displacement"]
            <= static_2d_motion_threshold
        ):
            summary["wrong_2d_stable"] += 1

    for item in scene_report["extra"]:
        feature = primary_feature(item)
        if feature is None:
            continue
        summary["extra_with_track_features"] += 1
        if feature["supported_views"] > 0:
            summary["extra_any_2d_support"] += 1

    for item in scene_report["missed"]:
        if primary_feature(item) is None:
            summary["missed_without_track_features"] += 1


def primary_feature(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    features = item.get("track_features") or []
    return features[0] if features else None


def finalize_summary(summary: dict[str, Any]) -> None:
    for key in (
        "wrong_3d_net_displacements_m",
        "wrong_max_2d_motion_norm",
        "wrong_supported_view_rates",
    ):
        values = summary.pop(key)
        summary[f"{key}_mean"] = round_float(sum(values) / len(values)) if values else 0.0
        summary[f"{key}_median"] = round_float(statistics.median(values)) if values else 0.0
        summary[f"{key}_max"] = round_float(max(values)) if values else 0.0
    summary["wrong_confusions"] = dict(sorted(summary["wrong_confusions"].items()))


def round_float(value: float, ndigits: int = 4) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return round(float(value), ndigits)


if __name__ == "__main__":
    main()
