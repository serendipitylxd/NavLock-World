#!/usr/bin/env python3
"""Analyze calibrated RTMDet 2D support for Hydro3DNet ship detections."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navlock_world.projection import bbox_iou, project_lidar_box_to_image  # noqa: E402


CALIBRATED_CAMERAS = ("CAM_1", "CAM_2", "CAM_4", "CAM_5", "CAM_6", "CAM_7")
RTMDET_CLASSES = (
    "Building",
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Lock_gate",
    "Tree",
    "Unladen_cargo_ship",
    "Unladen_container_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
    "Lock_footbridge",
    "Crew_member",
    "Mooring_line",
    "Tugboat",
    "Unknown_vessel",
)
SHIP_2D_CLASSES = {
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Unladen_container_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
}
SHIP_3D_CLASSES = {
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--hydro-predictions", type=Path, default=None)
    parser.add_argument("--rtmdet-predictions", type=Path, default=None)
    parser.add_argument("--hydro-score-threshold", type=float, default=0.05)
    parser.add_argument("--rtmdet-score-threshold", type=float, default=0.30)
    parser.add_argument("--support-iou-threshold", type=float, default=0.30)
    parser.add_argument("--high-confidence-score", type=float, default=0.70)
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Analyze every sequence frame instead of prediction input frames only.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_file = args.sequence_file or (
        args.data_root / "navlock_sequences" / f"scene_sequences_{args.split}.json"
    )
    hydro_predictions = args.hydro_predictions or (
        Path("outputs") / "hydro3dnet_navlock" / f"{args.split}_predictions.json"
    )
    rtmdet_predictions = args.rtmdet_predictions or (
        Path("outputs") / "mmdet2d" / "navlock_rtmdet_s" / f"{args.split}_predictions.pkl"
    )

    sequences = json.loads(sequence_file.read_text(encoding="utf-8"))["sequences"]
    hydro_by_token = load_hydro_predictions(hydro_predictions)
    rtmdet_by_path = load_rtmdet_ship_boxes(rtmdet_predictions, args.rtmdet_score_threshold)
    summary = analyze_sequences(
        sequences,
        args.data_root,
        hydro_by_token,
        rtmdet_by_path,
        hydro_score_threshold=args.hydro_score_threshold,
        support_iou_threshold=args.support_iou_threshold,
        high_confidence_score=args.high_confidence_score,
        prediction_input_only=not args.all_frames,
    )
    summary["settings"] = {
        "split": args.split,
        "sequence_file": str(sequence_file),
        "hydro_predictions": str(hydro_predictions),
        "rtmdet_predictions": str(rtmdet_predictions),
        "hydro_score_threshold": args.hydro_score_threshold,
        "rtmdet_score_threshold": args.rtmdet_score_threshold,
        "support_iou_threshold": args.support_iou_threshold,
        "high_confidence_score": args.high_confidence_score,
        "prediction_input_only": not args.all_frames,
        "calibrated_cameras": list(CALIBRATED_CAMERAS),
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


def load_hydro_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {
        item["sample_token"]: item
        for item in json.loads(path.read_text(encoding="utf-8"))
        if item.get("sample_token")
    }


def load_rtmdet_ship_boxes(path: Path, score_threshold: float) -> dict[str, list[dict[str, Any]]]:
    with path.open("rb") as handle:
        predictions = pickle.load(handle)

    by_path: dict[str, list[dict[str, Any]]] = {}
    for item in predictions:
        pred = item["pred_instances"]
        scores = to_list(pred["scores"])
        labels = to_list(pred["labels"])
        boxes = pred["bboxes"].detach().cpu().tolist()
        kept = []
        for label, score, box in zip(labels, scores, boxes):
            label_index = int(label)
            if label_index < 0 or label_index >= len(RTMDET_CLASSES):
                continue
            label_name = RTMDET_CLASSES[label_index]
            if label_name not in SHIP_2D_CLASSES or float(score) < score_threshold:
                continue
            kept.append(
                {
                    "bbox": [float(value) for value in box],
                    "score": float(score),
                    "label_name": label_name,
                }
            )
        by_path[item["img_path"]] = kept
    return by_path


def to_list(value: Any) -> list[Any]:
    return value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)


def analyze_sequences(
    sequences: list[dict[str, Any]],
    data_root: Path,
    hydro_by_token: dict[str, dict[str, Any]],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    *,
    hydro_score_threshold: float,
    support_iou_threshold: float,
    high_confidence_score: float,
    prediction_input_only: bool,
) -> dict[str, Any]:
    summary = new_summary()
    for sequence in sequences:
        frames = selected_frames(sequence, prediction_input_only)
        for frame in frames:
            analyze_frame(
                frame,
                data_root,
                hydro_by_token.get(frame.get("sample_token"), {}),
                rtmdet_by_path,
                hydro_score_threshold=hydro_score_threshold,
                support_iou_threshold=support_iou_threshold,
                high_confidence_score=high_confidence_score,
                summary=summary,
            )
    finalize_summary(summary)
    return summary


def selected_frames(sequence: dict[str, Any], prediction_input_only: bool) -> list[dict[str, Any]]:
    frames = sequence.get("frames") or []
    if not prediction_input_only:
        return frames
    if not sequence.get("has_prediction_target"):
        return []
    return [frames[index] for index in sequence.get("prediction_input_frame_indices") or []]


def analyze_frame(
    frame: dict[str, Any],
    data_root: Path,
    hydro_prediction: dict[str, Any],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    *,
    hydro_score_threshold: float,
    support_iou_threshold: float,
    high_confidence_score: float,
    summary: dict[str, Any],
) -> None:
    summary["num_frames"] += 1
    hydro_detections = hydro_ship_detections(hydro_prediction, hydro_score_threshold)
    summary["hydro_ship_detections"] += len(hydro_detections)

    support_counts_by_detection = [0 for _ in hydro_detections]
    best_ious_by_detection = [0.0 for _ in hydro_detections]

    for channel in CALIBRATED_CAMERAS:
        image = (frame.get("images") or {}).get(channel)
        if not image or not image.get("is_calibrated"):
            continue
        image_path = str(data_root / image["file_name"])
        rtmdet_boxes = rtmdet_by_path.get(image_path, [])
        camera_summary = summary["per_camera"][channel]
        camera_summary["rtmdet_ship_boxes"] += len(rtmdet_boxes)
        camera_summary["rtmdet_high_conf_ship_boxes"] += sum(
            1 for box in rtmdet_boxes if box["score"] >= high_confidence_score
        )

        projected = []
        for det_index, detection in enumerate(hydro_detections):
            bbox = project_lidar_box_to_image(
                detection["box"],
                image["calibration"],
                image["width"],
                image["height"],
            )
            if bbox is None:
                continue
            projected.append({"det_index": det_index, "bbox": bbox})
            summary["hydro_projected_camera_views"] += 1
            camera_summary["hydro_projected_views"] += 1

            best_iou = max((bbox_iou(bbox, box["bbox"]) for box in rtmdet_boxes), default=0.0)
            best_ious_by_detection[det_index] = max(best_ious_by_detection[det_index], best_iou)
            camera_summary["best_iou_sum"] += best_iou
            camera_summary["best_iou_count"] += 1
            if best_iou >= support_iou_threshold:
                support_counts_by_detection[det_index] += 1
                summary["hydro_supported_camera_views"] += 1
                camera_summary["hydro_supported_views"] += 1

        matched_rtmdet = matched_rtmdet_indices(projected, rtmdet_boxes, support_iou_threshold)
        camera_summary["rtmdet_matched_to_hydro"] += len(matched_rtmdet)
        camera_summary["rtmdet_unmatched_to_hydro"] += len(rtmdet_boxes) - len(matched_rtmdet)
        camera_summary["rtmdet_high_conf_unmatched_to_hydro"] += sum(
            1
            for index, box in enumerate(rtmdet_boxes)
            if index not in matched_rtmdet and box["score"] >= high_confidence_score
        )

    for support_count, best_iou in zip(support_counts_by_detection, best_ious_by_detection):
        summary["hydro_support_camera_count_hist"][str(support_count)] += 1
        summary["hydro_best_iou_sum"] += best_iou
        summary["hydro_best_iou_count"] += 1
        if support_count > 0:
            summary["hydro_supported_detections_any_camera"] += 1
        else:
            summary["hydro_unsupported_detections"] += 1


def hydro_ship_detections(prediction: dict[str, Any], score_threshold: float) -> list[dict[str, Any]]:
    detections = []
    for index, (box, label_name, score) in enumerate(
        zip(
            prediction.get("boxes") or [],
            prediction.get("label_names") or [],
            prediction.get("scores") or [],
        )
    ):
        if label_name not in SHIP_3D_CLASSES or float(score) < score_threshold:
            continue
        detections.append(
            {
                "detection_id": index,
                "box": [float(value) for value in box],
                "label_name": label_name,
                "score": float(score),
            }
        )
    return detections


def matched_rtmdet_indices(
    projected_hydro: list[dict[str, Any]],
    rtmdet_boxes: list[dict[str, Any]],
    support_iou_threshold: float,
) -> set[int]:
    candidates = []
    for hydro_index, hydro in enumerate(projected_hydro):
        for rtmdet_index, rtmdet in enumerate(rtmdet_boxes):
            iou = bbox_iou(hydro["bbox"], rtmdet["bbox"])
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


def new_summary() -> dict[str, Any]:
    return {
        "num_frames": 0,
        "hydro_ship_detections": 0,
        "hydro_projected_camera_views": 0,
        "hydro_supported_camera_views": 0,
        "hydro_supported_detections_any_camera": 0,
        "hydro_unsupported_detections": 0,
        "hydro_best_iou_sum": 0.0,
        "hydro_best_iou_count": 0,
        "hydro_support_camera_count_hist": Counter(),
        "per_camera": defaultdict(
            lambda: {
                "hydro_projected_views": 0,
                "hydro_supported_views": 0,
                "rtmdet_ship_boxes": 0,
                "rtmdet_high_conf_ship_boxes": 0,
                "rtmdet_matched_to_hydro": 0,
                "rtmdet_unmatched_to_hydro": 0,
                "rtmdet_high_conf_unmatched_to_hydro": 0,
                "best_iou_sum": 0.0,
                "best_iou_count": 0,
            }
        ),
    }


def finalize_summary(summary: dict[str, Any]) -> None:
    hydro_total = summary["hydro_ship_detections"]
    projected_views = summary["hydro_projected_camera_views"]
    summary["hydro_supported_detection_rate"] = (
        summary["hydro_supported_detections_any_camera"] / hydro_total if hydro_total else 0.0
    )
    summary["hydro_supported_camera_view_rate"] = (
        summary["hydro_supported_camera_views"] / projected_views if projected_views else 0.0
    )
    summary["hydro_mean_best_iou"] = (
        summary["hydro_best_iou_sum"] / summary["hydro_best_iou_count"]
        if summary["hydro_best_iou_count"]
        else 0.0
    )
    summary["hydro_support_camera_count_hist"] = dict(
        sorted(summary["hydro_support_camera_count_hist"].items(), key=lambda item: int(item[0]))
    )
    per_camera = {}
    for channel in CALIBRATED_CAMERAS:
        stats = dict(summary["per_camera"][channel])
        stats["hydro_supported_view_rate"] = (
            stats["hydro_supported_views"] / stats["hydro_projected_views"]
            if stats["hydro_projected_views"]
            else 0.0
        )
        stats["rtmdet_match_rate_to_hydro"] = (
            stats["rtmdet_matched_to_hydro"] / stats["rtmdet_ship_boxes"]
            if stats["rtmdet_ship_boxes"]
            else 0.0
        )
        stats["mean_best_iou"] = (
            stats["best_iou_sum"] / stats["best_iou_count"] if stats["best_iou_count"] else 0.0
        )
        stats.pop("best_iou_sum", None)
        stats.pop("best_iou_count", None)
        per_camera[channel] = stats
    summary["per_camera"] = per_camera
    summary.pop("hydro_best_iou_sum", None)
    summary.pop("hydro_best_iou_count", None)


if __name__ == "__main__":
    main()
