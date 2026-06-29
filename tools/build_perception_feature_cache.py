#!/usr/bin/env python3
"""Build per-frame perception feature cache from trained detector outputs."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any


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

NAVLOCK_3D_CLASSES = (
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
    "Lock_footbridge",
)

SHIP_2D_CLASSES = {
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Unladen_container_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
}
SHIP_3D_CLASSES = set(NAVLOCK_3D_CLASSES) - {"Lock_footbridge"}

DET3D_DISPLAY_NAMES = {
    "hydro3dnet": "Hydro3DNet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--rtmdet-predictions", default=None)
    parser.add_argument(
        "--det3d-predictions",
        default=None,
        help=(
            "3D detector prediction JSON. Defaults to "
            "outputs/hydro3dnet_navlock/<split>_predictions.json."
        ),
    )
    parser.add_argument(
        "--sample-info",
        default=None,
        help="Sample info pkl. Defaults to data/huaiyin_infos_<split>.pkl.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Minimum detector score included in aggregate counts.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to outputs/perception_features/perception_features_<split>.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    sequence_file = data_root / "navlock_sequences" / f"scene_sequences_{args.split}.json"
    rtmdet_predictions = (
        Path(args.rtmdet_predictions)
        if args.rtmdet_predictions
        else Path("outputs") / "mmdet2d" / "navlock_rtmdet_s" / f"{args.split}_predictions.pkl"
    )
    det3d_backend = "hydro3dnet"
    det3d_predictions = _default_det3d_predictions(args)
    output = (
        Path(args.output)
        if args.output
        else Path("outputs") / "perception_features" / f"perception_features_{args.split}.json"
    )

    sequences_payload = _load_json(sequence_file)
    rtmdet_by_path = _load_rtmdet(rtmdet_predictions, args.score_threshold)
    det3d_by_sample = _load_det3d_predictions(det3d_predictions, args.score_threshold)
    sample_idx_to_data_index = _load_sample_index(
        Path(args.sample_info)
        if args.sample_info
        else data_root / f"huaiyin_infos_{args.split}.pkl"
    )

    frames = []
    missing_2d = 0
    missing_3d = 0
    for sequence in sequences_payload["sequences"]:
        for frame in sequence["frames"]:
            image_features = {}
            for channel, image in frame["images"].items():
                image_path = str(data_root / image["file_name"])
                pred = rtmdet_by_path.get(image_path)
                if pred is None:
                    missing_2d += 1
                    pred = _empty_2d_feature()
                image_features[channel] = pred

            data_index = sample_idx_to_data_index.get(frame["sample_idx"])
            if data_index is None:
                data_index = sample_idx_to_data_index.get(frame["sample_token"])
            pred_3d = (
                det3d_by_sample.get(data_index) if data_index is not None else None
            )
            if pred_3d is None:
                missing_3d += 1
                pred_3d = _empty_3d_feature()

            frames.append(
                {
                    "scene_token": sequence["scene_token"],
                    "scene_name": sequence["scene_name"],
                    "sample_token": frame["sample_token"],
                    "sample_idx": frame["sample_idx"],
                    "frame_index": frame["frame_index"],
                    "timestamp": frame["timestamp"],
                    "relative_time_sec": frame["relative_time_sec"],
                    "image_features": image_features,
                    "lidar_3d_features": pred_3d,
                    "flat_features": _flat_features(image_features, pred_3d),
                }
            )

    payload = {
        "metadata": {
            "split": args.split,
            "score_threshold": args.score_threshold,
            "rtmdet_predictions": str(rtmdet_predictions),
            "detector_sources": _detector_sources(det3d_backend),
            "det3d_backend": det3d_backend,
            "det3d_predictions": str(det3d_predictions),
            "hydro3dnet_predictions": str(det3d_predictions),
            "sample_info": str(
                args.sample_info
                if args.sample_info
                else data_root / f"huaiyin_infos_{args.split}.pkl"
            ),
            "num_frames": len(frames),
            "missing_2d_camera_predictions": missing_2d,
            "missing_3d_frame_predictions": missing_3d,
            "flat_feature_names": _flat_feature_names(),
        },
        "frames": frames,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={output}")
    print(f"num_frames={len(frames)}")
    print(f"missing_2d_camera_predictions={missing_2d}")
    print(f"missing_3d_frame_predictions={missing_3d}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rtmdet(path: Path, score_threshold: float) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        predictions = pickle.load(f)

    by_path = {}
    for item in predictions:
        pred = item["pred_instances"]
        scores = pred["scores"].detach().cpu().tolist()
        labels = pred["labels"].detach().cpu().tolist()
        kept = [
            (int(label), float(score))
            for label, score in zip(labels, scores)
            if float(score) >= score_threshold
        ]
        by_path[item["img_path"]] = _summarize_labels(kept, RTMDET_CLASSES, SHIP_2D_CLASSES)
    return by_path


def _default_det3d_predictions(args: argparse.Namespace) -> Path:
    if args.det3d_predictions:
        return Path(args.det3d_predictions)
    return Path("outputs") / "hydro3dnet_navlock" / f"{args.split}_predictions.json"


def _detector_sources(det3d_backend: str) -> dict[str, str]:
    return {
        "2d": "RTMDet",
        "3d": DET3D_DISPLAY_NAMES[det3d_backend],
        "fusion": "Structured fusion of RTMDet image summaries and Hydro3DNet LiDAR geometry.",
    }


def _load_det3d_predictions(path: Path, score_threshold: float) -> dict[int, dict[str, Any]]:
    predictions = _load_json(path)
    by_sample = {}
    for item in predictions:
        kept = [
            (int(label), float(score))
            for label, score in zip(item["labels"], item["scores"])
            if float(score) >= score_threshold
        ]
        by_sample[int(item["sample_idx"])] = _summarize_labels(
            kept, NAVLOCK_3D_CLASSES, SHIP_3D_CLASSES
        )
    return by_sample


def _load_sample_index(path: Path) -> dict[str, int]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    data_list = payload["data_list"] if isinstance(payload, dict) else payload

    sample_to_index = {}
    for index, item in enumerate(data_list):
        for key in ("sample_idx", "sample_token", "token"):
            value = item.get(key)
            if value is not None:
                sample_to_index[str(value)] = index
        lidar_path = item.get("lidar_points", {}).get("lidar_path")
        if lidar_path:
            stem = Path(lidar_path).stem
            if stem.startswith("lidar_"):
                sample_to_index[stem.removeprefix("lidar_")] = index
    return sample_to_index


def _summarize_labels(
    label_scores: list[tuple[int, float]], classes: tuple[str, ...], ship_classes: set[str]
) -> dict[str, Any]:
    counts = {name: 0 for name in classes}
    score_sums = {name: 0.0 for name in classes}
    top_score = 0.0
    ship_count = 0
    ship_score_sum = 0.0
    for label, score in label_scores:
        if label < 0 or label >= len(classes):
            continue
        name = classes[label]
        counts[name] += 1
        score_sums[name] += score
        top_score = max(top_score, score)
        if name in ship_classes:
            ship_count += 1
            ship_score_sum += score

    total = len(label_scores)
    return {
        "num_detections": total,
        "top_score": top_score,
        "mean_score": sum(score for _, score in label_scores) / total if total else 0.0,
        "num_ship_detections": ship_count,
        "mean_ship_score": ship_score_sum / ship_count if ship_count else 0.0,
        "counts_by_class": counts,
        "score_sums_by_class": score_sums,
    }


def _empty_2d_feature() -> dict[str, Any]:
    return _summarize_labels([], RTMDET_CLASSES, SHIP_2D_CLASSES)


def _empty_3d_feature() -> dict[str, Any]:
    return _summarize_labels([], NAVLOCK_3D_CLASSES, SHIP_3D_CLASSES)


def _flat_feature_names() -> list[str]:
    return [
        "camera_num_detections",
        "camera_num_ship_detections",
        "camera_top_score",
        "camera_mean_ship_score",
        "lidar_num_detections",
        "lidar_num_ship_detections",
        "lidar_top_score",
        "lidar_mean_ship_score",
    ]


def _flat_features(
    image_features: dict[str, dict[str, Any]], lidar_3d_features: dict[str, Any]
) -> dict[str, float]:
    camera_total = sum(item["num_detections"] for item in image_features.values())
    camera_ship_total = sum(item["num_ship_detections"] for item in image_features.values())
    camera_top = max((item["top_score"] for item in image_features.values()), default=0.0)
    ship_scores = [
        item["mean_ship_score"]
        for item in image_features.values()
        if item["num_ship_detections"] > 0
    ]
    return {
        "camera_num_detections": float(camera_total),
        "camera_num_ship_detections": float(camera_ship_total),
        "camera_top_score": float(camera_top),
        "camera_mean_ship_score": float(sum(ship_scores) / len(ship_scores))
        if ship_scores
        else 0.0,
        "lidar_num_detections": float(lidar_3d_features["num_detections"]),
        "lidar_num_ship_detections": float(lidar_3d_features["num_ship_detections"]),
        "lidar_top_score": float(lidar_3d_features["top_score"]),
        "lidar_mean_ship_score": float(lidar_3d_features["mean_ship_score"]),
    }


if __name__ == "__main__":
    main()
