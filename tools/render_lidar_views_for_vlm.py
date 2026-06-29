#!/usr/bin/env python3
"""Render NavLock LiDAR point clouds into VLM-friendly BEV/range-view images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


DEFAULT_POINT_CLOUD_RANGE = (0.0, 0.0, -10.0, 100.0, 320.0, 15.0)
DEFAULT_VLM_SEMANTIC_FILES = (
    "outputs/vlm_semantic/navlock_vlm_semantic_train.jsonl",
    "outputs/vlm_semantic/navlock_vlm_semantic_val.jsonl",
    "outputs/vlm_semantic/navlock_vlm_semantic_test.jsonl",
    "outputs/vlm_semantic/navlock_vlm_semantic_recognition_train.jsonl",
    "outputs/vlm_semantic/navlock_vlm_semantic_recognition_val.jsonl",
    "outputs/vlm_semantic/navlock_vlm_semantic_recognition_test.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help=(
            "VLM semantic JSONL files to scan. Defaults to the six prediction and "
            "recognition files under outputs/vlm_semantic/."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/vlm_semantic/lidar_views",
        help="Root directory for rendered <split>/<sample_token>_{bev,range}.png.",
    )
    parser.add_argument("--num-point-features", type=int, default=5)
    parser.add_argument("--bev-width", type=int, default=256)
    parser.add_argument("--bev-height", type=int, default=512)
    parser.add_argument("--range-width", type=int, default=512)
    parser.add_argument("--range-height", type=int, default=128)
    parser.add_argument(
        "--point-cloud-range",
        default=",".join(str(value) for value in DEFAULT_POINT_CLOUD_RANGE),
        help="Comma-separated x_min,y_min,z_min,x_max,y_max,z_max.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in (args.input or DEFAULT_VLM_SEMANTIC_FILES)]
    output_root = Path(args.output_root)
    point_cloud_range = parse_point_cloud_range(args.point_cloud_range)
    records = collect_lidar_records(input_paths)
    if args.limit is not None:
        records = records[: args.limit]

    rendered = 0
    skipped = 0
    missing = 0
    manifest_rows = []
    for record in records:
        result = render_record(
            record=record,
            output_root=output_root,
            point_cloud_range=point_cloud_range,
            num_point_features=args.num_point_features,
            bev_size=(args.bev_width, args.bev_height),
            range_size=(args.range_width, args.range_height),
            overwrite=args.overwrite,
        )
        manifest_rows.append(result)
        if result["status"] == "rendered":
            rendered += 1
        elif result["status"] == "skipped_existing":
            skipped += 1
        elif result["status"] == "missing_lidar":
            missing += 1

    manifest = {
        "metadata": {
            "num_records": len(records),
            "rendered": rendered,
            "skipped_existing": skipped,
            "missing_lidar": missing,
            "point_cloud_range": list(point_cloud_range),
            "bev_size": [args.bev_width, args.bev_height],
            "range_size": [args.range_width, args.range_height],
        },
        "records": manifest_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"output_root={output_root}")
    print(f"num_records={len(records)}")
    print(f"rendered={rendered}")
    print(f"skipped_existing={skipped}")
    print(f"missing_lidar={missing}")


def collect_lidar_records(input_paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for input_path in input_paths:
        if not input_path.exists():
            continue
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            split = item.get("split", "unknown")
            for frame in item.get("input", {}).get("frames", []):
                lidar = frame.get("lidar") or {}
                sample_token = frame.get("sample_token")
                lidar_path = lidar.get("path")
                if not sample_token or not lidar_path:
                    continue
                key = (split, sample_token)
                records.setdefault(
                    key,
                    {
                        "split": split,
                        "sample_token": sample_token,
                        "scene_token": item.get("scene_token"),
                        "frame_index": frame.get("frame_index"),
                        "relative_time_sec": frame.get("relative_time_sec"),
                        "lidar_path": lidar_path,
                    },
                )
    return [records[key] for key in sorted(records)]


def render_record(
    record: dict[str, Any],
    output_root: Path,
    point_cloud_range: tuple[float, float, float, float, float, float],
    num_point_features: int,
    bev_size: tuple[int, int],
    range_size: tuple[int, int],
    overwrite: bool,
) -> dict[str, Any]:
    split_root = output_root / record["split"]
    bev_path = split_root / f"{record['sample_token']}_bev.png"
    range_path = split_root / f"{record['sample_token']}_range.png"
    result = {
        **record,
        "bev_path": str(bev_path),
        "range_view_path": str(range_path),
    }
    if bev_path.exists() and range_path.exists() and not overwrite:
        result["status"] = "skipped_existing"
        return result

    lidar_path = Path(record["lidar_path"])
    if not lidar_path.is_absolute():
        lidar_path = Path.cwd() / lidar_path
    if not lidar_path.exists():
        result["status"] = "missing_lidar"
        return result

    points = load_lidar_points(lidar_path, num_point_features=num_point_features)
    bev = render_bev(points, point_cloud_range=point_cloud_range, size=bev_size)
    range_view = render_range_view(
        points,
        point_cloud_range=point_cloud_range,
        size=range_size,
    )

    split_root.mkdir(parents=True, exist_ok=True)
    write_rgb_png(bev_path, bev)
    write_rgb_png(range_path, range_view)
    result["status"] = "rendered"
    result["num_points"] = int(points.shape[0])
    return result


def load_lidar_points(path: Path, num_point_features: int = 5) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % num_point_features != 0:
        raise ValueError(
            f"{path} has {raw.size} floats, not divisible by {num_point_features}"
        )
    return raw.reshape(-1, num_point_features)


def render_bev(
    points: np.ndarray,
    point_cloud_range: tuple[float, float, float, float, float, float],
    size: tuple[int, int] = (256, 512),
) -> np.ndarray:
    width, height = size
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    filtered = filter_points(points, point_cloud_range)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if filtered.size == 0:
        return annotate_image(image, "BEV LIDAR_TOP")

    x = filtered[:, 0]
    y = filtered[:, 1]
    z = filtered[:, 2]
    col = np.clip(((x - x_min) / (x_max - x_min) * (width - 1)).astype(np.int32), 0, width - 1)
    row = np.clip((height - 1 - (y - y_min) / (y_max - y_min) * (height - 1)).astype(np.int32), 0, height - 1)
    flat_index = row * width + col

    count = np.zeros(height * width, dtype=np.float32)
    max_z = np.full(height * width, z_min, dtype=np.float32)
    min_range = np.full(height * width, np.inf, dtype=np.float32)
    distance = np.hypot(x, y).astype(np.float32)
    np.add.at(count, flat_index, 1.0)
    np.maximum.at(max_z, flat_index, z.astype(np.float32))
    np.minimum.at(min_range, flat_index, distance)

    valid = count > 0
    density = np.zeros_like(count)
    density[valid] = np.log1p(count[valid]) / np.log1p(max(float(count.max()), 1.0))
    height_norm = np.zeros_like(count)
    height_norm[valid] = np.clip((max_z[valid] - z_min) / (z_max - z_min), 0.0, 1.0)
    range_norm = np.zeros_like(count)
    range_norm[valid] = 1.0 - np.clip(
        (min_range[valid] - 0.0) / max(np.hypot(x_max, y_max), 1.0),
        0.0,
        1.0,
    )

    rgb = np.stack(
        [
            (height_norm.reshape(height, width) * 255).astype(np.uint8),
            (density.reshape(height, width) * 255).astype(np.uint8),
            (range_norm.reshape(height, width) * 255).astype(np.uint8),
        ],
        axis=-1,
    )
    return annotate_image(rgb, "BEV LIDAR_TOP")


def render_range_view(
    points: np.ndarray,
    point_cloud_range: tuple[float, float, float, float, float, float],
    size: tuple[int, int] = (512, 128),
    yaw_limits_deg: tuple[float, float] = (-20.0, 110.0),
    pitch_limits_deg: tuple[float, float] = (-12.0, 12.0),
) -> np.ndarray:
    width, height = size
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    filtered = filter_points(points, point_cloud_range)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if filtered.size == 0:
        return annotate_image(image, "RANGE LIDAR_TOP")

    x = filtered[:, 0]
    y = filtered[:, 1]
    z = filtered[:, 2]
    distance_xy = np.hypot(x, y)
    distance = np.sqrt(distance_xy * distance_xy + z * z)
    yaw = np.degrees(np.arctan2(y, x))
    pitch = np.degrees(np.arctan2(z, np.maximum(distance_xy, 1e-6)))
    yaw_min, yaw_max = yaw_limits_deg
    pitch_min, pitch_max = pitch_limits_deg
    mask = (
        (yaw >= yaw_min)
        & (yaw <= yaw_max)
        & (pitch >= pitch_min)
        & (pitch <= pitch_max)
        & np.isfinite(distance)
    )
    if not np.any(mask):
        return annotate_image(image, "RANGE LIDAR_TOP")

    yaw = yaw[mask]
    pitch = pitch[mask]
    z = z[mask]
    distance = distance[mask]
    col = np.clip(((yaw - yaw_min) / (yaw_max - yaw_min) * (width - 1)).astype(np.int32), 0, width - 1)
    row = np.clip((height - 1 - (pitch - pitch_min) / (pitch_max - pitch_min) * (height - 1)).astype(np.int32), 0, height - 1)
    flat_index = row * width + col

    count = np.zeros(height * width, dtype=np.float32)
    max_z = np.full(height * width, z_min, dtype=np.float32)
    min_range = np.full(height * width, np.inf, dtype=np.float32)
    np.add.at(count, flat_index, 1.0)
    np.maximum.at(max_z, flat_index, z.astype(np.float32))
    np.minimum.at(min_range, flat_index, distance.astype(np.float32))

    valid = count > 0
    density = np.zeros_like(count)
    density[valid] = np.log1p(count[valid]) / np.log1p(max(float(count.max()), 1.0))
    height_norm = np.zeros_like(count)
    height_norm[valid] = np.clip((max_z[valid] - z_min) / (z_max - z_min), 0.0, 1.0)
    range_norm = np.zeros_like(count)
    range_norm[valid] = 1.0 - np.clip(
        min_range[valid] / max(np.hypot(x_max, y_max), 1.0),
        0.0,
        1.0,
    )
    rgb = np.stack(
        [
            (range_norm.reshape(height, width) * 255).astype(np.uint8),
            (height_norm.reshape(height, width) * 255).astype(np.uint8),
            (density.reshape(height, width) * 255).astype(np.uint8),
        ],
        axis=-1,
    )
    return annotate_image(rgb, "RANGE LIDAR_TOP")


def filter_points(
    points: np.ndarray,
    point_cloud_range: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, points.shape[-1] if points.ndim == 2 else 5)
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    xyz = points[:, :3]
    mask = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= x_min)
        & (xyz[:, 0] <= x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] <= y_max)
        & (xyz[:, 2] >= z_min)
        & (xyz[:, 2] <= z_max)
    )
    return points[mask]


def annotate_image(image: np.ndarray, title: str) -> np.ndarray:
    annotated = image.copy()
    cv2.putText(
        annotated,
        title,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        annotated,
        (0, 0),
        (annotated.shape[1] - 1, annotated.shape[0] - 1),
        (80, 80, 80),
        1,
    )
    return annotated


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise IOError(f"failed to write {path}")


def parse_point_cloud_range(raw: str) -> tuple[float, float, float, float, float, float]:
    values = tuple(float(part) for part in raw.split(","))
    if len(values) != 6:
        raise ValueError(
            "--point-cloud-range must contain six comma-separated values"
        )
    return values


if __name__ == "__main__":
    main()
