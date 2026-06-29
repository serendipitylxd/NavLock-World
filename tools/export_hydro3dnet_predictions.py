#!/usr/bin/env python3
"""Convert Hydro3DNet result.pkl files to the NavLock 3D prediction JSON schema."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


NAVLOCK_3D_CLASSES = (
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
    "Lock_footbridge",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--input", required=True, help="Hydro3DNet result.pkl path.")
    parser.add_argument(
        "--sample-info",
        default=None,
        help="Sample info pkl. Defaults to data/huaiyin_infos_<split>.pkl.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "outputs/hydro3dnet_navlock/<split>_predictions.json."
        ),
    )
    parser.add_argument(
        "--strict-frame-id",
        action="store_true",
        help="Fail if Hydro3DNet frame_id does not match the info-file order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    info_path = (
        Path(args.sample_info)
        if args.sample_info
        else Path("data") / f"huaiyin_infos_{args.split}.pkl"
    )
    output_path = (
        Path(args.output)
        if args.output
        else Path("outputs") / "hydro3dnet_navlock" / f"{args.split}_predictions.json"
    )

    hydro_predictions = _load_pickle(input_path)
    info_items = _load_info_items(info_path)
    if len(hydro_predictions) != len(info_items):
        raise ValueError(
            f"prediction/info length mismatch: {len(hydro_predictions)} vs {len(info_items)}"
        )

    class_to_index = {name: index for index, name in enumerate(NAVLOCK_3D_CLASSES)}
    converted = []
    frame_mismatches = 0
    for data_index, (prediction, info_item) in enumerate(zip(hydro_predictions, info_items)):
        expected_frame_id = _frame_id_from_info(info_item)
        frame_id = str(prediction.get("frame_id", ""))
        if expected_frame_id and frame_id and frame_id != expected_frame_id:
            frame_mismatches += 1
            if args.strict_frame_id:
                raise ValueError(
                    "frame_id mismatch at index "
                    f"{data_index}: prediction={frame_id!r}, info={expected_frame_id!r}"
                )

        names = _as_list(prediction.get("name", []))
        scores = [float(score) for score in _as_list(prediction.get("score", []))]
        boxes = _as_list(prediction.get("boxes_lidar", []))
        raw_labels = [int(label) for label in _as_list(prediction.get("pred_labels", []))]

        labels = []
        label_names = []
        kept_boxes = []
        kept_scores = []
        for item_index, name in enumerate(names):
            name = str(name)
            if name in class_to_index:
                label = class_to_index[name]
            elif item_index < len(raw_labels):
                # OpenPCDet-style labels are 1-based for known classes.
                label = raw_labels[item_index] - 1
                name = NAVLOCK_3D_CLASSES[label] if 0 <= label < len(NAVLOCK_3D_CLASSES) else name
            else:
                continue
            if not 0 <= label < len(NAVLOCK_3D_CLASSES):
                continue
            labels.append(label)
            label_names.append(name)
            kept_boxes.append(boxes[item_index] if item_index < len(boxes) else [])
            kept_scores.append(scores[item_index] if item_index < len(scores) else 0.0)

        converted.append(
            {
                "sample_idx": data_index,
                "sample_token": _sample_token_from_info(info_item),
                "frame_id": frame_id or expected_frame_id,
                "boxes": kept_boxes,
                "labels": labels,
                "label_names": label_names,
                "scores": kept_scores,
                "source_model": "Hydro3DNet",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={output_path}")
    print(f"split={args.split}")
    print(f"num_predictions={len(converted)}")
    print(f"frame_id_mismatches={frame_mismatches}")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_info_items(path: Path) -> list[dict[str, Any]]:
    payload = _load_pickle(path)
    return payload["data_list"] if isinstance(payload, dict) else payload


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _frame_id_from_info(info_item: dict[str, Any]) -> str:
    lidar_path = info_item.get("lidar_points", {}).get("lidar_path")
    if lidar_path:
        stem = Path(lidar_path).stem
        return stem.removeprefix("lidar_") if stem.startswith("lidar_") else stem
    sample_idx = info_item.get("sample_idx")
    return str(sample_idx) if sample_idx is not None else ""


def _sample_token_from_info(info_item: dict[str, Any]) -> str:
    for key in ("sample_token", "token", "sample_idx"):
        value = info_item.get(key)
        if value is not None:
            return str(value)
    return ""


if __name__ == "__main__":
    main()
