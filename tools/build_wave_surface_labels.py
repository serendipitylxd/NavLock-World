#!/usr/bin/env python3
"""Build targeted weak wave labels from NavLock water_state annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WAVE_RULES = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to outputs/wave_labels/navlock_wave_labels_<split>.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output = (
        Path(args.output)
        if args.output
        else Path("outputs") / "wave_labels" / f"navlock_wave_labels_{args.split}.jsonl"
    )
    labels = build_wave_labels(data_root=data_root, split=args.split)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for item in labels:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"wrote={output}")
    print(f"split={args.split}")
    print(f"num_wave_labels={len(labels)}")
    counts: dict[str, int] = {}
    for item in labels:
        counts[item["water_state"]] = counts.get(item["water_state"], 0) + 1
    print(f"counts_by_water_state={json.dumps(counts, ensure_ascii=False, sort_keys=True)}")


def build_wave_labels(data_root: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(
        (data_root / "navlock_sequences" / f"scene_sequences_{split}.json").read_text(
            encoding="utf-8"
        )
    )
    labels: list[dict[str, Any]] = []
    for sequence in payload["sequences"]:
        for frame in sequence["frames"]:
            water_state = frame["lock_state"]["water_state"]
            rule = WAVE_RULES.get(water_state)
            if not rule:
                continue
            image = frame["images"][rule["camera"]]
            labels.append(
                {
                    "id": f"{split}:wave:{frame['sample_token']}:{rule['camera']}",
                    "split": split,
                    "scene_token": sequence["scene_token"],
                    "scene_name": sequence["scene_name"],
                    "sample_token": frame["sample_token"],
                    "frame_index": frame["frame_index"],
                    "timestamp": frame["timestamp"],
                    "relative_time_sec": frame["relative_time_sec"],
                    "water_state": water_state,
                    "water_level": frame["lock_state"].get("water_level"),
                    "camera": rule["camera"],
                    "image_path": str(data_root / image["file_name"]),
                    "wave_label": "wave_expected",
                    "wave_expected": True,
                    "region_id": rule["region_id"],
                    "region_description": rule["region_description"],
                    "label_source": "derived_from_water_state_target_region_rule",
                    "image_verified": False,
                    "image_level_waterline_annotation_required": False,
                    "numeric_water_level_available": frame["lock_state"].get("water_level")
                    is not None,
                }
            )
    return labels


if __name__ == "__main__":
    main()
