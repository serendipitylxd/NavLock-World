#!/usr/bin/env python3
"""Build legacy-denominator world-state GT from current generated GT files.

The historical val+test fused summary evaluated current world-state on all
prediction scenes plus test recognition-only scenes, but not val recognition-only
scenes. This helper keeps current scene/berth annotations, then strips the
current world-state fields from val recognition-only rows so future re-runs use
the same denominator as the old official summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/lock_world_state"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lock_world_state"))
    parser.add_argument(
        "--tag",
        default="legacy_current_gt",
        help="Output tag: lock_world_state_<split>_<tag>.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for split in ("val", "test"):
        sequence_path = args.data_root / "navlock_sequences" / f"scene_sequences_{split}.json"
        input_path = args.input_dir / f"lock_world_state_{split}.jsonl"
        output_path = args.output_dir / f"lock_world_state_{split}_{args.tag}.jsonl"
        result = build_split(
            split=split,
            sequence_path=sequence_path,
            input_path=input_path,
            output_path=output_path,
        )
        summary[split] = result
        print(
            f"split={split} wrote={output_path} rows={result['rows']} "
            f"stripped_rows={result['stripped_current_rows']}"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_split(
    *,
    split: str,
    sequence_path: Path,
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    sequences = {
        item["scene_token"]: item
        for item in json.loads(sequence_path.read_text(encoding="utf-8")).get(
            "sequences", []
        )
    }
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    stats = {
        "rows": len(rows),
        "stripped_current_rows": 0,
        "stripped_current_slots": 0,
        "stripped_current_occupied": 0,
        "stripped_current_motion": 0,
        "kept_current_slots": 0,
        "kept_current_occupied": 0,
        "kept_current_motion": 0,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            scene_token = row.get("scene_token")
            sequence = sequences.get(scene_token) or {}
            is_recognition_only = not bool(sequence.get("has_prediction_target"))
            should_strip_current = split == "val" and is_recognition_only
            if should_strip_current:
                strip_current_world_state(row, stats)
            else:
                add_current_stats(row, stats)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def strip_current_world_state(row: dict[str, Any], stats: dict[str, Any]) -> None:
    lock = row.get("lock_occupancy") or {}
    flow = row.get("vessel_motion_flow") or {}
    current = lock.pop("current", {}) or {}
    input_window = flow.pop("input_window", []) or []
    slots = current.get("berth_slots") or []
    stats["stripped_current_rows"] += 1
    stats["stripped_current_slots"] += len(slots)
    stats["stripped_current_occupied"] += sum(1 for slot in slots if slot.get("occupied"))
    stats["stripped_current_motion"] += len(input_window)
    row["lock_occupancy"] = lock
    row["vessel_motion_flow"] = flow


def add_current_stats(row: dict[str, Any], stats: dict[str, Any]) -> None:
    current = ((row.get("lock_occupancy") or {}).get("current") or {})
    input_window = ((row.get("vessel_motion_flow") or {}).get("input_window") or [])
    slots = current.get("berth_slots") or []
    stats["kept_current_slots"] += len(slots)
    stats["kept_current_occupied"] += sum(1 for slot in slots if slot.get("occupied"))
    stats["kept_current_motion"] += len(input_window)


if __name__ == "__main__":
    main()
