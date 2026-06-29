#!/usr/bin/env python3
"""Build berth-aware lock-occupancy + vessel-motion-flow world-state labels.

Reads the structured ``scene_sequences_<split>.json`` and ``scene.json`` and
derives lock occupancy and vessel motion flow (see
:mod:`navlock_world.lock_world_state`) into a JSONL. Scenes without a future
prediction target still receive current/input-window labels; future fields are
omitted for those rows so future metrics keep their prediction-scene denominator.
This is a ship-lock-specific world state, not a generic 3D voxel occupancy.

Run from the repository root:

    python tools/build_lock_world_state_labels.py --data-root data --split train \
      --output outputs/lock_world_state/lock_world_state_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow `python tools/build_lock_world_state_labels.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navlock_world.lock_world_state import derive_sequence_world_state, load_scene_berths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "all"))
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--scene-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_json = args.scene_json or (args.data_root / "v1.0-trainval" / "scene.json")
    berths = load_scene_berths(scene_json)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        sequence_file = args.sequence_file or (
            args.data_root / "navlock_sequences" / f"scene_sequences_{split}.json"
        )
        output = args.output or (
            Path("outputs") / "lock_world_state" / f"lock_world_state_{split}.jsonl"
        )
        if args.split == "all":
            # In all-split mode ignore single-file overrides to avoid collisions.
            sequence_file = args.data_root / "navlock_sequences" / f"scene_sequences_{split}.json"
            output = Path("outputs") / "lock_world_state" / f"lock_world_state_{split}.jsonl"

        num = _build_split(sequence_file, berths, output)
        print(f"split={split} sequence_file={sequence_file} wrote={output} num={num}")


def _build_split(
    sequence_file: Path,
    berths: dict[str, list[dict[str, Any]]],
    output: Path,
) -> int:
    payload = json.loads(sequence_file.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    num = 0
    with output.open("w", encoding="utf-8") as handle:
        for sequence in payload.get("sequences", []):
            if not sequence.get("prediction_input_frame_indices"):
                continue
            state = derive_sequence_world_state(
                sequence, berths.get(sequence.get("scene_token"), [])
            )
            handle.write(json.dumps(state, ensure_ascii=False) + "\n")
            num += 1
    return num


if __name__ == "__main__":
    main()
