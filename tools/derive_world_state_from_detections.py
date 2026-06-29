#!/usr/bin/env python3
"""Derive lock world state directly from perception detections -- no VLM, no training.

``lock_occupancy`` and ``vessel_motion_flow`` are deterministic geometric functions
of the detected 3D instances (the same ``instances_3d`` that are the model's input),
so the observed parts need no language model at all:

* ``lock_occupancy.current`` and ``vessel_motion_flow.input_window`` are read out
  geometrically from the INPUT frames -- exact vs the GT labels by construction.
* ``lock_occupancy.future_10s`` and ``vessel_motion_flow.target_window`` are not
  observable at inference. Berth occupancy is predicted by persistence because it
  barely changes over a ~10 s horizon; motion flow uses a non-leaky settle-aware
  transition rule by default so movers that have reached/approached a berth are
  not blindly kept as entering/leaving. Scenes without a future prediction target
  keep only the current/input-window fields.

Everything here uses only the input frames + the static berth prior, so the output
is fully non-leaky. Output rows carry ``scene_token`` + ``lock_occupancy`` +
``vessel_motion_flow`` and feed straight into
``tools/evaluate_lock_world_state_from_predictions.py``.

Run from the repository root:

    python tools/derive_world_state_from_detections.py --data-root data --split test \
      --output outputs/lock_world_state/derived_test_from_detections.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Allow `python tools/derive_world_state_from_detections.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navlock_world.berth_ship_intentions import (  # noqa: E402
    _end_speed,
    _inside_box,
    _track_ships,
)
from navlock_world.lock_world_state import (  # noqa: E402
    _chamber_bounds,
    _compute_flow,
    _compute_occupancy,
    load_scene_berths,
)

FUTURE_MOTION_MODES = ("settle_aware", "persistence")
FUTURE_HORIZON_SEC = 10.0
FUTURE_NEAR_BERTH_M = 30.0
FUTURE_PROJECTED_NEAR_BERTH_M = 40.0
FUTURE_SETTLED_END_SPEED_MPS = 0.5
MOVING_STATES = frozenset({"ship_entering_lock", "ship_leaving_lock", "ship_moving"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--scene-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--future-motion-mode",
        default="settle_aware",
        choices=FUTURE_MOTION_MODES,
        help=(
            "How to predict vessel_motion_flow.target_window from input frames. "
            "settle_aware uses only input tracks + berth geometry; persistence "
            "keeps the old target=input baseline."
        ),
    )
    return parser.parse_args()


def derive_prediction_from_input(
    sequence: dict[str, Any],
    berths: list[dict[str, Any]],
    *,
    future_motion_mode: str = "settle_aware",
) -> dict[str, Any]:
    """Geometric world-state prediction using ONLY the input frames (non-leaky).

    current/input are exact geometric read-outs. Future occupancy is a persistence
    baseline; future motion defaults to a settle-aware transition rule and can be
    forced back to persistence for ablations.
    """
    frames = sequence.get("frames", [])
    input_idx = sequence.get("prediction_input_frame_indices") or []
    input_frames = [frames[i] for i in input_idx]
    chamber = _chamber_bounds(berths)
    current_frame = input_frames[-1] if input_frames else {}

    current_occupancy = _compute_occupancy(current_frame, berths, chamber)
    input_flow = _compute_flow(input_frames, berths, chamber)
    output = {
        "scene_token": sequence.get("scene_token"),
        "sample_token": current_frame.get("sample_token"),
        "lock_occupancy": {
            "current": current_occupancy,
        },
        "vessel_motion_flow": {
            "input_window": input_flow,
        },
    }
    if not sequence.get("has_prediction_target"):
        return output

    if future_motion_mode == "persistence":
        target_flow = input_flow
    elif future_motion_mode == "settle_aware":
        target_flow = _settle_aware_target_flow(input_frames, input_flow, berths)
    else:
        raise ValueError(f"Unsupported future_motion_mode: {future_motion_mode}")
    output["lock_occupancy"]["future_10s"] = current_occupancy  # persistence
    output["vessel_motion_flow"]["target_window"] = target_flow
    return output


def _settle_aware_target_flow(
    input_frames: list[dict[str, Any]],
    input_flow: list[dict[str, Any]],
    berths: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Predict target-window flow from input tracks without reading target frames."""
    if not input_frames or not berths:
        return input_flow

    tracks = _track_ships(input_frames)
    target_flow = []
    for item in input_flow:
        pred = dict(item)
        motion_state = pred.get("motion_state")
        token = pred.get("instance_token")
        track = tracks.get(token)
        if motion_state in MOVING_STATES and track is not None:
            should_settle, inside_berth = _should_settle_in_future(track.get("txy") or [], berths)
            if should_settle:
                pred["motion_state"] = "ship_berthed" if inside_berth else "ship_static"
                pred["direction_label"] = "static_or_settled"
                pred["delta_xy"] = [0.0, 0.0]
                pred["end_speed_mps"] = 0.0
                end_region = pred.get("end_region")
                if end_region is not None:
                    pred["start_region"] = end_region
        target_flow.append(pred)
    return target_flow


def _should_settle_in_future(
    txy: list[tuple[float, float, float]],
    berths: list[dict[str, Any]],
) -> tuple[bool, bool]:
    if not txy:
        return False, False
    xn, yn = txy[-1][1], txy[-1][2]
    inside_berth = any(_inside_box(xn, yn, box) for box in berths)
    berth_distance = min((_point_box_distance(xn, yn, box) for box in berths), default=float("inf"))
    px, py = _project_last_velocity(txy, FUTURE_HORIZON_SEC)
    projected_inside = any(_inside_box(px, py, box) for box in berths)
    projected_distance = min(
        (_point_box_distance(px, py, box) for box in berths),
        default=float("inf"),
    )
    slow_at_window_end = _end_speed(txy) <= FUTURE_SETTLED_END_SPEED_MPS
    should_settle = (
        inside_berth
        or berth_distance <= FUTURE_NEAR_BERTH_M
        or projected_inside
        or projected_distance <= FUTURE_PROJECTED_NEAR_BERTH_M
        or slow_at_window_end
    )
    return should_settle, inside_berth


def _project_last_velocity(
    txy: list[tuple[float, float, float]],
    horizon_sec: float,
) -> tuple[float, float]:
    xn, yn = txy[-1][1], txy[-1][2]
    if len(txy) < 2:
        return xn, yn
    t0, x0, y0 = txy[-2]
    tn, _, _ = txy[-1]
    dt = tn - t0
    if dt <= 0:
        return xn, yn
    return xn + ((xn - x0) / dt) * horizon_sec, yn + ((yn - y0) / dt) * horizon_sec


def _point_box_distance(x: float, y: float, box: dict[str, Any]) -> float:
    dx = max(box["x_min"] - x, 0.0, x - box["x_max"])
    dy = max(box["y_min"] - y, 0.0, y - box["y_max"])
    return math.hypot(dx, dy)


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
            Path("outputs") / "lock_world_state" / f"derived_{split}_from_detections.jsonl"
        )
        if args.split == "all":
            sequence_file = args.data_root / "navlock_sequences" / f"scene_sequences_{split}.json"
            output = Path("outputs") / "lock_world_state" / f"derived_{split}_from_detections.jsonl"

        num = _build_split(sequence_file, berths, output, args.future_motion_mode)
        print(f"split={split} sequence_file={sequence_file} wrote={output} num={num}")


def _build_split(
    sequence_file: Path,
    berths: dict[str, list[dict[str, Any]]],
    output: Path,
    future_motion_mode: str,
) -> int:
    payload = json.loads(sequence_file.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    num = 0
    with output.open("w", encoding="utf-8") as handle:
        for sequence in payload.get("sequences", []):
            if not sequence.get("prediction_input_frame_indices"):
                continue
            pred = derive_prediction_from_input(
                sequence,
                berths.get(sequence.get("scene_token"), []),
                future_motion_mode=future_motion_mode,
            )
            handle.write(json.dumps(pred, ensure_ascii=False) + "\n")
            num += 1
    return num


if __name__ == "__main__":
    main()
