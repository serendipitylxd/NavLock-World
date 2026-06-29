"""Berth-aware lock occupancy and vessel motion-flow world state.

This is a ship-lock-specific world-model state, NOT a generic 3D voxel occupancy
grid. It describes, per prediction scene:

* ``lock_occupancy`` -- which ideal berth slots are occupied and a coarse spatial
  bucketing of every ship (upper/lower gate zone, outside the lock width, or
  between berths), for the current (last input frame) and future (last target
  frame) states.
* ``vessel_motion_flow`` -- a per-ship motion/trend label (berthed, entering,
  leaving, moving, static) plus direction toward the upper/lower gate, over the
  input window and the target window.

It reuses the geometry from :mod:`navlock_world.berth_ship_intentions`: the
``scene.json`` ``ideal_berth_positions`` / ``ideal_berth_aabb_xy`` boxes laid out
along the chamber Y axis (upper gate = high Y, lower gate = low Y), and the
``Lock_footbridge`` non-ship category is excluded. The ``current`` occupancy and
``input_window`` flow use only the input frames, so they are non-leaky inputs;
``future_10s`` / ``target_window`` use the target frames and are labels only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

from navlock_world.berth_ship_intentions import (
    OPEN_GATE_STATES,  # noqa: F401  (re-exported for callers)
    _end_speed,
    _inside_box,
    _open_gate_direction,
    _track_ships,
    is_ship_category,
    load_scene_berths,  # noqa: F401  (re-exported: the module provides it)
)

# A ship counts as static when its whole-window net displacement stays under this
# many metres; combined with a low end-of-window speed it reads as settled.
STATIC_DISPLACEMENT_M = 3.0
SETTLED_END_SPEED_MPS = 0.2
END_SPEED_FRAMES = 5

COARSE_REGION_IDS = (
    "upper_gate_zone",
    "lower_gate_zone",
    "outside_lock_width",
    "between_berths",
)


def derive_sequence_world_state(
    sequence: dict[str, Any],
    berths: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the lock-occupancy + vessel-motion-flow world state for one scene.

    ``sequence`` is a ``scene_sequences_<split>.json`` entry (with ``frames``,
    ``prediction_input_frame_indices`` and ``prediction_target_frame_indices``);
    ``berths`` is that scene's ideal berth boxes. ``current`` occupancy / the
    input flow window use only input frames (non-leaky); ``future_10s`` / the
    target flow window use the target frames.
    """
    frames = sequence.get("frames", [])
    input_idx = sequence.get("prediction_input_frame_indices") or []
    target_idx = sequence.get("prediction_target_frame_indices") or []
    input_frames = [frames[i] for i in input_idx]
    target_frames = [frames[i] for i in target_idx]

    chamber = _chamber_bounds(berths)
    current_frame = input_frames[-1] if input_frames else {}
    state = {
        "scene_token": sequence.get("scene_token"),
        "sample_token": current_frame.get("sample_token"),
        "lock_occupancy": {
            "current": _compute_occupancy(current_frame, berths, chamber),
        },
        "vessel_motion_flow": {
            "input_window": _compute_flow(input_frames, berths, chamber),
        },
    }
    if target_frames:
        future_frame = target_frames[-1]
        state["lock_occupancy"]["future_10s"] = _compute_occupancy(
            future_frame, berths, chamber
        )
        state["vessel_motion_flow"]["target_window"] = _compute_flow(
            target_frames, berths, chamber
        )
    return state


def _chamber_bounds(berths: list[dict[str, Any]]) -> Optional[dict[str, float]]:
    if not berths:
        return None
    return {
        "x_min": min(b["x_min"] for b in berths),
        "x_max": max(b["x_max"] for b in berths),
        "y_min": min(b["y_min"] for b in berths),
        "y_max": max(b["y_max"] for b in berths),
        "y_mean": sum(b["cy"] for b in berths) / len(berths),
    }


def load_lock_chamber_bounds(map_json_path: str | Path) -> Optional[dict[str, float]]:
    """Load the physical lock-chamber XY bounds from a NavLock map JSON."""
    payload = json.loads(Path(map_json_path).read_text(encoding="utf-8"))
    regions = payload.get("regions") if isinstance(payload, dict) else None
    if not isinstance(regions, list):
        return None
    for region in regions:
        if not isinstance(region, dict) or region.get("name") != "lock_chamber":
            continue
        x_range = region.get("x_range")
        y_range = region.get("y_range")
        if not (
            isinstance(x_range, list)
            and isinstance(y_range, list)
            and len(x_range) == 2
            and len(y_range) == 2
        ):
            return None
        x_min, x_max = sorted(float(value) for value in x_range)
        y_min, y_max = sorted(float(value) for value in y_range)
        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "y_mean": (y_min + y_max) / 2.0,
        }
    return None


def _frame_ships(frame: dict[str, Any]) -> list[dict[str, Any]]:
    ships = []
    for inst in frame.get("instances_3d") or []:
        if not is_ship_category(inst.get("category")):
            continue
        token = inst.get("instance_token")
        translation = inst.get("translation")
        if token is None or not translation:
            continue
        ships.append(
            {
                "instance_token": token,
                "category": inst.get("category"),
                "x": float(translation[0]),
                "y": float(translation[1]),
            }
        )
    return ships


def _coarse_region(x: float, y: float, chamber: Optional[dict[str, float]]) -> str:
    """Coarse spatial bucket of a ship position (partitions every ship)."""
    if chamber is None:
        return "outside_lock_width"
    if x < chamber["x_min"] or x > chamber["x_max"]:
        return "outside_lock_width"
    if y > chamber["y_max"]:
        return "upper_gate_zone"
    if y < chamber["y_min"]:
        return "lower_gate_zone"
    return "between_berths"


def _compute_occupancy(
    frame: dict[str, Any],
    berths: list[dict[str, Any]],
    chamber: Optional[dict[str, float]],
) -> dict[str, Any]:
    ships = _frame_ships(frame)

    berth_slots = []
    for index, box in enumerate(berths, start=1):
        inside = [s for s in ships if _inside_box(s["x"], s["y"], box)]
        berth_slots.append(
            {
                "region_id": f"berth_slot_{index:02d}",
                "occupied": bool(inside),
                "ship_count": len(inside),
                "ship_tokens": [s["instance_token"] for s in inside],
            }
        )

    coarse: dict[str, list[str]] = {rid: [] for rid in COARSE_REGION_IDS}
    for ship in ships:
        coarse[_coarse_region(ship["x"], ship["y"], chamber)].append(ship["instance_token"])
    coarse_regions = [
        {"region_id": rid, "ship_count": len(tokens), "ship_tokens": tokens}
        for rid, tokens in coarse.items()
    ]

    return {
        "berth_slots": berth_slots,
        "coarse_regions": coarse_regions,
        "num_occupied_berths": sum(1 for slot in berth_slots if slot["occupied"]),
        "num_ships": len(ships),
    }


def _compute_flow(
    frames: list[dict[str, Any]],
    berths: list[dict[str, Any]],
    chamber: Optional[dict[str, float]],
) -> list[dict[str, Any]]:
    if not frames:
        return []
    open_gate_direction = _open_gate_direction(frames) if berths else None
    tracks = _track_ships(frames)

    flow = []
    for token, track in tracks.items():
        txy = track["txy"]
        if not txy:
            continue
        x0, y0 = txy[0][1], txy[0][2]
        xn, yn = txy[-1][1], txy[-1][2]
        dx, dy = xn - x0, yn - y0
        net_disp = math.hypot(dx, dy)
        end_speed = _end_speed(txy)
        inside_berth = any(_inside_box(xn, yn, box) for box in berths)

        motion_state, direction_label = _motion_state(
            dy=dy,
            net_disp=net_disp,
            end_speed=end_speed,
            inside_berth=inside_berth,
            open_gate_direction=open_gate_direction,
            chamber=chamber,
        )
        flow.append(
            {
                "instance_token": token,
                "category": track["category"],
                "motion_state": motion_state,
                "direction_label": direction_label,
                "delta_xy": [round(dx, 4), round(dy, 4)],
                "end_speed_mps": round(end_speed, 4),
                "start_region": _coarse_region(x0, y0, chamber),
                "end_region": _coarse_region(xn, yn, chamber),
            }
        )
    return flow


def _motion_state(
    dy: float,
    net_disp: float,
    end_speed: float,
    inside_berth: bool,
    open_gate_direction: Optional[float],
    chamber: Optional[dict[str, float]],
) -> tuple[str, str]:
    if inside_berth and (net_disp < STATIC_DISPLACEMENT_M or end_speed < SETTLED_END_SPEED_MPS):
        return "ship_berthed", "static_or_settled"

    if dy > 0:
        direction_label = "moving_to_upper_gate"
    elif dy < 0:
        direction_label = "moving_to_lower_gate"
    else:
        direction_label = "static_or_settled"

    if net_disp >= STATIC_DISPLACEMENT_M and open_gate_direction is not None and chamber is not None:
        toward_open = dy * open_gate_direction
        # Moving toward the open gate -> leaving; away from it (deeper in) -> entering.
        motion_state = "ship_leaving_lock" if toward_open > 0 else "ship_entering_lock"
    elif net_disp >= STATIC_DISPLACEMENT_M:
        motion_state = "ship_moving"
    else:
        motion_state = "ship_static"
    return motion_state, direction_label
