"""Derive ship intentions from ideal berth geometry.

The lock-chamber ideal berths are a known a-priori prior (`scene.json`
`ideal_berth_positions`): one berth slot per ship in the lockage, laid out along
the chamber Y axis. Combined with the input-window ship tracks and the gate
state, the berths give a non-leaky geometric estimate of each ship's intention:

* ``ship_berthed``  -- ship center sits inside an ideal berth box and has barely
  moved across the input window (near-deterministic; berthed ships are 143/143
  inside a box in the labeled data).
* ``ship_entering_lock`` / ``ship_leaving_lock`` -- the ship is moving. A lockage
  never mixes entering and leaving ships at once (validated: 10/10250 frames),
  so every moving ship in a scene shares one phase. The phase is decided by a
  consensus vote of all movers' net Y displacement relative to the OPEN-gate end
  (upper gate = high Y, lower gate = low Y): moving toward the open gate = leaving,
  away from it = entering.

Only real ship categories get an intention; lock infrastructure such as
``Lock_footbridge`` is excluded (those carry no ship-intention label).

This module operates on input-window frames only (current observations) and the
static berth prior, so it is safe to use as a non-leaky prior or prompt context
for both current-state recognition and future-state prediction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

# Lock infrastructure detected as 3D instances but which never carry a ship
# intention. Everything else (cargo ships/fleets/containers) is a real ship.
NON_SHIP_CATEGORIES = frozenset({"Lock_footbridge"})

# A ship counts as berthed when its whole-input-window net displacement stays
# under this many metres while it sits inside an ideal berth box. Whole-window
# displacement predicts the future berthed state better than instantaneous speed
# because it ignores per-frame detection jitter.
BERTHED_NET_DISPLACEMENT_M = 3.0

# A ship that entered DURING the window and has just parked inside a berth has a
# large net displacement but will stay berthed in the future. Catch it when it
# has settled (low end-of-window speed) right at a berth centre.
ARRIVED_END_SPEED_MPS = 0.2
ARRIVED_BERTH_CENTER_M = 5.0
END_SPEED_FRAMES = 5

OPEN_GATE_STATES = frozenset({"open", "opening"})


def is_ship_category(category: Optional[str]) -> bool:
    return category not in NON_SHIP_CATEGORIES


def load_scene_berths(scene_json_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Map ``scene_token -> [ideal_berth_aabb_xy, ...]`` from ``scene.json``."""
    scenes = json.loads(Path(scene_json_path).read_text(encoding="utf-8"))
    berths: dict[str, list[dict[str, Any]]] = {}
    for scene in scenes:
        boxes = [
            berth["ideal_berth_aabb_xy"]
            for berth in scene.get("ideal_berth_positions") or []
            if isinstance(berth, dict) and isinstance(berth.get("ideal_berth_aabb_xy"), dict)
        ]
        berths[scene["token"]] = boxes
    return berths


def _inside_box(x: float, y: float, box: dict[str, Any]) -> bool:
    return box["x_min"] <= x <= box["x_max"] and box["y_min"] <= y <= box["y_max"]


def _open_gate_direction(frames: list[dict[str, Any]]) -> Optional[float]:
    """Y direction toward the open-gate chamber end.

    Scans the latest input frames so a brief gate transition does not hide the
    open side. Returns ``None`` when neither gate is (un-ambiguously) open.
    """
    for frame in reversed(frames):
        lock_state = frame.get("lock_state") or {}
        upper = lock_state.get("upper_gate_state")
        lower = lock_state.get("lower_gate_state")
        upper_open = upper in OPEN_GATE_STATES
        lower_open = lower in OPEN_GATE_STATES
        if upper_open and not lower_open:
            return 1.0
        if lower_open and not upper_open:
            return -1.0
    return None


def _track_ships(frames: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collect per-ship ``(t, x, y)`` tracks and category across the frames."""
    tracks: dict[str, dict[str, Any]] = {}
    for index, frame in enumerate(frames):
        time = frame.get("relative_time_sec", index)
        for inst in frame.get("instances_3d") or []:
            category = inst.get("category")
            if not is_ship_category(category):
                continue
            token = inst.get("instance_token")
            translation = inst.get("translation")
            if token is None or not translation:
                continue
            track = tracks.setdefault(token, {"category": category, "txy": []})
            track["txy"].append(
                (float(time), float(translation[0]), float(translation[1]))
            )
    return tracks


def _end_speed(txy: list[tuple[float, float, float]]) -> float:
    """Average speed over the last ``END_SPEED_FRAMES`` samples (m/s)."""
    sub = txy[-END_SPEED_FRAMES:]
    if len(sub) < 2:
        return 0.0
    dt = sub[-1][0] - sub[0][0]
    if dt <= 0:
        return 0.0
    dist = math.hypot(sub[-1][1] - sub[0][1], sub[-1][2] - sub[0][2])
    return dist / dt


def derive_ship_intentions(
    frames: list[dict[str, Any]],
    berths: list[dict[str, Any]],
    *,
    berthed_net_displacement_m: float = BERTHED_NET_DISPLACEMENT_M,
) -> list[dict[str, Any]]:
    """Return ``[{instance_token, category, ship_intentions:[label]}, ...]``.

    ``frames`` are the input-window frames (each with ``instances_3d`` and
    ``lock_state``); ``berths`` are the scene's ideal berth boxes. Ships with no
    usable track or no berths produce no item.
    """
    if not berths or not frames:
        return []
    tracks = _track_ships(frames)

    berthed: dict[str, str] = {}
    movers: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    for token, track in tracks.items():
        txy = track["txy"]
        if not txy:
            continue
        x0, y0 = txy[0][1], txy[0][2]
        xn, yn = txy[-1][1], txy[-1][2]
        net_disp = math.hypot(xn - x0, yn - y0)
        inside = any(_inside_box(xn, yn, box) for box in berths)
        # Settled in place across the whole window, OR entered during the window
        # and has just parked (low end speed) right at a berth centre -> berthed.
        arrived = (
            _end_speed(txy) < ARRIVED_END_SPEED_MPS
            and min(math.hypot(xn - box["cx"], yn - box["cy"]) for box in berths)
            < ARRIVED_BERTH_CENTER_M
        )
        if inside and (net_disp < berthed_net_displacement_m or arrived):
            berthed[token] = "ship_berthed"
        else:
            movers.append((token, (x0, y0), (xn, yn)))

    mover_label: Optional[str] = None
    if movers:
        toward_open = _open_gate_direction(frames)
        if toward_open is not None:
            consensus = sum(
                (yn - y0) * toward_open for _, (_, y0), (_, yn) in movers
            )
            mover_label = (
                "ship_leaving_lock" if consensus > 0 else "ship_entering_lock"
            )
        else:
            # No clearly open gate (e.g. mid water-change): a moving ship is
            # heading to its berth, so default to entering.
            mover_label = "ship_entering_lock"

    items = []
    for token, track in tracks.items():
        if token in berthed:
            label = berthed[token]
        elif mover_label is not None and any(m[0] == token for m in movers):
            label = mover_label
        else:
            continue
        items.append(
            {
                "instance_token": token,
                "category": track["category"],
                "ship_intentions": [label],
            }
        )
    return items
