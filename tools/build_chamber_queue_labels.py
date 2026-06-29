#!/usr/bin/env python3
"""Build chamber-capacity and weak dispatch-queue labels for NavLock frames."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from navlock_world.berth_ship_intentions import _inside_box, is_ship_category
from navlock_world.lock_world_state import load_lock_chamber_bounds


DEFAULT_APPROACH_MARGIN_M = 10.0
DEFAULT_MAX_PARALLEL_ACTIONS = 2

LABEL_FIELDS = (
    "chamber_capacity_available",
    "available_berth_slots",
    "occupied_berth_slots",
    "num_occupied_berths",
    "num_ships_in_chamber",
    "all_in_chamber_ships_berthed_or_static",
    "no_ship_entering_or_leaving_inside_chamber",
    "queue_rank",
    "next_ship_to_enter_weak",
    "next_ship_to_leave_weak",
    "max_parallel_entries",
    "max_parallel_departures",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--lock-boundary-map",
        type=Path,
        default=None,
        help="Defaults to <data-root>/maps/huaiyin_lock_boundary.json.",
    )
    parser.add_argument(
        "--approach-margin-m",
        type=float,
        default=DEFAULT_APPROACH_MARGIN_M,
        help="Allowed x margin around chamber width for outside queue candidates.",
    )
    parser.add_argument(
        "--max-parallel-actions",
        type=int,
        default=DEFAULT_MAX_PARALLEL_ACTIONS,
        help="Scene-specific cap for simultaneous entry/departure recommendations.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/chamber_queue_labels/summary.json"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("outputs/chamber_queue_labels/chamber_queue.jsonl"),
    )
    parser.add_argument(
        "--no-update-sample",
        action="store_true",
        help="Only write summary/jsonl, do not update sample.json.",
    )
    parser.add_argument(
        "--no-update-pkl",
        action="store_true",
        help="Do not synchronize huaiyin_infos_*.pkl files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version_root = args.data_root / "v1.0-trainval"
    sample_path = version_root / "sample.json"
    lock_boundary_map = (
        args.lock_boundary_map or args.data_root / "maps" / "huaiyin_lock_boundary.json"
    )
    chamber = load_lock_chamber_bounds(lock_boundary_map)
    if chamber is None:
        raise SystemExit(f"failed to load lock chamber bounds from {lock_boundary_map}")

    rows = json.loads(sample_path.read_text(encoding="utf-8"))
    ships_by_sample = load_ship_centers_by_sample(version_root)
    berths_by_scene = load_scene_berth_slots(version_root / "scene.json")
    direction_by_scene = load_scene_directions(version_root)
    labels, diagnostics = build_chamber_queue_labels(
        rows,
        ships_by_sample=ships_by_sample,
        berths_by_scene=berths_by_scene,
        direction_by_scene=direction_by_scene,
        chamber=chamber,
        approach_margin_m=args.approach_margin_m,
        max_parallel_actions=args.max_parallel_actions,
    )

    if not args.no_update_sample:
        update_sample_rows(rows, labels)
        sample_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    pkl_report = []
    if not args.no_update_pkl:
        pkl_report = update_info_pkls(args.data_root, labels)

    write_jsonl(args.jsonl_output, rows, labels, diagnostics)
    summary = build_summary(
        rows,
        labels=labels,
        diagnostics=diagnostics,
        sample_path=sample_path,
        lock_boundary_map=lock_boundary_map,
        chamber=chamber,
        approach_margin_m=args.approach_margin_m,
        max_parallel_actions=args.max_parallel_actions,
        pkl_report=pkl_report,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote={args.summary_output}")
    print(f"wrote={args.jsonl_output}")
    print(f"num_samples={len(rows)}")
    print(f"capacity_counts={summary['capacity_counts']}")
    print(f"queue_candidate_frames={summary['queue_candidate_frames']}")
    print(f"updated_sample={not args.no_update_sample}")
    print(f"updated_pkl={not args.no_update_pkl}")


def load_ship_centers_by_sample(version_root: Path) -> dict[str, list[dict[str, Any]]]:
    attributes = {
        item["token"]: item["name"]
        for item in json.loads((version_root / "attribute.json").read_text(encoding="utf-8"))
    }
    categories = {
        item["token"]: item["name"]
        for item in json.loads((version_root / "category.json").read_text(encoding="utf-8"))
    }
    instances = {
        item["token"]: categories[item["category_token"]]
        for item in json.loads((version_root / "instance.json").read_text(encoding="utf-8"))
    }
    ships_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in json.loads((version_root / "sample_annotation.json").read_text(encoding="utf-8")):
        category = instances.get(ann.get("instance_token"))
        if not is_ship_category(category):
            continue
        translation = ann.get("translation")
        if not isinstance(translation, list) or len(translation) < 2:
            continue
        velocity = ann.get("velocity")
        vx = float(velocity[0]) if isinstance(velocity, list) and len(velocity) >= 2 else 0.0
        vy = float(velocity[1]) if isinstance(velocity, list) and len(velocity) >= 2 else 0.0
        attribute_names = [
            attributes[token]
            for token in ann.get("attribute_tokens") or []
            if token in attributes
        ]
        ships_by_sample[ann["sample_token"]].append(
            {
                "instance_token": ann.get("instance_token"),
                "category": category,
                "x": float(translation[0]),
                "y": float(translation[1]),
                "speed_mps": math.hypot(vx, vy),
                "ship_intentions": ship_intentions_from_attributes(attribute_names),
                "attribute_names": attribute_names,
            }
        )
    return ships_by_sample


def ship_intentions_from_attributes(attribute_names: list[str]) -> list[str]:
    mapping = {
        "ship.entering_lock": "ship_entering_lock",
        "ship.leaving_lock": "ship_leaving_lock",
        "ship.berthed": "ship_berthed",
        "object.static": "object_static",
    }
    return [mapping[name] for name in attribute_names if name in mapping]


def load_scene_berth_slots(scene_json_path: Path) -> dict[str, list[dict[str, Any]]]:
    scenes = json.loads(scene_json_path.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    for scene in scenes:
        slots = []
        for index, item in enumerate(scene.get("ideal_berth_positions") or [], start=1):
            box = item.get("ideal_berth_aabb_xy") if isinstance(item, dict) else None
            if not isinstance(box, dict):
                continue
            slot = dict(box)
            slot["berth_id"] = item.get("berth_id") or f"berth_{index:03d}"
            slot["slot_id"] = f"berth_slot_{index:02d}"
            slot["slot_index"] = index
            slots.append(slot)
        out[scene["token"]] = sorted(slots, key=lambda slot: (slot["cy"], slot["cx"]))
    return out


def load_scene_directions(version_root: Path) -> dict[str, str]:
    directions: dict[str, str] = {}
    summary_path = version_root / "scene_frame_summary_direction_fixed.json"
    if summary_path.exists():
        for item in json.loads(summary_path.read_text(encoding="utf-8")):
            direction = item.get("direction")
            if direction in {"upstream", "downstream"}:
                directions[item["scene_token"]] = direction
    for scene in json.loads((version_root / "scene.json").read_text(encoding="utf-8")):
        if scene["token"] in directions:
            continue
        direction = direction_from_text(scene.get("name", "")) or direction_from_text(
            scene.get("description", "")
        )
        if direction:
            directions[scene["token"]] = direction
    return directions


def direction_from_text(text: str) -> Optional[str]:
    if "_upstream_" in text or "Direction: upstream" in text:
        return "upstream"
    if "_downstream_" in text or "Direction: downstream" in text:
        return "downstream"
    return None


def build_chamber_queue_labels(
    rows: list[dict[str, Any]],
    *,
    ships_by_sample: dict[str, list[dict[str, Any]]],
    berths_by_scene: dict[str, list[dict[str, Any]]],
    direction_by_scene: dict[str, str],
    chamber: dict[str, float],
    approach_margin_m: float,
    max_parallel_actions: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    labels: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for row in rows:
        token = row["token"]
        scene_token = row.get("scene_token")
        direction = direction_by_scene.get(scene_token, "unknown")
        entry_side, exit_side = entry_exit_sides(direction)
        ships = ships_by_sample.get(token, [])
        berths = berths_by_scene.get(scene_token, [])

        berth_slots = berth_slot_occupancy(ships, berths)
        occupied_slots = [slot["slot_id"] for slot in berth_slots if slot["occupied"]]
        available_slots = [slot["slot_id"] for slot in berth_slots if not slot["occupied"]]

        in_chamber = [ship for ship in ships if point_in_chamber(ship["x"], ship["y"], chamber)]
        moving_inside = [
            ship
            for ship in in_chamber
            if ship_is_entering_or_leaving(ship)
        ]
        all_berthed_or_static = all(ship_is_berthed_or_static(ship) for ship in in_chamber)
        no_entering_or_leaving = len(moving_inside) == 0

        queue = rank_entry_queue(
            ships,
            side=entry_side,
            chamber=chamber,
            approach_margin_m=approach_margin_m,
        )
        leave_queue = rank_leave_queue(in_chamber, side=exit_side, chamber=chamber)

        max_entries = min(max_parallel_actions, len(available_slots), len(queue))
        max_departures = min(max_parallel_actions, len(leave_queue))
        labels[token] = {
            "chamber_capacity_available": bool(available_slots),
            "available_berth_slots": available_slots,
            "occupied_berth_slots": occupied_slots,
            "num_occupied_berths": len(occupied_slots),
            "num_ships_in_chamber": len(in_chamber),
            "all_in_chamber_ships_berthed_or_static": all_berthed_or_static,
            "no_ship_entering_or_leaving_inside_chamber": no_entering_or_leaving,
            "queue_rank": queue,
            "next_ship_to_enter_weak": queue[0] if queue else None,
            "next_ship_to_leave_weak": leave_queue[0] if leave_queue else None,
            "max_parallel_entries": max_entries,
            "max_parallel_departures": max_departures,
        }
        diagnostics[token] = {
            "direction": direction,
            "entry_side": entry_side,
            "exit_side": exit_side,
            "in_chamber_ship_tokens": [str(ship["instance_token"]) for ship in in_chamber],
            "moving_inside_ship_tokens": [
                str(ship["instance_token"]) for ship in moving_inside
            ],
            "leave_queue": leave_queue,
            "berth_slots": berth_slots,
        }
    return labels, diagnostics


def entry_exit_sides(direction: str) -> tuple[Optional[str], Optional[str]]:
    if direction == "upstream":
        return "lower", "upper"
    if direction == "downstream":
        return "upper", "lower"
    return None, None


def berth_slot_occupancy(
    ships: list[dict[str, Any]], berths: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    slots = []
    for slot in berths:
        inside = [
            ship
            for ship in ships
            if _inside_box(float(ship["x"]), float(ship["y"]), slot)
        ]
        slots.append(
            {
                "slot_id": slot["slot_id"],
                "berth_id": slot["berth_id"],
                "occupied": bool(inside),
                "ship_count": len(inside),
                "ship_tokens": [str(ship["instance_token"]) for ship in inside],
            }
        )
    return slots


def point_in_chamber(x: float, y: float, chamber: dict[str, float]) -> bool:
    return (
        chamber["x_min"] <= float(x) <= chamber["x_max"]
        and chamber["y_min"] <= float(y) <= chamber["y_max"]
    )


def ship_is_berthed_or_static(ship: dict[str, Any]) -> bool:
    intentions = set(ship.get("ship_intentions") or [])
    return bool(intentions & {"ship_berthed", "object_static"})


def ship_is_entering_or_leaving(ship: dict[str, Any]) -> bool:
    intentions = set(ship.get("ship_intentions") or [])
    return bool(intentions & {"ship_entering_lock", "ship_leaving_lock"})


def rank_entry_queue(
    ships: list[dict[str, Any]],
    *,
    side: Optional[str],
    chamber: dict[str, float],
    approach_margin_m: float,
) -> list[dict[str, Any]]:
    if side is None:
        return []
    candidates = []
    for ship in ships:
        if point_in_chamber(ship["x"], ship["y"], chamber):
            continue
        distance = outside_distance_to_gate(ship["x"], ship["y"], chamber, side)
        if distance is None:
            continue
        if not (
            chamber["x_min"] - approach_margin_m
            <= float(ship["x"])
            <= chamber["x_max"] + approach_margin_m
        ):
            continue
        candidates.append(
            queue_item(
                ship,
                rank=0,
                side=side,
                distance_m=distance,
                source="outside_entry_side_distance",
            )
        )
    candidates.sort(key=lambda item: (item["distance_to_gate_m"], item["instance_token"]))
    return [dict(item, rank=index) for index, item in enumerate(candidates, start=1)]


def rank_leave_queue(
    ships: list[dict[str, Any]], *, side: Optional[str], chamber: dict[str, float]
) -> list[dict[str, Any]]:
    if side is None:
        return []
    candidates = []
    for ship in ships:
        distance = inside_distance_to_gate(ship["y"], chamber, side)
        candidates.append(
            queue_item(
                ship,
                rank=0,
                side=side,
                distance_m=distance,
                source="inside_exit_side_distance",
            )
        )
    candidates.sort(key=lambda item: (item["distance_to_gate_m"], item["instance_token"]))
    return [dict(item, rank=index) for index, item in enumerate(candidates, start=1)]


def outside_distance_to_gate(
    x: float, y: float, chamber: dict[str, float], side: str
) -> Optional[float]:
    del x
    if side == "lower" and float(y) < chamber["y_min"]:
        return round(chamber["y_min"] - float(y), 4)
    if side == "upper" and float(y) > chamber["y_max"]:
        return round(float(y) - chamber["y_max"], 4)
    return None


def inside_distance_to_gate(y: float, chamber: dict[str, float], side: str) -> float:
    if side == "lower":
        return round(max(0.0, float(y) - chamber["y_min"]), 4)
    if side == "upper":
        return round(max(0.0, chamber["y_max"] - float(y)), 4)
    raise ValueError(f"unsupported side: {side}")


def queue_item(
    ship: dict[str, Any], *, rank: int, side: str, distance_m: float, source: str
) -> dict[str, Any]:
    return {
        "rank": rank,
        "instance_token": str(ship["instance_token"]),
        "category": ship.get("category"),
        "side": side,
        "distance_to_gate_m": round(float(distance_m), 4),
        "speed_mps": round(float(ship.get("speed_mps", 0.0)), 4),
        "source": source,
    }


def update_sample_rows(rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> None:
    for row in rows:
        clear_label_fields(row)
        label = labels.get(row.get("token"))
        if label:
            row.update(label)


def update_info_pkls(
    data_root: Path, labels: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    report = []
    seen: set[Path] = set()
    for pattern_root in (data_root, data_root / "infos"):
        for path in sorted(pattern_root.glob("huaiyin_infos_*.pkl")):
            if path in seen:
                continue
            seen.add(path)
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            data_list = payload.get("data_list") if isinstance(payload, dict) else payload
            if not isinstance(data_list, list):
                continue
            changed = 0
            matched_rows = 0
            for item in data_list:
                if not isinstance(item, dict):
                    continue
                clear_label_fields(item)
                label = labels.get(item.get("sample_token"))
                if not label:
                    continue
                matched_rows += 1
                before = {key: item.get(key) for key in label}
                item.update(label)
                if any(before[key] != item.get(key) for key in label):
                    changed += 1
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            report.append(
                {
                    "path": str(path),
                    "matched_rows": matched_rows,
                    "changed_rows": changed,
                }
            )
    return report


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: int(item["timestamp"])):
            out = {
                "sample_token": row["token"],
                "sample_idx": row.get("timestamp_str")
                or row.get("token", "").replace("sample_", ""),
                "timestamp": row.get("timestamp"),
                "scene_token": row.get("scene_token"),
                **labels[row["token"]],
                **diagnostics[row["token"]],
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def build_summary(
    rows: list[dict[str, Any]],
    *,
    labels: dict[str, dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
    sample_path: Path,
    lock_boundary_map: Path,
    chamber: dict[str, float],
    approach_margin_m: float,
    max_parallel_actions: int,
    pkl_report: list[dict[str, Any]],
) -> dict[str, Any]:
    capacity_counts = Counter(
        label["chamber_capacity_available"] for label in labels.values()
    )
    all_berthed_counts = Counter(
        label["all_in_chamber_ships_berthed_or_static"] for label in labels.values()
    )
    queue_frames = {
        "entry_queue_nonempty": sum(
            1 for label in labels.values() if label["next_ship_to_enter_weak"]
        ),
        "leave_queue_nonempty": sum(
            1 for label in labels.values() if label["next_ship_to_leave_weak"]
        ),
    }
    direction_counts = Counter(diag["direction"] for diag in diagnostics.values())
    max_queue_len = max((len(label["queue_rank"]) for label in labels.values()), default=0)
    max_leave_len = max((len(diag["leave_queue"]) for diag in diagnostics.values()), default=0)
    return {
        "settings": {
            "sample_path": str(sample_path),
            "lock_boundary_map": str(lock_boundary_map),
            "lock_chamber_bounds": chamber,
            "approach_margin_m": approach_margin_m,
            "max_parallel_actions_cap": max_parallel_actions,
            "source": (
                "3D ship centers + sample_annotation ship-intention attributes + "
                "ideal berth boxes + physical chamber bounds; "
                "queue labels are weak distance-based planner priors"
            ),
        },
        "num_samples": len(rows),
        "direction_counts": dict(sorted(direction_counts.items())),
        "capacity_counts": {
            str(flag): count for flag, count in sorted(capacity_counts.items())
        },
        "all_in_chamber_ships_berthed_or_static_counts": {
            str(flag): count for flag, count in sorted(all_berthed_counts.items())
        },
        "queue_candidate_frames": queue_frames,
        "num_ships_in_chamber_distribution": dict(
            sorted(Counter(label["num_ships_in_chamber"] for label in labels.values()).items())
        ),
        "available_slot_count_distribution": dict(
            sorted(Counter(len(label["available_berth_slots"]) for label in labels.values()).items())
        ),
        "max_queue_rank_length": max_queue_len,
        "max_leave_queue_length": max_leave_len,
        "pkl_update_report": pkl_report,
    }


def clear_label_fields(row: dict[str, Any]) -> None:
    for key in LABEL_FIELDS:
        row.pop(key, None)


if __name__ == "__main__":
    main()
