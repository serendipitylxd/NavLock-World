#!/usr/bin/env python3
"""Derive ship-level operation phase labels from ship intention annotations."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from navlock_world.berth_ship_intentions import is_ship_category
from navlock_world.lock_world_state import load_lock_chamber_bounds


DEFAULT_MAX_GAP_SEC = 120.0
PHASE_FIELDS = (
    "ship_operation_phase",
    "ship_phase_start_time",
    "ship_phase_end_time",
)
LEGACY_PHASE_FIELDS = PHASE_FIELDS + (
    "ship_phase_conflict",
    "ship_phase_conflict_reason",
    "ship_phase_conflict_tokens",
)
SHIP_OPERATION_PHASES = (
    "waiting_for_entry",
    "ship_entering",
    "all_ships_berthed",
    "ship_leaving",
    "lock_clear",
    "ship_phase_uncertain",
)
OPEN_GATE_STATES = frozenset({"open", "opening"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--max-gap-sec", type=float, default=DEFAULT_MAX_GAP_SEC)
    parser.add_argument(
        "--lock-boundary-map",
        type=Path,
        default=None,
        help="Defaults to <data-root>/maps/huaiyin_lock_boundary.json.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/ship_operation_phase_labels/summary.json"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("outputs/ship_operation_phase_labels/ship_operation_phases.jsonl"),
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
    ships_by_sample = load_annotated_ships_by_sample(version_root, chamber=chamber)
    direction_by_scene = load_scene_directions(version_root)
    lockage_key_by_scene = load_scene_lockage_keys(version_root)
    labels, diagnostics = build_ship_phase_labels(
        rows,
        ships_by_sample=ships_by_sample,
        direction_by_scene=direction_by_scene,
        lockage_key_by_scene=lockage_key_by_scene,
        max_gap_sec=args.max_gap_sec,
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
        max_gap_sec=args.max_gap_sec,
        pkl_report=pkl_report,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote={args.summary_output}")
    print(f"wrote={args.jsonl_output}")
    print(f"num_samples={len(rows)}")
    print(f"ship_phase_counts={summary['ship_phase_counts']}")
    print(f"updated_sample={not args.no_update_sample}")
    print(f"updated_pkl={not args.no_update_pkl}")


def load_annotated_ships_by_sample(
    version_root: Path, *, chamber: dict[str, float]
) -> dict[str, list[dict[str, Any]]]:
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
        x, y = float(translation[0]), float(translation[1])
        if not point_in_chamber(x, y, chamber):
            continue
        attribute_names = [
            attributes[token]
            for token in ann.get("attribute_tokens") or []
            if token in attributes
        ]
        ships_by_sample[ann["sample_token"]].append(
            {
                "instance_token": ann.get("instance_token"),
                "category": category,
                "x": x,
                "y": y,
                "ship_intentions": ship_intentions_from_attributes(attribute_names),
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


def point_in_chamber(x: float, y: float, chamber: dict[str, float]) -> bool:
    return (
        chamber["x_min"] <= float(x) <= chamber["x_max"]
        and chamber["y_min"] <= float(y) <= chamber["y_max"]
    )


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


def load_scene_lockage_keys(version_root: Path) -> dict[str, str]:
    summary_path = version_root / "scene_frame_summary_direction_fixed.json"
    if not summary_path.exists():
        return {}
    keys: dict[str, str] = {}
    for item in json.loads(summary_path.read_text(encoding="utf-8")):
        scene_token = item.get("scene_token")
        if not scene_token:
            continue
        parts = [
            item.get("operation_date"),
            item.get("direction"),
            item.get("operation_index"),
            item.get("line_index"),
        ]
        if any(part is None for part in parts):
            keys[scene_token] = str(scene_token)
        else:
            keys[scene_token] = "|".join(str(part) for part in parts)
    return keys


def direction_from_text(text: str) -> Optional[str]:
    if "_upstream_" in text or "Direction: upstream" in text:
        return "upstream"
    if "_downstream_" in text or "Direction: downstream" in text:
        return "downstream"
    return None


def build_ship_phase_labels(
    rows: list[dict[str, Any]],
    *,
    ships_by_sample: dict[str, list[dict[str, Any]]],
    direction_by_scene: dict[str, str],
    lockage_key_by_scene: Optional[dict[str, str]] = None,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    lockage_key_by_scene = lockage_key_by_scene or {}
    no_ship_context = build_no_ship_context(
        rows,
        ships_by_sample=ships_by_sample,
        lockage_key_by_scene=lockage_key_by_scene,
    )
    labels: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for timeline in build_timelines(
        rows, max_gap_sec=max_gap_sec, lockage_key_by_scene=lockage_key_by_scene
    ):
        phases = [
            classify_ship_operation_phase(
                row,
                ships=ships_by_sample.get(row["token"], []),
                direction=direction_by_scene.get(row.get("scene_token"), "unknown"),
                lockage_context=no_ship_context.get(row["token"], {}),
            )
            for row in timeline
        ]
        for start, end in phase_episode_ranges([item["ship_operation_phase"] for item in phases]):
            start_time = timeline[start]["timestamp"]
            end_time = timeline[end]["timestamp"]
            for index in range(start, end + 1):
                token = timeline[index]["token"]
                labels[token] = {
                    "ship_operation_phase": phases[index]["ship_operation_phase"],
                    "ship_phase_start_time": start_time,
                    "ship_phase_end_time": end_time,
                }
                diagnostics[token] = phases[index]["diagnostics"]
    return labels, diagnostics


def classify_ship_operation_phase(
    row: dict[str, Any],
    *,
    ships: list[dict[str, Any]],
    direction: str,
    lockage_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    lockage_context = lockage_context or {}
    counts = intention_counts(ships)
    entering = counts["ship_entering_lock"]
    leaving = counts["ship_leaving_lock"]
    berthed_or_static = counts["ship_berthed"] + counts["object_static"]
    num_ships = len(ships)
    conflict_tokens = {
        "entering": tokens_with_intention(ships, "ship_entering_lock"),
        "leaving": tokens_with_intention(ships, "ship_leaving_lock"),
    }

    mixed_resolution = None
    if entering and leaving:
        phase, mixed_resolution = resolve_mixed_phase_by_open_gate(
            row,
            direction=direction,
            entering_count=entering,
            leaving_count=leaving,
        )
    elif leaving:
        phase = "ship_leaving"
    elif entering:
        phase = "ship_entering"
    elif num_ships == 0:
        phase = lockage_context.get("no_ship_phase") or "lock_clear"
    elif berthed_or_static == num_ships:
        phase = "all_ships_berthed"
    else:
        phase = "ship_phase_uncertain"
    return {
        "ship_operation_phase": phase,
        "diagnostics": {
            "direction": direction,
            "num_annotated_ships_in_chamber": num_ships,
            "ship_intention_counts": dict(sorted(counts.items())),
            "entering_ship_tokens": conflict_tokens["entering"],
            "leaving_ship_tokens": conflict_tokens["leaving"],
            "berthed_or_static_ship_tokens": [
                ship["instance_token"]
                for ship in ships
                if set(ship.get("ship_intentions") or [])
                & {"ship_berthed", "object_static"}
            ],
            "lockage_key": lockage_context.get("lockage_key"),
            "future_entering_in_lockage": lockage_context.get(
                "future_entering_in_lockage", False
            ),
            "no_ship_phase_source": lockage_context.get("no_ship_phase_source"),
            "mixed_entering_leaving_resolution": mixed_resolution,
        },
    }


def resolve_mixed_phase_by_open_gate(
    row: dict[str, Any],
    *,
    direction: str,
    entering_count: int,
    leaving_count: int,
) -> tuple[str, str]:
    entry_side, exit_side = entry_exit_sides(direction)
    if entry_side is not None and exit_side is not None:
        entry_open = row.get(f"{entry_side}_gate_state") in OPEN_GATE_STATES
        exit_open = row.get(f"{exit_side}_gate_state") in OPEN_GATE_STATES
        if entry_open and not exit_open:
            return "ship_entering", "entry_gate_open"
        if exit_open and not entry_open:
            return "ship_leaving", "exit_gate_open"
    if leaving_count > entering_count:
        return "ship_leaving", "leaving_count_majority"
    return "ship_entering", "entering_count_majority_or_tie"


def build_no_ship_context(
    rows: list[dict[str, Any]],
    *,
    ships_by_sample: dict[str, list[dict[str, Any]]],
    lockage_key_by_scene: dict[str, str],
) -> dict[str, dict[str, Any]]:
    by_lockage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: int(item["timestamp"])):
        by_lockage[lockage_key(row, lockage_key_by_scene)].append(row)

    context: dict[str, dict[str, Any]] = {}
    for key, items in by_lockage.items():
        has_entering = [
            any(
                "ship_entering_lock" in set(ship.get("ship_intentions") or [])
                for ship in ships_by_sample.get(row["token"], [])
            )
            for row in items
        ]
        suffix_future_entering = [False] * len(items)
        seen_future_entering = False
        for index in range(len(items) - 1, -1, -1):
            suffix_future_entering[index] = seen_future_entering
            seen_future_entering = seen_future_entering or has_entering[index]
        for index, row in enumerate(items):
            future_entering = suffix_future_entering[index]
            context[row["token"]] = {
                "lockage_key": key,
                "future_entering_in_lockage": future_entering,
                "no_ship_phase": "waiting_for_entry" if future_entering else "lock_clear",
                "no_ship_phase_source": "future_entering_in_same_lockage"
                if future_entering
                else "no_future_entering_in_same_lockage",
            }
    return context


def intention_counts(ships: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ship in ships:
        for label in ship.get("ship_intentions") or []:
            counts[label] += 1
    return counts


def tokens_with_intention(ships: list[dict[str, Any]], intention: str) -> list[str]:
    return [
        str(ship["instance_token"])
        for ship in ships
        if intention in set(ship.get("ship_intentions") or [])
    ]


def entry_exit_sides(direction: str) -> tuple[Optional[str], Optional[str]]:
    if direction == "upstream":
        return "lower", "upper"
    if direction == "downstream":
        return "upper", "lower"
    return None, None


def build_timelines(
    rows: list[dict[str, Any]],
    *,
    max_gap_sec: float,
    lockage_key_by_scene: Optional[dict[str, str]] = None,
) -> list[list[dict[str, Any]]]:
    lockage_key_by_scene = lockage_key_by_scene or {}
    ordered = sorted(rows, key=lambda row: int(row["timestamp"]))
    timelines: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    max_gap_us = int(max_gap_sec * 1_000_000)
    for row in ordered:
        if current:
            prev = current[-1]
            must_split = (
                date_of(prev) != date_of(row)
                or int(row["timestamp"]) - int(prev["timestamp"]) > max_gap_us
                or lockage_key(prev, lockage_key_by_scene)
                != lockage_key(row, lockage_key_by_scene)
            )
            if must_split:
                timelines.append(current)
                current = []
        current.append(row)
    if current:
        timelines.append(current)
    return timelines


def lockage_key(row: dict[str, Any], lockage_key_by_scene: dict[str, str]) -> str:
    scene_token = str(row.get("scene_token") or "")
    return lockage_key_by_scene.get(scene_token, scene_token)


def phase_episode_ranges(phases: list[str]) -> list[tuple[int, int]]:
    if not phases:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(phases)):
        if phases[index] != phases[index - 1]:
            ranges.append((start, index - 1))
            start = index
    ranges.append((start, len(phases) - 1))
    return ranges


def update_sample_rows(rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> None:
    for row in rows:
        clear_phase_fields(row)
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
                clear_phase_fields(item)
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
                "upper_gate_state": row.get("upper_gate_state"),
                "lower_gate_state": row.get("lower_gate_state"),
                "lock_water_state": row.get("lock_water_state"),
                "operation_phase": row.get("operation_phase"),
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
    max_gap_sec: float,
    pkl_report: list[dict[str, Any]],
) -> dict[str, Any]:
    timelines = build_timelines(rows, max_gap_sec=max_gap_sec)
    phase_counts = Counter(label["ship_operation_phase"] for label in labels.values())
    return {
        "settings": {
            "sample_path": str(sample_path),
            "lock_boundary_map": str(lock_boundary_map),
            "max_gap_sec": max_gap_sec,
            "ship_operation_phases": list(SHIP_OPERATION_PHASES),
            "source": (
                "ship-level phase labels from sample_annotation.attribute_tokens; "
                "empty-chamber frames use whole-lockage context: if a future "
                "ship_entering_lock annotation exists in the same lockage, the "
                "frame is waiting_for_entry; otherwise it is lock_clear"
            ),
        },
        "num_samples": len(rows),
        "num_timelines": len(timelines),
        "timeline_lengths": {
            "min": min((len(timeline) for timeline in timelines), default=0),
            "max": max((len(timeline) for timeline in timelines), default=0),
        },
        "ship_phase_counts": dict(sorted(phase_counts.items())),
        "episode_counts": episode_counts(labels),
        "intention_frame_counts": {
            "entering_frames": sum(
                1
                for diag in diagnostics.values()
                if diag["ship_intention_counts"].get("ship_entering_lock", 0)
            ),
            "leaving_frames": sum(
                1
                for diag in diagnostics.values()
                if diag["ship_intention_counts"].get("ship_leaving_lock", 0)
            ),
            "berthed_or_static_frames": sum(
                1 for diag in diagnostics.values() if diag["berthed_or_static_ship_tokens"]
            ),
        },
        "mixed_entering_leaving_resolution_counts": dict(
            sorted(
                Counter(
                    diag.get("mixed_entering_leaving_resolution")
                    for diag in diagnostics.values()
                    if diag.get("mixed_entering_leaving_resolution")
                ).items()
            )
        ),
        "pkl_update_report": pkl_report,
    }


def episode_counts(labels: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    seen: set[tuple[Any, Any, Any]] = set()
    for label in labels.values():
        key = (
            label.get("ship_operation_phase"),
            label.get("ship_phase_start_time"),
            label.get("ship_phase_end_time"),
        )
        if key in seen:
            continue
        seen.add(key)
        counts[str(label.get("ship_operation_phase"))] += 1
    return dict(sorted(counts.items()))


def clear_phase_fields(row: dict[str, Any]) -> None:
    for key in LEGACY_PHASE_FIELDS:
        row.pop(key, None)


def date_of(row: dict[str, Any]) -> str:
    timestamp_str = str(row.get("timestamp_str") or row.get("sample_idx") or "")
    if len(timestamp_str) >= 10:
        return timestamp_str[:10]
    token = str(row.get("token") or "")
    return token.replace("sample_", "")[:10]


if __name__ == "__main__":
    main()
