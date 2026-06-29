#!/usr/bin/env python3
"""Derive entry/exit path-clear labels from 3D ship centers.

Path regions use the physical lock chamber width and a configurable gate-side
length. Ships already inside a scene's ideal berth boxes do not count as path
blockers.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from navlock_world.berth_ship_intentions import _inside_box, is_ship_category
from navlock_world.lock_world_state import load_lock_chamber_bounds, load_scene_berths


DEFAULT_PATH_LENGTH_M = 30.0
LABEL_FIELDS = ("entry_path_clear", "exit_path_clear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--lock-boundary-map",
        type=Path,
        default=None,
        help="Defaults to <data-root>/maps/huaiyin_lock_boundary.json.",
    )
    parser.add_argument("--path-length-m", type=float, default=DEFAULT_PATH_LENGTH_M)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/path_clear_labels/summary.json"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("outputs/path_clear_labels/path_clear.jsonl"),
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
    berths_by_scene = load_scene_berths(version_root / "scene.json")
    direction_by_scene = load_scene_directions(version_root)
    labels, diagnostics = build_path_clear_labels(
        rows,
        ships_by_sample=ships_by_sample,
        berths_by_scene=berths_by_scene,
        direction_by_scene=direction_by_scene,
        chamber=chamber,
        path_length_m=args.path_length_m,
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
        path_length_m=args.path_length_m,
        pkl_report=pkl_report,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote={args.summary_output}")
    print(f"wrote={args.jsonl_output}")
    print(f"num_samples={len(rows)}")
    print(f"clear_counts={summary['clear_counts']}")
    print(f"blocked_counts={summary['blocked_counts']}")
    print(f"updated_sample={not args.no_update_sample}")
    print(f"updated_pkl={not args.no_update_pkl}")


def load_ship_centers_by_sample(version_root: Path) -> dict[str, list[dict[str, Any]]]:
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
        ships_by_sample[ann["sample_token"]].append(
            {
                "instance_token": ann.get("instance_token"),
                "category": category,
                "x": float(translation[0]),
                "y": float(translation[1]),
            }
        )
    return ships_by_sample


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


def build_path_clear_labels(
    rows: list[dict[str, Any]],
    *,
    ships_by_sample: dict[str, list[dict[str, Any]]],
    berths_by_scene: dict[str, list[dict[str, Any]]],
    direction_by_scene: dict[str, str],
    chamber: dict[str, float],
    path_length_m: float,
) -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, Any]]]:
    labels: dict[str, dict[str, bool]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for row in rows:
        token = row["token"]
        scene_token = row.get("scene_token")
        direction = direction_by_scene.get(scene_token, "unknown")
        entry_side, exit_side = entry_exit_sides(direction)
        berths = berths_by_scene.get(scene_token, [])
        blockers_by_side = path_blockers_by_side(
            ships_by_sample.get(token, []),
            berths=berths,
            chamber=chamber,
            path_length_m=path_length_m,
        )
        entry_blockers = blockers_by_side.get(entry_side, []) if entry_side else []
        exit_blockers = blockers_by_side.get(exit_side, []) if exit_side else []
        labels[token] = {
            "entry_path_clear": not entry_blockers,
            "exit_path_clear": not exit_blockers,
        }
        diagnostics[token] = {
            "direction": direction,
            "entry_path_side": entry_side,
            "exit_path_side": exit_side,
            "entry_path_blocker_count": len(entry_blockers),
            "exit_path_blocker_count": len(exit_blockers),
            "entry_path_blocker_tokens": [str(ship["instance_token"]) for ship in entry_blockers],
            "exit_path_blocker_tokens": [str(ship["instance_token"]) for ship in exit_blockers],
        }
    return labels, diagnostics


def entry_exit_sides(direction: str) -> tuple[Optional[str], Optional[str]]:
    if direction == "upstream":
        return "lower", "upper"
    if direction == "downstream":
        return "upper", "lower"
    return None, None


def path_blockers_by_side(
    ships: list[dict[str, Any]],
    *,
    berths: list[dict[str, Any]],
    chamber: dict[str, float],
    path_length_m: float,
) -> dict[str, list[dict[str, Any]]]:
    blockers = {"upper": [], "lower": []}
    for ship in ships:
        x = float(ship["x"])
        y = float(ship["y"])
        if any(_inside_box(x, y, berth) for berth in berths):
            continue
        for side in ("upper", "lower"):
            if point_in_path_region(x, y, chamber, side, path_length_m):
                blockers[side].append(ship)
    return blockers


def point_in_path_region(
    x: float,
    y: float,
    chamber: dict[str, float],
    side: str,
    path_length_m: float,
) -> bool:
    if x < chamber["x_min"] or x > chamber["x_max"]:
        return False
    if side == "lower":
        return chamber["y_min"] <= y <= chamber["y_min"] + path_length_m
    if side == "upper":
        return chamber["y_max"] - path_length_m <= y <= chamber["y_max"]
    raise ValueError(f"unsupported path side: {side}")


def update_sample_rows(
    rows: list[dict[str, Any]], labels: dict[str, dict[str, bool]]
) -> None:
    for row in rows:
        clear_label_fields(row)
        label = labels.get(row.get("token"))
        if label:
            row.update(label)


def update_info_pkls(
    data_root: Path, labels: dict[str, dict[str, bool]]
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
    labels: dict[str, dict[str, bool]],
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
    labels: dict[str, dict[str, bool]],
    diagnostics: dict[str, dict[str, Any]],
    sample_path: Path,
    lock_boundary_map: Path,
    chamber: dict[str, float],
    path_length_m: float,
    pkl_report: list[dict[str, Any]],
) -> dict[str, Any]:
    clear_counts = {
        key: Counter(label[key] for label in labels.values())
        for key in LABEL_FIELDS
    }
    blocked_counts = {
        "entry_path_blocked_frames": sum(
            1 for diag in diagnostics.values() if diag["entry_path_blocker_count"] > 0
        ),
        "exit_path_blocked_frames": sum(
            1 for diag in diagnostics.values() if diag["exit_path_blocker_count"] > 0
        ),
    }
    direction_counts = Counter(diag["direction"] for diag in diagnostics.values())
    return {
        "settings": {
            "sample_path": str(sample_path),
            "lock_boundary_map": str(lock_boundary_map),
            "lock_chamber_bounds": chamber,
            "path_length_m": path_length_m,
            "upper_path_y_range": [
                chamber["y_max"] - path_length_m,
                chamber["y_max"],
            ],
            "lower_path_y_range": [
                chamber["y_min"],
                chamber["y_min"] + path_length_m,
            ],
            "berth_exclusion": "ships whose 3D center is inside any scene ideal_berth_aabb_xy do not block the path",
            "direction_mapping": {
                "upstream": {"entry": "lower", "exit": "upper"},
                "downstream": {"entry": "upper", "exit": "lower"},
            },
        },
        "num_samples": len(rows),
        "direction_counts": dict(sorted(direction_counts.items())),
        "clear_counts": {
            key: {str(flag): count for flag, count in sorted(counter.items())}
            for key, counter in clear_counts.items()
        },
        "blocked_counts": blocked_counts,
        "max_blocker_count": {
            "entry_path": max(
                (diag["entry_path_blocker_count"] for diag in diagnostics.values()),
                default=0,
            ),
            "exit_path": max(
                (diag["exit_path_blocker_count"] for diag in diagnostics.values()),
                default=0,
            ),
        },
        "pkl_update_report": pkl_report,
    }


def clear_label_fields(row: dict[str, Any]) -> None:
    for key in LABEL_FIELDS:
        row.pop(key, None)


if __name__ == "__main__":
    main()
