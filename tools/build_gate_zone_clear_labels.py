#!/usr/bin/env python3
"""Derive 10m gate-zone clear labels from 3D ship centers."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from navlock_world.berth_ship_intentions import is_ship_category
from navlock_world.lock_world_state import load_lock_chamber_bounds


DEFAULT_GATE_ZONE_LENGTH_M = 10.0
LABEL_FIELDS = ("no_ship_in_upper_gate_zone", "no_ship_in_lower_gate_zone")


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
        "--gate-zone-length-m", type=float, default=DEFAULT_GATE_ZONE_LENGTH_M
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/gate_zone_clear_labels/summary.json"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("outputs/gate_zone_clear_labels/gate_zone_clear.jsonl"),
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
    sample_path = args.data_root / "v1.0-trainval" / "sample.json"
    version_root = args.data_root / "v1.0-trainval"
    lock_boundary_map = (
        args.lock_boundary_map or args.data_root / "maps" / "huaiyin_lock_boundary.json"
    )
    chamber = load_lock_chamber_bounds(lock_boundary_map)
    if chamber is None:
        raise SystemExit(f"failed to load lock chamber bounds from {lock_boundary_map}")

    rows = json.loads(sample_path.read_text(encoding="utf-8"))
    ships_by_sample = load_ship_centers_by_sample(version_root)
    labels, diagnostics = build_gate_zone_labels(
        rows,
        ships_by_sample=ships_by_sample,
        chamber=chamber,
        gate_zone_length_m=args.gate_zone_length_m,
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
        gate_zone_length_m=args.gate_zone_length_m,
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
    print(f"occupied_counts={summary['occupied_counts']}")
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


def build_gate_zone_labels(
    rows: list[dict[str, Any]],
    *,
    ships_by_sample: dict[str, list[dict[str, Any]]],
    chamber: dict[str, float],
    gate_zone_length_m: float,
) -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, Any]]]:
    labels: dict[str, dict[str, bool]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for row in rows:
        token = row["token"]
        ships = ships_by_sample.get(token, [])
        upper_tokens = [
            str(ship["instance_token"])
            for ship in ships
            if point_in_gate_zone(ship["x"], ship["y"], chamber, "upper", gate_zone_length_m)
        ]
        lower_tokens = [
            str(ship["instance_token"])
            for ship in ships
            if point_in_gate_zone(ship["x"], ship["y"], chamber, "lower", gate_zone_length_m)
        ]
        labels[token] = {
            "no_ship_in_upper_gate_zone": not upper_tokens,
            "no_ship_in_lower_gate_zone": not lower_tokens,
        }
        diagnostics[token] = {
            "upper_gate_zone_ship_count": len(upper_tokens),
            "lower_gate_zone_ship_count": len(lower_tokens),
            "upper_gate_zone_ship_tokens": upper_tokens,
            "lower_gate_zone_ship_tokens": lower_tokens,
        }
    return labels, diagnostics


def point_in_gate_zone(
    x: float,
    y: float,
    chamber: dict[str, float],
    side: str,
    gate_zone_length_m: float,
) -> bool:
    if x < chamber["x_min"] or x > chamber["x_max"]:
        return False
    if side == "lower":
        return chamber["y_min"] <= y <= chamber["y_min"] + gate_zone_length_m
    if side == "upper":
        return chamber["y_max"] - gate_zone_length_m <= y <= chamber["y_max"]
    raise ValueError(f"unsupported gate zone side: {side}")


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
    gate_zone_length_m: float,
    pkl_report: list[dict[str, Any]],
) -> dict[str, Any]:
    clear_counts = {
        key: Counter(label[key] for label in labels.values())
        for key in LABEL_FIELDS
    }
    occupied_counts = {
        "upper_gate_zone_occupied_frames": sum(
            1 for diag in diagnostics.values() if diag["upper_gate_zone_ship_count"] > 0
        ),
        "lower_gate_zone_occupied_frames": sum(
            1 for diag in diagnostics.values() if diag["lower_gate_zone_ship_count"] > 0
        ),
    }
    return {
        "settings": {
            "sample_path": str(sample_path),
            "lock_boundary_map": str(lock_boundary_map),
            "lock_chamber_bounds": chamber,
            "gate_zone_length_m": gate_zone_length_m,
            "upper_gate_zone_y_range": [
                chamber["y_max"] - gate_zone_length_m,
                chamber["y_max"],
            ],
            "lower_gate_zone_y_range": [
                chamber["y_min"],
                chamber["y_min"] + gate_zone_length_m,
            ],
            "source": "3D ship centers inside physical 10m gate zones",
        },
        "num_samples": len(rows),
        "clear_counts": {
            key: {str(flag): count for flag, count in sorted(counter.items())}
            for key, counter in clear_counts.items()
        },
        "occupied_counts": occupied_counts,
        "max_ship_count": {
            "upper_gate_zone": max(
                (diag["upper_gate_zone_ship_count"] for diag in diagnostics.values()),
                default=0,
            ),
            "lower_gate_zone": max(
                (diag["lower_gate_zone_ship_count"] for diag in diagnostics.values()),
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
