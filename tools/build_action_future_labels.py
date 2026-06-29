#!/usr/bin/env python3
"""Build low-cost action-conditioned future and ship-context labels.

The sample-level future labels are observed-trajectory labels, not
counterfactual outcomes. A frame only looks ahead inside the same 60-second
scene segment.
"""

from __future__ import annotations

import argparse
import json
import pickle
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from navlock_world.berth_ship_intentions import _inside_box


DEFAULT_HORIZONS_SEC = (10, 20, 30)
DEFAULT_MAX_TIME_DELTA_SEC = 2.0

SAMPLE_FUTURE_FIELDS = (
    "state_t_plus_10s",
    "state_t_plus_20s",
    "state_t_plus_30s",
    "phase_t_plus_10s",
    "phase_t_plus_20s",
    "phase_t_plus_30s",
    "future_state_after_observed_action",
    "future_phase_after_observed_action",
)

SHIP_CONTEXT_FIELDS = (
    "assigned_berth_slot",
    "occlusion_state",
    "visibility_level",
)

OCCLUSION_BY_VISIBILITY = {
    "1": "severe_occlusion",
    "2": "moderate_occlusion",
    "3": "mild_occlusion",
    "4": "no_or_minor_occlusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--horizons-sec",
        default=",".join(str(item) for item in DEFAULT_HORIZONS_SEC),
        help="Comma-separated future horizons in seconds. Defaults to 10,20,30.",
    )
    parser.add_argument(
        "--max-time-delta-sec",
        type=float,
        default=DEFAULT_MAX_TIME_DELTA_SEC,
        help="Maximum allowed nearest-frame time delta for a future horizon.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/action_future_labels/summary.json"),
    )
    parser.add_argument(
        "--future-jsonl-output",
        type=Path,
        default=Path("outputs/action_future_labels/action_future_labels.jsonl"),
    )
    parser.add_argument(
        "--ship-jsonl-output",
        type=Path,
        default=Path("outputs/action_future_labels/ship_context_labels.jsonl"),
    )
    parser.add_argument(
        "--no-update-sample",
        action="store_true",
        help="Only write summary/jsonl, do not update sample.json.",
    )
    parser.add_argument(
        "--no-update-annotations",
        action="store_true",
        help="Do not update sample_annotation.json.",
    )
    parser.add_argument(
        "--no-update-pkl",
        action="store_true",
        help="Do not synchronize huaiyin_infos_*.pkl files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons_sec = parse_horizons(args.horizons_sec)
    version_root = args.data_root / "v1.0-trainval"
    sample_path = version_root / "sample.json"
    annotation_path = version_root / "sample_annotation.json"
    scene_path = version_root / "scene.json"
    visibility_path = version_root / "visibility.json"
    category_path = version_root / "category.json"
    instance_path = version_root / "instance.json"

    sample_rows = json.loads(sample_path.read_text(encoding="utf-8"))
    annotation_rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    berths_by_scene = load_scene_berth_slots(scene_path)
    visibility_levels = load_visibility_levels(visibility_path)
    category_by_instance = load_category_by_instance(category_path, instance_path)
    scene_by_sample = {row["token"]: row.get("scene_token") for row in sample_rows}

    future_labels = build_action_future_labels(
        sample_rows,
        horizons_sec=horizons_sec,
        max_time_delta_sec=args.max_time_delta_sec,
    )
    ship_labels = build_ship_context_labels(
        annotation_rows,
        scene_by_sample=scene_by_sample,
        category_by_instance=category_by_instance,
        berths_by_scene=berths_by_scene,
        visibility_levels=visibility_levels,
    )

    if not args.no_update_sample:
        update_sample_rows(sample_rows, future_labels)
        sample_path.write_text(
            json.dumps(sample_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if not args.no_update_annotations:
        update_annotation_rows(annotation_rows, ship_labels)
        annotation_path.write_text(
            json.dumps(annotation_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    pkl_report = []
    if not args.no_update_pkl:
        pkl_report = update_info_pkls(args.data_root, future_labels, ship_labels)

    write_future_jsonl(args.future_jsonl_output, sample_rows, future_labels)
    write_ship_jsonl(args.ship_jsonl_output, annotation_rows, ship_labels)
    summary = build_summary(
        sample_rows,
        annotation_rows,
        future_labels=future_labels,
        ship_labels=ship_labels,
        horizons_sec=horizons_sec,
        max_time_delta_sec=args.max_time_delta_sec,
        pkl_report=pkl_report,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote={args.summary_output}")
    print(f"wrote={args.future_jsonl_output}")
    print(f"wrote={args.ship_jsonl_output}")
    print(f"num_samples={len(sample_rows)}")
    print(f"num_ship_annotations={summary['ship_context']['num_ship_annotations']}")
    print(f"horizon_coverage={summary['future_labels']['horizon_coverage']}")
    print(f"updated_sample={not args.no_update_sample}")
    print(f"updated_annotations={not args.no_update_annotations}")
    print(f"updated_pkl={not args.no_update_pkl}")


def parse_horizons(raw: str) -> list[int]:
    horizons = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not horizons:
        raise SystemExit("--horizons-sec must contain at least one horizon")
    unsupported = [item for item in horizons if item <= 0]
    if unsupported:
        raise SystemExit(f"unsupported non-positive horizons: {unsupported}")
    return sorted(set(horizons))


def build_action_future_labels(
    rows: list[dict[str, Any]],
    *,
    horizons_sec: list[int],
    max_time_delta_sec: float,
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    max_delta_us = int(max_time_delta_sec * 1_000_000)
    for scene_rows in rows_by_scene(rows).values():
        ordered = sorted(scene_rows, key=lambda row: int(row["timestamp"]))
        timestamps = [int(row["timestamp"]) for row in ordered]
        for row in ordered:
            token = row["token"]
            labels[token] = build_empty_future_label(
                row, horizons_sec=horizons_sec
            )
            for horizon_sec in horizons_sec:
                future = find_nearest_future_row(
                    ordered,
                    timestamps,
                    int(row["timestamp"]) + horizon_sec * 1_000_000,
                    max_delta_us=max_delta_us,
                )
                state_key = future_state_field(horizon_sec)
                phase_key = future_phase_field(horizon_sec)
                if future is None:
                    labels[token][state_key] = None
                    labels[token][phase_key] = None
                    continue
                labels[token][state_key] = state_snapshot(
                    future,
                    horizon_sec=horizon_sec,
                    target_timestamp=int(row["timestamp"]) + horizon_sec * 1_000_000,
                )
                labels[token][phase_key] = future.get("operation_phase")

            labels[token]["future_state_after_observed_action"] = {
                "conditioning_action": row.get("observed_action"),
                "source": "observed_trajectory_lookup_same_scene",
                **{
                    f"t_plus_{horizon_sec}s": labels[token][
                        future_state_field(horizon_sec)
                    ]
                    for horizon_sec in horizons_sec
                },
            }
            labels[token]["future_phase_after_observed_action"] = {
                "conditioning_action": row.get("observed_action"),
                "source": "observed_trajectory_lookup_same_scene",
                **{
                    f"t_plus_{horizon_sec}s": labels[token][
                        future_phase_field(horizon_sec)
                    ]
                    for horizon_sec in horizons_sec
                },
            }
    return labels


def rows_by_scene(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("scene_token"))].append(row)
    return out


def build_empty_future_label(
    row: dict[str, Any], *, horizons_sec: list[int]
) -> dict[str, Any]:
    label: dict[str, Any] = {}
    for horizon_sec in horizons_sec:
        label[future_state_field(horizon_sec)] = None
        label[future_phase_field(horizon_sec)] = None
    label["future_state_after_observed_action"] = {
        "conditioning_action": row.get("observed_action"),
        "source": "observed_trajectory_lookup_same_scene",
    }
    label["future_phase_after_observed_action"] = {
        "conditioning_action": row.get("observed_action"),
        "source": "observed_trajectory_lookup_same_scene",
    }
    return label


def find_nearest_future_row(
    ordered: list[dict[str, Any]],
    timestamps: list[int],
    target_timestamp: int,
    *,
    max_delta_us: int,
) -> Optional[dict[str, Any]]:
    index = bisect_left(timestamps, target_timestamp)
    candidates = []
    if index < len(ordered):
        candidates.append(ordered[index])
    if index > 0:
        candidates.append(ordered[index - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda row: abs(int(row["timestamp"]) - target_timestamp))
    if abs(int(best["timestamp"]) - target_timestamp) > max_delta_us:
        return None
    return best


def state_snapshot(
    row: dict[str, Any], *, horizon_sec: int, target_timestamp: int
) -> dict[str, Any]:
    timestamp = int(row["timestamp"])
    return {
        "horizon_sec": horizon_sec,
        "sample_token": row.get("token"),
        "timestamp": timestamp,
        "time_delta_sec": round((timestamp - target_timestamp) / 1_000_000.0, 4),
        "upper_gate_state": row.get("upper_gate_state"),
        "lower_gate_state": row.get("lower_gate_state"),
        "water_state": row.get("lock_water_state"),
        "water_level": row.get("water_level"),
        "operation_phase": row.get("operation_phase"),
        "observed_action": row.get("observed_action"),
    }


def future_state_field(horizon_sec: int) -> str:
    return f"state_t_plus_{horizon_sec}s"


def future_phase_field(horizon_sec: int) -> str:
    return f"phase_t_plus_{horizon_sec}s"


def load_visibility_levels(path: Path) -> dict[str, str]:
    return {
        item["token"]: item.get("level", "")
        for item in json.loads(path.read_text(encoding="utf-8"))
    }


def load_category_by_instance(category_path: Path, instance_path: Path) -> dict[str, str]:
    categories = {
        item["token"]: item["name"]
        for item in json.loads(category_path.read_text(encoding="utf-8"))
    }
    return {
        item["token"]: categories[item["category_token"]]
        for item in json.loads(instance_path.read_text(encoding="utf-8"))
    }


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
            slots.append(slot)
        out[scene["token"]] = sorted(slots, key=lambda slot: (slot["cy"], slot["cx"]))
    return out


def build_ship_context_labels(
    annotations: list[dict[str, Any]],
    *,
    scene_by_sample: dict[str, str],
    category_by_instance: dict[str, str],
    berths_by_scene: dict[str, list[dict[str, Any]]],
    visibility_levels: dict[str, str],
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for ann in annotations:
        category = category_by_instance.get(ann.get("instance_token"), "")
        if not is_vessel_category(category):
            continue
        scene_token = scene_by_sample.get(ann.get("sample_token"))
        berths = berths_by_scene.get(scene_token, [])
        visibility_token = str(ann.get("visibility_token", ""))
        labels[ann["token"]] = {
            "assigned_berth_slot": assigned_berth_slot(ann, berths),
            "occlusion_state": OCCLUSION_BY_VISIBILITY.get(
                visibility_token, "unknown_occlusion"
            ),
            "visibility_level": visibility_levels.get(visibility_token, ""),
        }
    return labels


def is_vessel_category(category: str) -> bool:
    normalized = str(category or "").lower()
    return any(
        marker in normalized
        for marker in ("ship", "fleet", "vessel", "tugboat")
    )


def assigned_berth_slot(
    ann: dict[str, Any], berths: list[dict[str, Any]]
) -> Optional[str]:
    translation = ann.get("translation")
    if not isinstance(translation, list) or len(translation) < 2:
        return None
    x, y = float(translation[0]), float(translation[1])
    for slot in berths:
        if _inside_box(x, y, slot):
            return str(slot["slot_id"])
    return None


def update_sample_rows(
    rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> None:
    for row in rows:
        clear_sample_future_fields(row)
        label = labels.get(row.get("token"))
        if label:
            row.update(label)


def update_annotation_rows(
    rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> None:
    for row in rows:
        clear_ship_context_fields(row)
        label = labels.get(row.get("token"))
        if label:
            row.update(label)


def update_info_pkls(
    data_root: Path,
    future_labels: dict[str, dict[str, Any]],
    ship_labels: dict[str, dict[str, Any]],
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
            sample_matched = 0
            sample_changed = 0
            instance_matched = 0
            instance_changed = 0
            for item in data_list:
                if not isinstance(item, dict):
                    continue
                clear_sample_future_fields(item)
                label = future_labels.get(item.get("sample_token"))
                if label:
                    sample_matched += 1
                    before = {key: item.get(key) for key in label}
                    item.update(label)
                    if any(before[key] != item.get(key) for key in label):
                        sample_changed += 1
                for instance in item.get("instances") or []:
                    if not isinstance(instance, dict):
                        continue
                    clear_ship_context_fields(instance)
                    ship_label = ship_labels.get(instance.get("ann_token"))
                    if not ship_label:
                        continue
                    instance_matched += 1
                    before = {key: instance.get(key) for key in ship_label}
                    instance.update(ship_label)
                    if any(
                        before[key] != instance.get(key) for key in ship_label
                    ):
                        instance_changed += 1
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            report.append(
                {
                    "path": str(path),
                    "sample_matched_rows": sample_matched,
                    "sample_changed_rows": sample_changed,
                    "instance_matched_rows": instance_matched,
                    "instance_changed_rows": instance_changed,
                }
            )
    return report


def clear_sample_future_fields(row: dict[str, Any]) -> None:
    for field in list(row.keys()):
        if field in SAMPLE_FUTURE_FIELDS or (
            field.startswith("state_t_plus_") or field.startswith("phase_t_plus_")
        ):
            row.pop(field, None)


def clear_ship_context_fields(row: dict[str, Any]) -> None:
    for field in SHIP_CONTEXT_FIELDS:
        row.pop(field, None)


def write_future_jsonl(
    path: Path, rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
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
                "observed_action": row.get("observed_action"),
                **labels[row["token"]],
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def write_ship_jsonl(
    path: Path,
    annotations: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ann in annotations:
            label = labels.get(ann.get("token"))
            if not label:
                continue
            out = {
                "sample_token": ann.get("sample_token"),
                "annotation_token": ann.get("token"),
                "instance_token": ann.get("instance_token"),
                "visibility_token": ann.get("visibility_token"),
                **label,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def build_summary(
    sample_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
    *,
    future_labels: dict[str, dict[str, Any]],
    ship_labels: dict[str, dict[str, Any]],
    horizons_sec: list[int],
    max_time_delta_sec: float,
    pkl_report: list[dict[str, Any]],
) -> dict[str, Any]:
    horizon_coverage = {}
    phase_counts = {}
    for horizon_sec in horizons_sec:
        state_key = future_state_field(horizon_sec)
        phase_key = future_phase_field(horizon_sec)
        horizon_coverage[f"{horizon_sec}s"] = sum(
            1 for label in future_labels.values() if label.get(state_key) is not None
        )
        phase_counts[f"{horizon_sec}s"] = dict(
            Counter(
                label.get(phase_key)
                for label in future_labels.values()
                if label.get(phase_key) is not None
            )
        )

    assigned_counts = Counter(
        label["assigned_berth_slot"] or "unassigned"
        for label in ship_labels.values()
    )
    occlusion_counts = Counter(
        label["occlusion_state"] for label in ship_labels.values()
    )
    return {
        "settings": {
            "horizons_sec": horizons_sec,
            "max_time_delta_sec": max_time_delta_sec,
            "future_label_source": "same-scene observed trajectory lookup",
            "ship_occlusion_source": "sample_annotation.visibility_token",
            "assigned_berth_source": "ship center inside scene ideal_berth_aabb_xy",
        },
        "future_labels": {
            "num_samples": len(sample_rows),
            "horizon_coverage": horizon_coverage,
            "phase_counts": phase_counts,
        },
        "ship_context": {
            "num_annotations": len(annotation_rows),
            "num_ship_annotations": len(ship_labels),
            "assigned_berth_slot_counts": dict(assigned_counts),
            "occlusion_state_counts": dict(occlusion_counts),
        },
        "pkl_update_report": pkl_report,
    }


if __name__ == "__main__":
    main()
