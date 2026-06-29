#!/usr/bin/env python3
"""Build annotation-backed ship dispatch labels.

These labels are kept separate from ``observed_action`` so gate/water operation
actions and ship-dispatch actions can coexist in the planner state.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SHIP_DISPATCH_FIELDS = (
    "ship_dispatch_action",
    "ship_dispatch_targets",
    "ship_dispatch_target_count",
    "ship_dispatch_source",
    "ship_dispatch_confidence",
    "ship_dispatch_conflict",
)

VISIBILITY_CONFIDENCE = {
    "1": 0.60,
    "2": 0.75,
    "3": 0.90,
    "4": 1.00,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/ship_dispatch_labels/summary.json"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("outputs/ship_dispatch_labels/ship_dispatch_labels.jsonl"),
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
    rows = json.loads(sample_path.read_text(encoding="utf-8"))
    dispatch_targets = load_dispatch_targets_by_sample(version_root)
    labels = build_ship_dispatch_labels(rows, dispatch_targets)

    if not args.no_update_sample:
        update_sample_rows(rows, labels)
        sample_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    pkl_report = []
    if not args.no_update_pkl:
        pkl_report = update_info_pkls(args.data_root, labels)

    write_jsonl(args.jsonl_output, rows, labels)
    summary = build_summary(rows, labels, sample_path=sample_path, pkl_report=pkl_report)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote={args.summary_output}")
    print(f"wrote={args.jsonl_output}")
    print(f"num_samples={len(rows)}")
    print(f"action_counts={summary['action_counts']}")
    print(f"updated_sample={not args.no_update_sample}")
    print(f"updated_pkl={not args.no_update_pkl}")


def load_dispatch_targets_by_sample(
    version_root: Path,
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
    targets_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in json.loads((version_root / "sample_annotation.json").read_text(encoding="utf-8")):
        category = instances.get(ann.get("instance_token"), "")
        if not is_vessel_category(category):
            continue
        attribute_names = [
            attributes[token]
            for token in ann.get("attribute_tokens") or []
            if token in attributes
        ]
        dispatch_intention = dispatch_intention_from_attributes(attribute_names)
        if dispatch_intention is None:
            continue
        targets_by_sample[ann["sample_token"]].append(
            dispatch_target_item(
                ann,
                category=category,
                dispatch_intention=dispatch_intention,
            )
        )
    return targets_by_sample


def is_vessel_category(category: str) -> bool:
    normalized = str(category or "").lower()
    return any(
        marker in normalized
        for marker in ("ship", "fleet", "vessel", "tugboat")
    )


def dispatch_intention_from_attributes(attribute_names: list[str]) -> str | None:
    names = set(attribute_names)
    if "ship.entering_lock" in names:
        return "ship_entering_lock"
    if "ship.leaving_lock" in names:
        return "ship_leaving_lock"
    return None


def dispatch_target_item(
    ann: dict[str, Any],
    *,
    category: str,
    dispatch_intention: str,
) -> dict[str, Any]:
    visibility_token = str(ann.get("visibility_token", ""))
    action = (
        "dispatch_enter"
        if dispatch_intention == "ship_entering_lock"
        else "dispatch_exit"
    )
    return {
        "instance_token": ann.get("instance_token"),
        "annotation_token": ann.get("token"),
        "category": category,
        "dispatch_action": action,
        "ship_intention": dispatch_intention,
        "assigned_berth_slot": ann.get("assigned_berth_slot"),
        "occlusion_state": ann.get("occlusion_state"),
        "visibility_level": ann.get("visibility_level"),
        "visibility_token": visibility_token,
        "confidence": VISIBILITY_CONFIDENCE.get(visibility_token, 0.50),
    }


def build_ship_dispatch_labels(
    rows: list[dict[str, Any]],
    targets_by_sample: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for row in rows:
        targets = sorted(
            targets_by_sample.get(row["token"], []),
            key=lambda item: (item["dispatch_action"], item["instance_token"] or ""),
        )
        actions = {target["dispatch_action"] for target in targets}
        conflict = len(actions) > 1
        if conflict:
            action = "dispatch_conflict"
        elif actions:
            action = next(iter(actions))
        else:
            action = "hold"
        confidence = (
            min(float(target.get("confidence", 0.5)) for target in targets)
            if targets
            else 1.0
        )
        labels[row["token"]] = {
            "ship_dispatch_action": action,
            "ship_dispatch_targets": targets,
            "ship_dispatch_target_count": len(targets),
            "ship_dispatch_source": "sample_annotation.attribute_tokens",
            "ship_dispatch_confidence": round(confidence, 4),
            "ship_dispatch_conflict": conflict,
        }
    return labels


def update_sample_rows(
    rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> None:
    for row in rows:
        clear_ship_dispatch_fields(row)
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
            matched_rows = 0
            changed_rows = 0
            for item in data_list:
                if not isinstance(item, dict):
                    continue
                clear_ship_dispatch_fields(item)
                label = labels.get(item.get("sample_token"))
                if not label:
                    continue
                matched_rows += 1
                before = {key: item.get(key) for key in label}
                item.update(label)
                if any(before[key] != item.get(key) for key in label):
                    changed_rows += 1
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            report.append(
                {
                    "path": str(path),
                    "matched_rows": matched_rows,
                    "changed_rows": changed_rows,
                }
            )
    return report


def clear_ship_dispatch_fields(row: dict[str, Any]) -> None:
    for field in SHIP_DISPATCH_FIELDS:
        row.pop(field, None)


def write_jsonl(
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
                **labels[row["token"]],
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def build_summary(
    rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    *,
    sample_path: Path,
    pkl_report: list[dict[str, Any]],
) -> dict[str, Any]:
    action_counts = Counter(
        label["ship_dispatch_action"] for label in labels.values()
    )
    target_count_distribution = Counter(
        label["ship_dispatch_target_count"] for label in labels.values()
    )
    occlusion_counts = Counter()
    target_action_counts = Counter()
    confidence_counts = Counter()
    for label in labels.values():
        for target in label["ship_dispatch_targets"]:
            target_action_counts[target["dispatch_action"]] += 1
            occlusion_counts[target.get("occlusion_state") or "unknown_occlusion"] += 1
            confidence_counts[str(target.get("confidence"))] += 1
    return {
        "settings": {
            "sample_path": str(sample_path),
            "source": "sample_annotation.attribute_tokens",
            "scope_note": (
                "ship_dispatch_action is separate from observed_action; "
                "it records annotation-backed ship entering/leaving targets."
            ),
        },
        "num_samples": len(rows),
        "action_counts": dict(action_counts),
        "target_count_distribution": dict(target_count_distribution),
        "target_action_counts": dict(target_action_counts),
        "target_occlusion_counts": dict(occlusion_counts),
        "target_confidence_counts": dict(confidence_counts),
        "conflict_frames": sum(
            1 for label in labels.values() if label["ship_dispatch_conflict"]
        ),
        "pkl_update_report": pkl_report,
    }


if __name__ == "__main__":
    main()
