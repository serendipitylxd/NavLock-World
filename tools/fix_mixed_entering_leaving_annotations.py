#!/usr/bin/env python3
"""Fix known mixed entering/leaving ship-intention annotation frames."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Optional

from navlock_world.berth_ship_intentions import is_ship_category


ENTERING_TOKEN = "attribute_ship_entering_lock"
LEAVING_TOKEN = "attribute_ship_leaving_lock"
BERTHED_TOKEN = "attribute_ship_berthed"

EARLY_FRAME_IDS = {
    "2025_10_30_18_16_16_066473",
    "2025_10_30_18_16_16_299682",
    "2025_10_30_18_17_06_242020",
    "2025_10_30_18_17_16_220350",
}
LATE_FRAME_IDS = {
    "2025_10_30_18_42_22_306726",
    "2025_10_30_18_42_22_323111",
    "2025_10_30_18_42_22_437771",
    "2025_10_30_18_42_22_546102",
    "2025_10_30_18_42_22_782655",
    "2025_10_30_18_42_23_118575",
}
LATE_LEAVING_SCENE_TOKEN = (
    "scene_2025_10_30_upstream_03_line06_seg038_18_42_16_096287"
)
LATE_LEAVING_START = "2025_10_30_18_42_22_306726"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/annotation_fixes/mixed_entering_leaving_fix.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version_root = args.data_root / "v1.0-trainval"
    sample_path = version_root / "sample.json"
    ann_path = version_root / "sample_annotation.json"

    samples = json.loads(sample_path.read_text(encoding="utf-8"))
    annotations = json.loads(ann_path.read_text(encoding="utf-8"))
    category_by_instance = load_category_by_instance(version_root)
    direction_by_scene = load_scene_directions(version_root)
    samples_by_token = {row["token"]: row for row in samples}

    fixes = find_mixed_label_fixes(
        samples=samples,
        annotations=annotations,
        category_by_instance=category_by_instance,
        direction_by_scene=direction_by_scene,
    )
    changed_ann_tokens = {fix["ann_token"]: fix for fix in fixes}

    if not args.dry_run:
        for ann in annotations:
            fix = changed_ann_tokens.get(ann.get("token"))
            if not fix:
                continue
            ann["attribute_tokens"] = list(fix["new_attribute_tokens"])
        ann_path.write_text(
            json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pkl_report = update_info_pkls(args.data_root, changed_ann_tokens)
    else:
        pkl_report = []

    report = {
        "dry_run": args.dry_run,
        "sample_annotation_path": str(ann_path),
        "num_fixed_annotations": len(fixes),
        "num_fixed_samples": len({fix["sample_token"] for fix in fixes}),
        "fixes": [
            {
                **fix,
                "sample_idx": samples_by_token.get(fix["sample_token"], {}).get(
                    "timestamp_str"
                ),
            }
            for fix in fixes
        ],
        "pkl_update_report": pkl_report,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote={args.report_output}")
    print(f"dry_run={args.dry_run}")
    print(f"num_fixed_annotations={len(fixes)}")
    print(f"num_fixed_samples={report['num_fixed_samples']}")


def load_category_by_instance(version_root: Path) -> dict[str, str]:
    categories = {
        item["token"]: item["name"]
        for item in json.loads((version_root / "category.json").read_text(encoding="utf-8"))
    }
    return {
        item["token"]: categories[item["category_token"]]
        for item in json.loads((version_root / "instance.json").read_text(encoding="utf-8"))
    }


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


def find_mixed_label_fixes(
    *,
    samples: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    category_by_instance: dict[str, str],
    direction_by_scene: dict[str, str],
) -> list[dict[str, Any]]:
    anns_by_sample: dict[str, list[dict[str, Any]]] = {}
    for ann in annotations:
        anns_by_sample.setdefault(ann["sample_token"], []).append(ann)

    fixes: list[dict[str, Any]] = []
    for sample in samples:
        sample_idx = str(sample.get("timestamp_str") or sample["token"].replace("sample_", ""))
        ship_anns = [
            ann
            for ann in anns_by_sample.get(sample["token"], [])
            if is_ship_category(category_by_instance.get(ann.get("instance_token")))
        ]
        manual_fix = manual_frame_fix(sample, sample_idx, ship_anns)
        if manual_fix is not None:
            for fix in manual_fix:
                fix["scene_token"] = sample.get("scene_token")
            fixes.extend(manual_fix)
            continue
        entering = [
            ann for ann in ship_anns if ENTERING_TOKEN in (ann.get("attribute_tokens") or [])
        ]
        leaving = [
            ann for ann in ship_anns if LEAVING_TOKEN in (ann.get("attribute_tokens") or [])
        ]
        if not entering or not leaving:
            continue
        if sample_idx <= "2025_10_30_18_17_16_220350":
            to_fix = leaving
            active_phase = "ship_entering"
            new_attribute_tokens = [BERTHED_TOKEN]
            reason = "manual_fix_before_or_at_2025_10_30_18_17_16_220350_ship1_berthed_ship2_entering"
        elif sample_idx >= "2025_10_30_18_42_22_306726":
            to_fix = entering
            active_phase = "ship_leaving"
            new_attribute_tokens = [LEAVING_TOKEN]
            reason = "manual_fix_after_or_at_2025_10_30_18_42_22_306726_both_ships_leaving"
        else:
            raise SystemExit(
                f"ambiguous mixed entering/leaving frame: {sample.get('timestamp_str')}"
            )
        for ann in to_fix:
            fixes.append(
                {
                    "sample_token": sample["token"],
                    "scene_token": sample.get("scene_token"),
                    "ann_token": ann["token"],
                    "instance_token": ann.get("instance_token"),
                    "old_attribute_tokens": list(ann.get("attribute_tokens") or []),
                    "new_attribute_tokens": list(new_attribute_tokens),
                    "active_phase": active_phase,
                    "reason": reason,
                }
            )
    return fixes


def manual_frame_fix(
    sample: dict[str, Any], sample_idx: str, ship_anns: list[dict[str, Any]]
) -> Optional[list[dict[str, Any]]]:
    if sample_idx in EARLY_FRAME_IDS:
        return explicit_ship_fix(
            sample_idx=sample_idx,
            ship_anns=ship_anns,
            instance_suffix="ship_001",
            new_attribute_tokens=[BERTHED_TOKEN],
            active_phase="ship_entering",
            reason=(
                "manual_fix_2025_10_30_18_17_16_and_before_"
                "ship1_berthed_ship2_entering"
            ),
        )
    is_late_leaving_scene = (
        sample.get("scene_token") == LATE_LEAVING_SCENE_TOKEN
        and sample_idx >= LATE_LEAVING_START
    )
    if sample_idx in LATE_FRAME_IDS or is_late_leaving_scene:
        return explicit_ship_fix(
            sample_idx=sample_idx,
            ship_anns=ship_anns,
            instance_suffix="ship_002",
            new_attribute_tokens=[LEAVING_TOKEN],
            active_phase="ship_leaving",
            reason=(
                "manual_fix_2025_10_30_18_42_22_and_after_"
                "both_ships_leaving"
            ),
        )
    return None


def explicit_ship_fix(
    *,
    sample_idx: str,
    ship_anns: list[dict[str, Any]],
    instance_suffix: str,
    new_attribute_tokens: list[str],
    active_phase: str,
    reason: str,
) -> list[dict[str, Any]]:
    matched = [
        ann
        for ann in ship_anns
        if str(ann.get("instance_token") or "").endswith(instance_suffix)
    ]
    if not matched:
        raise SystemExit(f"{sample_idx}: missing target instance {instance_suffix}")
    fixes: list[dict[str, Any]] = []
    for ann in matched:
        old_attribute_tokens = list(ann.get("attribute_tokens") or [])
        if old_attribute_tokens == new_attribute_tokens:
            continue
        fixes.append(
            {
                "sample_token": ann["sample_token"],
                "scene_token": None,
                "ann_token": ann["token"],
                "instance_token": ann.get("instance_token"),
                "old_attribute_tokens": old_attribute_tokens,
                "new_attribute_tokens": list(new_attribute_tokens),
                "active_phase": active_phase,
                "reason": reason,
            }
        )
    return fixes


def entry_exit_sides(direction: str) -> tuple[Optional[str], Optional[str]]:
    if direction == "upstream":
        return "lower", "upper"
    if direction == "downstream":
        return "upper", "lower"
    return None, None


def update_info_pkls(
    data_root: Path, fixes_by_ann_token: dict[str, dict[str, Any]]
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
                item_changed = False
                for instance in item.get("instances") or []:
                    fix = fixes_by_ann_token.get(instance.get("ann_token"))
                    if not fix:
                        continue
                    instance["attribute_tokens"] = list(fix["new_attribute_tokens"])
                    changed += 1
                    item_changed = True
                if item_changed:
                    matched_rows += 1
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            report.append(
                {
                    "path": str(path),
                    "matched_rows": matched_rows,
                    "changed_instances": changed,
                }
            )
    return report


if __name__ == "__main__":
    main()
