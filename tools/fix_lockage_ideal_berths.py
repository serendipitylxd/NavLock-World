#!/usr/bin/env python3
"""Fix ideal berth boxes for the 2025-10-30 upstream line06 lockage."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


TARGET_LOCKAGE_TOKEN = "lockage_2025_10_30_upstream_03_line06"
KEEP_BERTH_ID = "berth_002"
TARGET_BERTH_COUNT = 3
CHAMBER = {"x_min": 39.7, "x_max": 62.7, "y_min": 17.2, "y_max": 307.2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/annotation_fixes/lockage_ideal_berth_fix.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = args.data_root / "v1.0-trainval" / "scene.json"
    scenes = json.loads(scene_path.read_text(encoding="utf-8"))

    target_scenes = [scene for scene in scenes if scene.get("lockage_token") == TARGET_LOCKAGE_TOKEN]
    if not target_scenes:
        raise SystemExit(f"missing target lockage: {TARGET_LOCKAGE_TOKEN}")

    template_scene = target_scenes[0]
    old_berths = template_scene.get("ideal_berth_positions") or []
    keep_berth = next(
        (item for item in old_berths if item.get("berth_id") == KEEP_BERTH_ID),
        None,
    )
    if keep_berth is None:
        raise SystemExit(f"missing {KEEP_BERTH_ID} in target lockage")

    new_berths = build_new_berths(keep_berth)
    checks = validate_berths(new_berths)
    if checks["errors"]:
        raise SystemExit(json.dumps(checks, ensure_ascii=False, indent=2))

    changed = []
    for scene in target_scenes:
        old = copy.deepcopy(scene.get("ideal_berth_positions") or [])
        scene["ideal_berth_positions"] = copy.deepcopy(new_berths)
        scene["lockage_ship_count"] = TARGET_BERTH_COUNT
        changed.append(
            {
                "scene_token": scene["token"],
                "old_lockage_ship_count": template_scene.get("lockage_ship_count"),
                "new_lockage_ship_count": TARGET_BERTH_COUNT,
                "old_num_berths": len(old),
                "new_num_berths": len(new_berths),
                "old_berths": old,
                "new_berths": copy.deepcopy(new_berths),
            }
        )

    if not args.dry_run:
        scene_path.write_text(
            json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    report = {
        "dry_run": args.dry_run,
        "scene_path": str(scene_path),
        "target_lockage_token": TARGET_LOCKAGE_TOKEN,
        "num_changed_scenes": len(changed),
        "geometry_checks": checks,
        "changed_scenes": changed,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote={args.report_output}")
    print(f"dry_run={args.dry_run}")
    print(f"num_changed_scenes={len(changed)}")
    print("new_berths=")
    print(json.dumps(summarize_berths(new_berths), ensure_ascii=False, indent=2))


def build_new_berths(keep_berth: dict[str, Any]) -> list[dict[str, Any]]:
    keep = copy.deepcopy(keep_berth)
    keep_box = keep["ideal_berth_aabb_xy"]
    keep_dx = round(float(keep_box["x_max"]) - float(keep_box["x_min"]), 4)
    new_x_max = CHAMBER["x_max"]
    new_x_min = round(new_x_max - keep_dx, 4)

    lower = berth_from_template(
        keep,
        berth_id="berth_001",
        x_min=new_x_min,
        x_max=new_x_max,
        y_min=27.2,
        y_max=107.2,
    )
    middle = berth_from_template(
        keep,
        berth_id="berth_003",
        x_min=new_x_min,
        x_max=new_x_max,
        y_min=120.0,
        y_max=180.0,
    )
    keep["berth_id"] = KEEP_BERTH_ID
    return [lower, middle, keep]


def berth_from_template(
    template: dict[str, Any],
    *,
    berth_id: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> dict[str, Any]:
    berth = copy.deepcopy(template)
    berth["berth_id"] = berth_id
    box = berth["ideal_berth_aabb_xy"]
    box["x_min"] = round(x_min, 4)
    box["x_max"] = round(x_max, 4)
    box["y_min"] = round(y_min, 4)
    box["y_max"] = round(y_max, 4)
    box["cx"] = round((x_min + x_max) / 2.0, 4)
    box["cy"] = round((y_min + y_max) / 2.0, 4)
    box["dx"] = round(x_max - x_min, 4)
    box["dy"] = round(y_max - y_min, 4)
    box["polygon_xy"] = [
        [box["x_min"], box["y_min"]],
        [box["x_max"], box["y_min"]],
        [box["x_max"], box["y_max"]],
        [box["x_min"], box["y_max"]],
    ]
    return berth


def validate_berths(berths: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = summarize_berths(berths)
    errors: list[str] = []
    for item in summaries:
        if item["id"] == "berth_003" and (
            item["lower_gate_margin_m"] < 10.0 or item["upper_gate_margin_m"] < 10.0
        ):
            errors.append("berth_003 is closer than 10m to a gate boundary")
        if item["x_min"] < CHAMBER["x_min"] - 1e-6 or item["x_max"] > CHAMBER["x_max"] + 0.25:
            errors.append(f"{item['id']} exceeds chamber x range too much")
        if item["y_min"] < CHAMBER["y_min"] - 1e-6 or item["y_max"] > CHAMBER["y_max"] + 1e-6:
            errors.append(f"{item['id']} exceeds chamber y range")
    overlaps = []
    for i, first in enumerate(berths):
        for second in berths[i + 1 :]:
            first_box = first["ideal_berth_aabb_xy"]
            second_box = second["ideal_berth_aabb_xy"]
            overlap_x = max(
                0.0,
                min(first_box["x_max"], second_box["x_max"])
                - max(first_box["x_min"], second_box["x_min"]),
            )
            overlap_y = max(
                0.0,
                min(first_box["y_max"], second_box["y_max"])
                - max(first_box["y_min"], second_box["y_min"]),
            )
            area = overlap_x * overlap_y
            overlaps.append(
                {
                    "first": first["berth_id"],
                    "second": second["berth_id"],
                    "overlap_area_m2": round(area, 6),
                    "y_gap_m": round(
                        max(
                            second_box["y_min"] - first_box["y_max"],
                            first_box["y_min"] - second_box["y_max"],
                            0.0,
                        ),
                        4,
                    ),
                }
            )
            if area > 1e-6:
                errors.append(f"{first['berth_id']} intersects {second['berth_id']}")
    return {"berths": summaries, "pairwise": overlaps, "errors": errors}


def summarize_berths(berths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for berth in berths:
        box = berth["ideal_berth_aabb_xy"]
        out.append(
            {
                "id": berth["berth_id"],
                "x_min": box["x_min"],
                "x_max": box["x_max"],
                "y_min": box["y_min"],
                "y_max": box["y_max"],
                "cx": box["cx"],
                "cy": box["cy"],
                "width_m": round(box["x_max"] - box["x_min"], 4),
                "length_m": round(box["y_max"] - box["y_min"], 4),
                "lower_gate_margin_m": round(box["y_min"] - CHAMBER["y_min"], 4),
                "upper_gate_margin_m": round(CHAMBER["y_max"] - box["y_max"], 4),
            }
        )
    return out


if __name__ == "__main__":
    main()
