#!/usr/bin/env python3
"""Visualize frames where entering/leaving ship labels coexist."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Optional

from navlock_world.berth_ship_intentions import is_ship_category
from navlock_world.lock_world_state import load_lock_chamber_bounds


ENTERING_TOKEN = "attribute_ship_entering_lock"
LEAVING_TOKEN = "attribute_ship_leaving_lock"
BERTHED_TOKEN = "attribute_ship_berthed"
EARLY_FRAME_IDS = {
    "2025_10_30_18_16_16_066473",
    "2025_10_30_18_16_16_299682",
    "2025_10_30_18_17_06_242020",
    "2025_10_30_18_17_16_220350",
}
LATE_LEAVING_SCENE_TOKEN = (
    "scene_2025_10_30_upstream_03_line06_seg038_18_42_16_096287"
)
LATE_LEAVING_START = "2025_10_30_18_42_22_306726"
COLORS = {
    ENTERING_TOKEN: "#1a9850",
    LEAVING_TOKEN: "#d73027",
    BERTHED_TOKEN: "#4575b4",
    "attribute_static": "#777777",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/annotation_fixes/mixed_entering_leaving_visualization.html"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version_root = args.data_root / "v1.0-trainval"
    samples = json.loads((version_root / "sample.json").read_text(encoding="utf-8"))
    annotations = json.loads(
        (version_root / "sample_annotation.json").read_text(encoding="utf-8")
    )
    scenes = {
        item["token"]: item
        for item in json.loads((version_root / "scene.json").read_text(encoding="utf-8"))
    }
    summary = load_scene_summary(version_root)
    category_by_instance = load_category_by_instance(version_root)
    attrs = {
        item["token"]: item["name"]
        for item in json.loads((version_root / "attribute.json").read_text(encoding="utf-8"))
    }
    direction_by_scene = load_scene_directions(version_root)
    chamber = load_lock_chamber_bounds(args.data_root / "maps" / "huaiyin_lock_boundary.json")
    if chamber is None:
        raise SystemExit("missing lock chamber map")

    anns_by_sample: dict[str, list[dict[str, Any]]] = {}
    for ann in annotations:
        anns_by_sample.setdefault(ann["sample_token"], []).append(ann)

    frames = []
    for sample in sorted(samples, key=lambda row: int(row["timestamp"])):
        ship_anns = [
            ann
            for ann in anns_by_sample.get(sample["token"], [])
            if is_ship_category(category_by_instance.get(ann.get("instance_token")))
        ]
        entering = [
            ann for ann in ship_anns if ENTERING_TOKEN in (ann.get("attribute_tokens") or [])
        ]
        leaving = [
            ann for ann in ship_anns if LEAVING_TOKEN in (ann.get("attribute_tokens") or [])
        ]
        sample_idx = str(sample.get("timestamp_str") or sample["token"].replace("sample_", ""))
        is_manual_review_frame = sample_idx in EARLY_FRAME_IDS or (
            sample.get("scene_token") == LATE_LEAVING_SCENE_TOKEN
            and sample_idx >= LATE_LEAVING_START
        )
        if (not entering or not leaving) and not is_manual_review_frame:
            continue
        scene = scenes[sample["scene_token"]]
        frame_summary = summary.get(sample["scene_token"], {})
        direction = direction_by_scene.get(sample.get("scene_token"), "unknown")
        frames.append(
            {
                "sample": sample,
                "scene": scene,
                "summary": frame_summary,
                "direction": direction,
                "ships": ship_anns,
                "entering": entering,
                "leaving": leaving,
                "review_reason": review_reason(sample, entering, leaving, is_manual_review_frame),
                "proposed": proposed_resolution(sample, direction),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(frames, chamber, attrs), encoding="utf-8")
    print(f"wrote={args.output}")
    print(f"num_frames={len(frames)}")


def load_scene_summary(version_root: Path) -> dict[str, dict[str, Any]]:
    path = version_root / "scene_frame_summary_direction_fixed.json"
    if not path.exists():
        return {}
    return {item["scene_token"]: item for item in json.loads(path.read_text(encoding="utf-8"))}


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
        name = scene.get("name", "")
        if "_upstream_" in name:
            directions[scene["token"]] = "upstream"
        elif "_downstream_" in name:
            directions[scene["token"]] = "downstream"
    return directions


def proposed_resolution(sample: dict[str, Any], direction: str) -> dict[str, str]:
    entry_side, exit_side = entry_exit_sides(direction)
    entry_open = entry_side and sample.get(f"{entry_side}_gate_state") == "open"
    exit_open = exit_side and sample.get(f"{exit_side}_gate_state") == "open"
    if entry_open and not exit_open:
        return {
            "main_phase": "ship_entering",
            "candidate_fix": "set leaving-labeled ship(s) to ship_berthed",
        }
    if exit_open and not entry_open:
        return {
            "main_phase": "ship_leaving",
            "candidate_fix": "set entering-labeled ship(s) to ship_berthed",
        }
    return {"main_phase": "ambiguous", "candidate_fix": "manual review"}


def review_reason(
    sample: dict[str, Any],
    entering: list[dict[str, Any]],
    leaving: list[dict[str, Any]],
    is_manual_review_frame: bool,
) -> str:
    if entering and leaving:
        return "same-frame entering/leaving labels"
    if is_manual_review_frame:
        return "manual corrected frame / late-lockage leaving check"
    return "manual review"


def entry_exit_sides(direction: str) -> tuple[Optional[str], Optional[str]]:
    if direction == "upstream":
        return "lower", "upper"
    if direction == "downstream":
        return "upper", "lower"
    return None, None


def render_html(
    frames: list[dict[str, Any]], chamber: dict[str, float], attrs: dict[str, str]
) -> str:
    sections = "\n".join(render_frame(frame, chamber, attrs) for frame in frames)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Mixed/manual ship-intention annotation review frames</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    h1 {{ font-size: 22px; margin-bottom: 4px; }}
    h2 {{ font-size: 16px; margin: 24px 0 8px; }}
    .note {{ color: #555; margin-bottom: 16px; }}
    .frame {{ border: 1px solid #ddd; border-radius: 6px; padding: 14px; margin: 14px 0; }}
    .grid {{ display: grid; grid-template-columns: 320px 1fr; gap: 16px; align-items: start; }}
    .meta, table {{ font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f6f6; }}
    .legend span {{ display: inline-block; margin-right: 12px; }}
    .swatch {{ width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
    code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Mixed/manual ship-intention annotation review frames</h1>
  <div class="note">Found {len(frames)} frames. Green = <code>ship.entering_lock</code>, red = <code>ship.leaving_lock</code>, blue = <code>ship.berthed</code>. The SVG is a lock-frame BEV using 3D annotation centers and ideal berth boxes.</div>
  <div class="legend">
    <span><i class="swatch" style="background:{COLORS[ENTERING_TOKEN]}"></i>entering</span>
    <span><i class="swatch" style="background:{COLORS[LEAVING_TOKEN]}"></i>leaving</span>
    <span><i class="swatch" style="background:{COLORS[BERTHED_TOKEN]}"></i>berthed</span>
  </div>
  {sections}
</body>
</html>
"""


def render_frame(
    frame: dict[str, Any], chamber: dict[str, float], attrs: dict[str, str]
) -> str:
    sample = frame["sample"]
    scene = frame["scene"]
    summary = frame["summary"]
    svg = render_bev_svg(frame, chamber, attrs)
    ship_rows = "\n".join(render_ship_row(ship, attrs) for ship in frame["ships"])
    title = html.escape(sample.get("timestamp_str") or sample["token"])
    scene_label = " / ".join(
        str(summary.get(key))
        for key in ("operation_date", "direction", "operation_index", "line_index", "segment_index")
        if summary.get(key) is not None
    )
    return f"""
<div class="frame">
  <h2>{title}</h2>
  <div class="grid">
    <div>{svg}</div>
    <div class="meta">
      <div><b>scene:</b> <code>{html.escape(sample["scene_token"])}</code></div>
      <div><b>lockage:</b> {html.escape(scene_label)}</div>
      <div><b>direction:</b> {html.escape(str(frame["direction"]))}</div>
      <div><b>review reason:</b> {html.escape(str(frame["review_reason"]))}</div>
      <div><b>gate/water:</b> upper={html.escape(str(sample.get("upper_gate_state")))}, lower={html.escape(str(sample.get("lower_gate_state")))}, water={html.escape(str(sample.get("lock_water_state")))}</div>
      <div><b>candidate main phase:</b> <code>{html.escape(frame["proposed"]["main_phase"])}</code></div>
      <div><b>candidate annotation fix:</b> {html.escape(frame["proposed"]["candidate_fix"])}</div>
      <table>
        <thead><tr><th>ship</th><th>attribute</th><th>x</th><th>y</th><th>velocity</th></tr></thead>
        <tbody>{ship_rows}</tbody>
      </table>
    </div>
  </div>
</div>
"""


def render_ship_row(ship: dict[str, Any], attrs: dict[str, str]) -> str:
    token = str(ship.get("instance_token", "")).split("_")[-1]
    attr_tokens = ship.get("attribute_tokens") or []
    attr_names = [attrs.get(token, token) for token in attr_tokens]
    x, y = ship.get("translation", [None, None])[:2]
    velocity = ship.get("velocity")
    return (
        "<tr>"
        f"<td>{html.escape(token)}</td>"
        f"<td>{html.escape(', '.join(attr_names))}</td>"
        f"<td>{float(x):.3f}</td>"
        f"<td>{float(y):.3f}</td>"
        f"<td>{html.escape(str(velocity))}</td>"
        "</tr>"
    )


def render_bev_svg(frame: dict[str, Any], chamber: dict[str, float], attrs: dict[str, str]) -> str:
    scene = frame["scene"]
    width, height = 300, 520
    margin = 28
    x_min, x_max = chamber["x_min"] - 5.0, chamber["x_max"] + 5.0
    y_min, y_max = chamber["y_min"] - 15.0, chamber["y_max"] + 15.0

    def sx(x: float) -> float:
        return margin + (float(x) - x_min) / (x_max - x_min) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - (float(y) - y_min) / (y_max - y_min) * (height - 2 * margin)

    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width/2}" y="16" text-anchor="middle" font-size="12">upper gate / upstream side</text>',
        f'<text x="{width/2}" y="{height-6}" text-anchor="middle" font-size="12">lower gate / downstream side</text>',
        rect(chamber["x_min"], chamber["y_min"], chamber["x_max"], chamber["y_max"], sx, sy, "#f9fafb", "#222", 1.5),
    ]
    for berth in scene.get("ideal_berth_positions") or []:
        box = berth.get("ideal_berth_aabb_xy") or {}
        parts.append(rect(box["x_min"], box["y_min"], box["x_max"], box["y_max"], sx, sy, "#e8f0fe", "#7aa0d8", 1.0))
        parts.append(
            f'<text x="{sx(box["cx"]):.1f}" y="{sy(box["cy"]):.1f}" '
            f'text-anchor="middle" font-size="10" fill="#3b5f9b">{html.escape(berth.get("berth_id", ""))}</text>'
        )
    for ship in frame["ships"]:
        x, y = ship["translation"][:2]
        attr_tokens = ship.get("attribute_tokens") or []
        color = next((COLORS[token] for token in attr_tokens if token in COLORS), "#444")
        name = str(ship.get("instance_token", "")).split("_")[-1]
        parts.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="7" fill="{color}" stroke="#111" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{sx(x)+9:.1f}" y="{sy(y)+4:.1f}" font-size="11" fill="#111">{html.escape(name)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    sx: Any,
    sy: Any,
    fill: str,
    stroke: str,
    stroke_width: float,
) -> str:
    left, right = sorted((sx(x1), sx(x2)))
    top, bottom = sorted((sy(y1), sy(y2)))
    return (
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{right-left:.1f}" '
        f'height="{bottom-top:.1f}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"/>'
    )


if __name__ == "__main__":
    main()
