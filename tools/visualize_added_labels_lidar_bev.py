#!/usr/bin/env python3
"""Render LiDAR BEV review images with newly added NavLock labels overlaid."""

from __future__ import annotations

import argparse
import html
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from render_lidar_views_for_vlm import load_lidar_points, render_bev, write_rgb_png


DEFAULT_POINT_CLOUD_RANGE = (0.0, 0.0, -10.0, 102.4, 320.0, 15.0)
HORIZONS = (10, 20, 30)
SHIP_COLORS_RGB = {
    "no_or_minor_occlusion": (66, 201, 143),
    "mild_occlusion": (247, 211, 88),
    "moderate_occlusion": (241, 142, 43),
    "severe_occlusion": (226, 74, 74),
    "unknown_occlusion": (180, 180, 180),
}
TEXT_WHITE = (238, 242, 245)
TEXT_MUTED = (176, 186, 196)
PANEL_BG = (20, 24, 28)
PANEL_LINE = (58, 68, 77)
BERTH_FILL = (30, 82, 140)
BERTH_LINE = (95, 160, 220)
CHAMBER_LINE = (235, 238, 240)
TARGET_LINE = (255, 96, 224)
NEXT_LINE = (108, 218, 255)
PATH_CLEAR = (66, 201, 143)
PATH_BLOCKED = (226, 74, 74)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated sequence splits to sample from.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/annotation_visualizations/lidar_bev_added_labels"),
    )
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Render every frame in the requested splits, ignoring --max-frames.",
    )
    parser.add_argument(
        "--index-page-size",
        type=int,
        default=100,
        help="Number of frames per HTML page when many frames are rendered.",
    )
    parser.add_argument("--sample-token", action="append", default=[])
    parser.add_argument("--scene-token", action="append", default=[])
    parser.add_argument("--bev-width", type=int, default=640)
    parser.add_argument("--bev-height", type=int, default=2000)
    parser.add_argument("--panel-width", type=int, default=620)
    parser.add_argument("--num-point-features", type=int, default=5)
    parser.add_argument(
        "--point-cloud-range",
        default=",".join(str(value) for value in DEFAULT_POINT_CLOUD_RANGE),
        help="Comma-separated x_min,y_min,z_min,x_max,y_max,z_max.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    if not splits:
        raise SystemExit("--splits must contain at least one split")
    point_range = parse_point_cloud_range(args.point_cloud_range)

    scenes = load_scenes(args.data_root / "v1.0-trainval" / "scene.json")
    chamber = load_chamber(args.data_root / "maps" / "huaiyin_lock_boundary.json")
    records = load_frame_records(args.data_root, splits, scenes)
    selected = (
        records
        if args.all_frames and not args.sample_token and not args.scene_token
        else select_records(
            records,
            sample_tokens=set(args.sample_token or []),
            scene_tokens=set(args.scene_token or []),
            max_frames=args.max_frames,
        )
    )
    if not selected:
        raise SystemExit("no frames selected")

    image_root = args.output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    start_time = time.monotonic()
    for index, record in enumerate(selected, start=1):
        output_path = image_root / f"{record['split']}_{record['sample_token']}.png"
        elapsed = time.monotonic() - start_time
        eta = format_duration(elapsed / max(index - 1, 1) * (len(selected) - index + 1))
        progress = progress_bar(index, len(selected))
        if output_path.exists() and not args.overwrite:
            status = "skipped_existing"
        else:
            status = "render"
        print(
            f"[{index}/{len(selected)}] {progress} eta={eta} {status} {record['sample_token']}",
            flush=True,
        )
        if status == "render":
            render_record(
                record,
                output_path=output_path,
                data_root=args.data_root,
                chamber=chamber,
                point_range=point_range,
                bev_size=(args.bev_width, args.bev_height),
                panel_width=args.panel_width,
                num_point_features=args.num_point_features,
            )
            status = "rendered"
        manifest_rows.append(
            {
                "status": status,
                "image": str(output_path),
                "split": record["split"],
                "scene_token": record["scene_token"],
                "sample_token": record["sample_token"],
                "timestamp_str": record.get("timestamp_str"),
                "dispatch_action": lock_state(record).get("ship_dispatch_action"),
                "dispatch_target_count": lock_state(record).get("ship_dispatch_target_count"),
                "operation_phase": lock_state(record).get("operation_phase"),
                "ship_operation_phase": lock_state(record).get("ship_operation_phase"),
                "num_instances_3d": len(record.get("instances_3d") or []),
                "score": record.get("_selection_score"),
                "selection_reasons": record.get("_selection_reasons", []),
                "added_labels": added_label_payload(record),
            }
        )

    manifest = {
        "metadata": {
            "splits": splits,
            "num_selected": len(selected),
            "output_root": str(args.output_root),
            "point_cloud_range": list(point_range),
            "bev_size": [args.bev_width, args.bev_height],
            "panel_width": args.panel_width,
            "label_note": (
                "PNG shows a readable overlay; index.html/manifest.json include "
                "the full added-label JSON for each selected frame."
            ),
        },
        "records": manifest_rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_html_pages(manifest_rows, args.output_root, page_size=args.index_page_size)
    print(f"wrote_index={args.output_root / 'index.html'}")
    print(f"wrote_manifest={args.output_root / 'manifest.json'}")


def progress_bar(index: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0.0%"
    filled = int(round(width * index / total))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {index / total * 100:5.1f}%"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    return f"{minutes:d}m{sec:02d}s"


def parse_point_cloud_range(raw: str) -> tuple[float, float, float, float, float, float]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 6:
        raise ValueError("--point-cloud-range must contain six values")
    return values


def load_scenes(path: Path) -> dict[str, dict[str, Any]]:
    return {scene["token"]: scene for scene in json.loads(path.read_text(encoding="utf-8"))}


def load_chamber(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for region in payload.get("regions") or []:
        if region.get("name") == "lock_chamber":
            x_min, x_max = region["x_range"]
            y_min, y_max = region["y_range"]
            return {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
            }
    raise ValueError(f"missing lock_chamber in {path}")


def load_frame_records(
    data_root: Path, splits: list[str], scenes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in splits:
        path = data_root / "navlock_sequences" / f"scene_sequences_{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sequence in payload.get("sequences") or []:
            scene = scenes.get(sequence.get("scene_token"), {})
            for frame in sequence.get("frames") or []:
                record = {
                    **frame,
                    "split": split,
                    "scene_token": sequence.get("scene_token"),
                    "scene_name": sequence.get("scene_name"),
                    "direction": sequence.get("direction"),
                    "operation_date": sequence.get("operation_date"),
                    "operation_index": sequence.get("operation_index"),
                    "line_index": sequence.get("line_index"),
                    "segment_index": sequence.get("segment_index"),
                    "scene": scene,
                }
                score, reasons = selection_score(record)
                record["_selection_score"] = score
                record["_selection_reasons"] = reasons
                records.append(record)
    return sorted(
        records,
        key=lambda item: (
            str(item.get("split")),
            int(item.get("timestamp") or 0),
            str(item.get("sample_token")),
        ),
    )


def lock_state(record: dict[str, Any]) -> dict[str, Any]:
    state = record.get("lock_state")
    return state if isinstance(state, dict) else {}


def selection_score(record: dict[str, Any]) -> tuple[int, list[str]]:
    state = lock_state(record)
    reasons: list[str] = []
    score = 0
    action = state.get("ship_dispatch_action")
    if action == "dispatch_enter":
        score += 90
        reasons.append("dispatch_enter")
    elif action == "dispatch_exit":
        score += 90
        reasons.append("dispatch_exit")
    elif action == "hold":
        score += 20
        reasons.append("hold")
    target_count = int(state.get("ship_dispatch_target_count") or 0)
    if target_count:
        score += 12 * target_count
        reasons.append(f"{target_count}_dispatch_targets")
    if state.get("queue_rank"):
        score += 20
        reasons.append("queue_rank")
    if state.get("next_ship_to_enter_weak"):
        score += 18
        reasons.append("next_enter")
    if state.get("next_ship_to_leave_weak"):
        score += 18
        reasons.append("next_leave")
    if state.get("available_berth_slots"):
        score += 12
        reasons.append("available_berth_slots")
    if state.get("occupied_berth_slots"):
        score += 12
        reasons.append("occupied_berth_slots")
    for horizon in HORIZONS:
        if state.get(f"state_t_plus_{horizon}s") is not None:
            score += 5
    occlusion_states = [
        ship.get("occlusion_state")
        for ship in record.get("instances_3d") or []
        if is_vessel_instance(ship)
    ]
    if "severe_occlusion" in occlusion_states:
        score += 35
        reasons.append("severe_occlusion")
    elif "moderate_occlusion" in occlusion_states:
        score += 22
        reasons.append("moderate_occlusion")
    return score, reasons


def select_records(
    records: list[dict[str, Any]],
    *,
    sample_tokens: set[str],
    scene_tokens: set[str],
    max_frames: int,
) -> list[dict[str, Any]]:
    if sample_tokens:
        return [record for record in records if record.get("sample_token") in sample_tokens]
    if scene_tokens:
        selected = [record for record in records if record.get("scene_token") in scene_tokens]
        return selected[:max_frames] if max_frames > 0 else selected

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_first(name: str, predicate: Callable[[dict[str, Any]], bool]) -> None:
        candidates = [record for record in records if predicate(record)]
        candidates.sort(key=lambda item: int(item.get("_selection_score") or 0), reverse=True)
        for record in candidates:
            token = str(record.get("sample_token"))
            if token not in seen:
                record.setdefault("_selection_reasons", []).append(name)
                selected.append(record)
                seen.add(token)
                return

    for split in sorted({str(record.get("split")) for record in records}):
        in_split = lambda record, split=split: record.get("split") == split
        add_first(
            f"{split}_dispatch_enter",
            lambda record, in_split=in_split: in_split(record)
            and lock_state(record).get("ship_dispatch_action") == "dispatch_enter",
        )
        add_first(
            f"{split}_dispatch_exit",
            lambda record, in_split=in_split: in_split(record)
            and lock_state(record).get("ship_dispatch_action") == "dispatch_exit",
        )
        add_first(
            f"{split}_hold",
            lambda record, in_split=in_split: in_split(record)
            and lock_state(record).get("ship_dispatch_action") == "hold",
        )
        add_first(
            f"{split}_severe_occlusion",
            lambda record, in_split=in_split: in_split(record)
            and any(
                ship.get("occlusion_state") == "severe_occlusion"
                for ship in record.get("instances_3d") or []
            ),
        )
        add_first(
            f"{split}_queue",
            lambda record, in_split=in_split: in_split(record)
            and bool(lock_state(record).get("queue_rank")),
        )
        add_first(
            f"{split}_next_enter",
            lambda record, in_split=in_split: in_split(record)
            and bool(lock_state(record).get("next_ship_to_enter_weak")),
        )
        add_first(
            f"{split}_next_leave",
            lambda record, in_split=in_split: in_split(record)
            and bool(lock_state(record).get("next_ship_to_leave_weak")),
        )

    if max_frames <= 0:
        return selected
    ranked_by_split: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(
        records, key=lambda item: int(item.get("_selection_score") or 0), reverse=True
    ):
        ranked_by_split.setdefault(str(record.get("split")), []).append(record)
    split_names = sorted(ranked_by_split)
    while len(selected) < max_frames and any(ranked_by_split.values()):
        made_progress = False
        for split in split_names:
            candidates = ranked_by_split.get(split) or []
            while candidates:
                record = candidates.pop(0)
                token = str(record.get("sample_token"))
                if token in seen:
                    continue
                selected.append(record)
                seen.add(token)
                made_progress = True
                break
            if len(selected) >= max_frames:
                break
        if not made_progress:
            break
    return selected[:max_frames]


def render_record(
    record: dict[str, Any],
    *,
    output_path: Path,
    data_root: Path,
    chamber: dict[str, float],
    point_range: tuple[float, float, float, float, float, float],
    bev_size: tuple[int, int],
    panel_width: int,
    num_point_features: int,
) -> None:
    lidar_path = lidar_file_path(record, data_root)
    points = load_lidar_points(lidar_path, num_point_features=num_point_features)
    bev = render_bev(points, point_cloud_range=point_range, size=bev_size)
    overlay_bev_labels(bev, record, chamber, point_range)
    panel = render_panel(record, height=bev.shape[0], width=panel_width)
    combined = np.concatenate([bev, panel], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rgb_png(output_path, combined)


def lidar_file_path(record: dict[str, Any], data_root: Path) -> Path:
    lidar = record.get("lidar") or {}
    raw = lidar.get("file_name") or lidar.get("path")
    if not raw:
        raise ValueError(f"missing lidar file for {record.get('sample_token')}")
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return data_root / path


def overlay_bev_labels(
    image: np.ndarray,
    record: dict[str, Any],
    chamber: dict[str, float],
    point_range: tuple[float, float, float, float, float, float],
) -> None:
    draw_rect_world(image, chamber_rect(chamber), point_range, CHAMBER_LINE, thickness=3)
    draw_gate_lines(image, record, chamber, point_range)
    draw_path_clearance(image, record, chamber, point_range)
    draw_berths(image, record, point_range)
    draw_ships(image, record, point_range)
    draw_overlay_title(image, record)


def chamber_rect(chamber: dict[str, float]) -> tuple[float, float, float, float]:
    return chamber["x_min"], chamber["y_min"], chamber["x_max"], chamber["y_max"]


def world_to_pixel(
    x: float,
    y: float,
    point_range: tuple[float, float, float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    x_min, y_min, _z_min, x_max, y_max, _z_max = point_range
    col = int(round((float(x) - x_min) / (x_max - x_min) * (width - 1)))
    row = int(round((height - 1) - (float(y) - y_min) / (y_max - y_min) * (height - 1)))
    return max(0, min(width - 1, col)), max(0, min(height - 1, row))


def draw_rect_world(
    image: np.ndarray,
    rect_xy: tuple[float, float, float, float],
    point_range: tuple[float, float, float, float, float, float],
    color: tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = rect_xy
    p1 = world_to_pixel(x1, y1, point_range, width, height)
    p2 = world_to_pixel(x2, y2, point_range, width, height)
    left, right = sorted((p1[0], p2[0]))
    top, bottom = sorted((p1[1], p2[1]))
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness, cv2.LINE_AA)


def draw_gate_lines(
    image: np.ndarray,
    record: dict[str, Any],
    chamber: dict[str, float],
    point_range: tuple[float, float, float, float, float, float],
) -> None:
    state = lock_state(record)
    height, width = image.shape[:2]
    for side, y_key, clear_key in (
        ("upper", "y_max", "no_ship_in_upper_gate_zone"),
        ("lower", "y_min", "no_ship_in_lower_gate_zone"),
    ):
        y = chamber[y_key]
        color = PATH_CLEAR if state.get(clear_key) is True else PATH_BLOCKED
        p1 = world_to_pixel(chamber["x_min"], y, point_range, width, height)
        p2 = world_to_pixel(chamber["x_max"], y, point_range, width, height)
        cv2.line(image, p1, p2, color, 6, cv2.LINE_AA)
        label = f"{side}: {state.get(f'{side}_gate_state')} clear={state.get(clear_key)}"
        put_text_with_bg(image, label, (min(p1[0], p2[0]) + 6, min(p1[1], p2[1]) - 8), color)


def draw_path_clearance(
    image: np.ndarray,
    record: dict[str, Any],
    chamber: dict[str, float],
    point_range: tuple[float, float, float, float, float, float],
) -> None:
    state = lock_state(record)
    height, width = image.shape[:2]
    center_x = (chamber["x_min"] + chamber["x_max"]) / 2.0
    y1, y2 = chamber["y_min"] + 16.0, chamber["y_max"] - 16.0
    entry_color = PATH_CLEAR if state.get("entry_path_clear") is True else PATH_BLOCKED
    exit_color = PATH_CLEAR if state.get("exit_path_clear") is True else PATH_BLOCKED
    p1 = world_to_pixel(center_x - 1.5, y1, point_range, width, height)
    p2 = world_to_pixel(center_x - 1.5, y2, point_range, width, height)
    p3 = world_to_pixel(center_x + 1.5, y1, point_range, width, height)
    p4 = world_to_pixel(center_x + 1.5, y2, point_range, width, height)
    cv2.line(image, p1, p2, entry_color, 2, cv2.LINE_AA)
    cv2.line(image, p3, p4, exit_color, 2, cv2.LINE_AA)
    put_text_with_bg(
        image,
        f"entry_clear={state.get('entry_path_clear')}",
        (p1[0] - 150, max(26, (p1[1] + p2[1]) // 2 - 12)),
        entry_color,
    )
    put_text_with_bg(
        image,
        f"exit_clear={state.get('exit_path_clear')}",
        (p3[0] + 16, max(46, (p3[1] + p4[1]) // 2 + 14)),
        exit_color,
    )


def draw_berths(
    image: np.ndarray,
    record: dict[str, Any],
    point_range: tuple[float, float, float, float, float, float],
) -> None:
    berths = berth_slots(record.get("scene") or {})
    for berth in berths:
        box = berth["box"]
        draw_filled_rect_world(
            image,
            (box["x_min"], box["y_min"], box["x_max"], box["y_max"]),
            point_range,
            fill=BERTH_FILL,
            line=BERTH_LINE,
            alpha=0.18,
            thickness=2,
        )
        height, width = image.shape[:2]
        cx, cy = world_to_pixel(box["cx"], box["cy"], point_range, width, height)
        put_text_with_bg(image, f"{berth['slot_id']} {berth['berth_id']}", (cx - 70, cy), BERTH_LINE)


def berth_slots(scene: dict[str, Any]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for index, berth in enumerate(scene.get("ideal_berth_positions") or [], start=1):
        box = berth.get("ideal_berth_aabb_xy") if isinstance(berth, dict) else None
        if not isinstance(box, dict):
            continue
        slots.append(
            {
                "slot_id": f"berth_slot_{index:02d}",
                "berth_id": str(berth.get("berth_id") or f"berth_{index:03d}"),
                "box": box,
            }
        )
    return sorted(slots, key=lambda item: (item["box"].get("cy", 0), item["box"].get("cx", 0)))


def draw_filled_rect_world(
    image: np.ndarray,
    rect_xy: tuple[float, float, float, float],
    point_range: tuple[float, float, float, float, float, float],
    *,
    fill: tuple[int, int, int],
    line: tuple[int, int, int],
    alpha: float,
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = rect_xy
    p1 = world_to_pixel(x1, y1, point_range, width, height)
    p2 = world_to_pixel(x2, y2, point_range, width, height)
    left, right = sorted((p1[0], p2[0]))
    top, bottom = sorted((p1[1], p2[1]))
    overlay = image.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), fill, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, dst=image)
    cv2.rectangle(image, (left, top), (right, bottom), line, thickness, cv2.LINE_AA)


def draw_ships(
    image: np.ndarray,
    record: dict[str, Any],
    point_range: tuple[float, float, float, float, float, float],
) -> None:
    state = lock_state(record)
    targets = {
        target.get("annotation_token") or target.get("instance_token"): target
        for target in state.get("ship_dispatch_targets") or []
    }
    target_instances = {target.get("instance_token") for target in state.get("ship_dispatch_targets") or []}
    next_enter = (state.get("next_ship_to_enter_weak") or {}).get("instance_token")
    next_leave = (state.get("next_ship_to_leave_weak") or {}).get("instance_token")
    height, width = image.shape[:2]
    vessel_index = 0
    for ship in record.get("instances_3d") or []:
        if not is_vessel_instance(ship):
            continue
        vessel_index += 1
        translation = ship.get("translation") or []
        if len(translation) < 2:
            continue
        x, y = float(translation[0]), float(translation[1])
        px, py = world_to_pixel(x, y, point_range, width, height)
        occ = str(ship.get("occlusion_state") or "unknown_occlusion")
        color = SHIP_COLORS_RGB.get(occ, SHIP_COLORS_RGB["unknown_occlusion"])
        if ship.get("size") and ship.get("rotation"):
            draw_ship_footprint(image, ship, point_range, color)
        cv2.circle(image, (px, py), 10, color, -1, cv2.LINE_AA)
        cv2.circle(image, (px, py), 12, (12, 12, 12), 2, cv2.LINE_AA)
        is_target = (
            ship.get("annotation_token") in targets
            or ship.get("instance_token") in target_instances
        )
        if is_target:
            cv2.circle(image, (px, py), 22, TARGET_LINE, 4, cv2.LINE_AA)
            put_text_with_bg(
                image,
                str((targets.get(ship.get("annotation_token")) or {}).get("dispatch_action") or state.get("ship_dispatch_action")),
                (px + 16, py - 42),
                TARGET_LINE,
            )
        if ship.get("instance_token") in {next_enter, next_leave}:
            cv2.circle(image, (px, py), 28, NEXT_LINE, 2, cv2.LINE_AA)
        label_lines = ship_label_lines(ship, vessel_index)
        draw_label_box(image, label_lines, (px + 14, py + 14), color)


def is_vessel_instance(item: dict[str, Any]) -> bool:
    category = str(item.get("category") or "").lower()
    if any(marker in category for marker in ("ship", "fleet", "vessel", "tugboat")):
        return True
    if item.get("ship_intentions"):
        return True
    if item.get("assigned_berth_slot"):
        return True
    return False


def draw_ship_footprint(
    image: np.ndarray,
    ship: dict[str, Any],
    point_range: tuple[float, float, float, float, float, float],
    color: tuple[int, int, int],
) -> None:
    translation = ship.get("translation") or []
    size = ship.get("size") or []
    if len(translation) < 2 or len(size) < 2:
        return
    width_m, length_m = float(size[0]), float(size[1])
    if not math.isfinite(width_m) or not math.isfinite(length_m) or width_m <= 0 or length_m <= 0:
        return
    yaw = yaw_from_quaternion(ship.get("rotation"))
    if yaw is None:
        return
    x, y = float(translation[0]), float(translation[1])
    # nuScenes boxes store size as [width, length, height], while the box
    # local x-axis is the length axis and local y-axis is the width axis.
    half_w = min(width_m, 28.0) / 2.0
    half_l = min(length_m, 240.0) / 2.0
    local = [(-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)]
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    height, width = image.shape[:2]
    points = []
    for lx, ly in local:
        wx = x + lx * cos_yaw - ly * sin_yaw
        wy = y + lx * sin_yaw + ly * cos_yaw
        points.append(world_to_pixel(wx, wy, point_range, width, height))
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.11, image, 0.89, 0, dst=image)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)


def yaw_from_quaternion(rotation: Any) -> Optional[float]:
    if not isinstance(rotation, list) or len(rotation) < 4:
        return None
    w, x, y, z = (float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def ship_label_lines(ship: dict[str, Any], index: int) -> list[str]:
    instance = str(ship.get("instance_token") or f"ship_{index:03d}")
    ship_name = short_ship_name(instance, index)
    intentions = ",".join(short_intention(item) for item in ship.get("ship_intentions") or [])
    attrs = ",".join(short_attribute(item) for item in ship.get("attribute_names") or [])
    return [
        ship_name,
        f"intent={intentions or attrs or '-'}",
        f"slot={short_slot(ship.get('assigned_berth_slot'))}",
        f"occ={short_occlusion(ship.get('occlusion_state'))}",
        f"vis={ship.get('visibility_level') or '-'}",
    ]


def short_ship_name(instance_token: str, fallback_index: int) -> str:
    if "_ship_" in instance_token:
        return "ship_" + instance_token.rsplit("_ship_", 1)[-1]
    suffix = instance_token.split("_")[-1]
    return suffix if suffix else f"ship_{fallback_index:03d}"


def short_intention(value: Any) -> str:
    text = str(value or "")
    return text.replace("ship_", "").replace("_lock", "")


def short_attribute(value: Any) -> str:
    text = str(value or "")
    return text.replace("ship.", "").replace("_lock", "")


def short_slot(value: Any) -> str:
    if not value:
        return "-"
    return str(value).replace("berth_slot_", "slot")


def short_occlusion(value: Any) -> str:
    return str(value or "-").replace("_occlusion", "").replace("no_or_minor", "no/minor")


def draw_overlay_title(image: np.ndarray, record: dict[str, Any]) -> None:
    state = lock_state(record)
    title = f"{record.get('split')} {record.get('timestamp_str')} {state.get('ship_dispatch_action')}"
    put_text_with_bg(image, title, (12, 34), TEXT_WHITE)
    legend = "occlusion: green=no/minor  yellow=mild  orange=moderate  red=severe  magenta=dispatch target"
    put_text_with_bg(image, legend, (12, image.shape[0] - 16), TEXT_MUTED)


def put_text_with_bg(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.48,
    thickness: int = 1,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(str(text), font, scale, thickness)
    x = max(0, min(image.shape[1] - tw - 6, x))
    y = max(th + 4, min(image.shape[0] - baseline - 4, y))
    cv2.rectangle(image, (x - 3, y - th - 5), (x + tw + 4, y + baseline + 4), (0, 0, 0), -1)
    cv2.putText(image, str(text), (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_label_box(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    accent: tuple[int, int, int],
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max((size[0] for size in text_sizes), default=40) + 12
    line_height = 17
    height = line_height * len(lines) + 8
    x = max(0, min(image.shape[1] - width - 1, x))
    y = max(height + 1, min(image.shape[0] - 1, y))
    top = y - height
    cv2.rectangle(image, (x, top), (x + width, y), (8, 10, 12), -1)
    cv2.rectangle(image, (x, top), (x + width, y), accent, 1, cv2.LINE_AA)
    for idx, line in enumerate(lines):
        color = accent if idx == 0 else TEXT_WHITE
        cv2.putText(
            image,
            line,
            (x + 6, top + 16 + idx * line_height),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def render_panel(record: dict[str, Any], *, height: int, width: int) -> np.ndarray:
    panel = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    state = lock_state(record)
    lines = panel_lines(record, state)
    y = 28
    for kind, text in lines:
        if y > height - 24:
            break
        if kind == "section":
            y += 8
            cv2.line(panel, (18, y), (width - 18, y), PANEL_LINE, 1, cv2.LINE_AA)
            y += 24
            color = (125, 190, 255)
            scale = 0.54
            thickness = 1
        elif kind == "muted":
            color = TEXT_MUTED
            scale = 0.43
            thickness = 1
        elif kind == "warn":
            color = (255, 184, 102)
            scale = 0.45
            thickness = 1
        else:
            color = TEXT_WHITE
            scale = 0.45
            thickness = 1
        wrapped = wrap_text(text, max_chars=78 if kind != "section" else 64)
        for part in wrapped:
            if y > height - 24:
                break
            cv2.putText(
                panel,
                part,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += 18 if kind != "section" else 20
    return panel


def panel_lines(record: dict[str, Any], state: dict[str, Any]) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = [
        ("section", "Frame"),
        ("normal", f"sample={record.get('sample_token')}"),
        ("normal", f"time={record.get('timestamp_str')} split={record.get('split')}"),
        ("muted", f"scene={record.get('scene_token')}"),
        (
            "muted",
            f"lockage={record.get('operation_date')} op={record.get('operation_index')} "
            f"line={record.get('line_index')} seg={record.get('segment_index')} dir={record.get('direction')}",
        ),
        ("section", "Gate / Water / Observed Action"),
        (
            "normal",
            f"upper={state.get('upper_gate_state')} lower={state.get('lower_gate_state')} "
            f"water={state.get('water_state')} chamber_level={state.get('water_level')}",
        ),
        (
            "normal",
            f"upstream_water_level={state.get('upstream_water_level')} "
            f"downstream_water_level={state.get('downstream_water_level')}",
        ),
        (
            "normal",
            f"observed_action={state.get('observed_action')} target={state.get('action_target')} "
            f"conf={state.get('action_confidence')}",
        ),
        ("normal", f"operation_phase={state.get('operation_phase')}"),
        ("normal", f"ship_operation_phase={state.get('ship_operation_phase')}"),
        ("section", "Rule / Planner State"),
        (
            "normal",
            f"valid_actions={compact_list(state.get('valid_actions'))}",
        ),
        (
            "normal",
            f"invalid_actions({len(state.get('invalid_actions') or [])})="
            f"{compact_list(state.get('invalid_actions'), max_items=8)}",
        ),
        (
            "normal",
            f"gate_zone_clear upper={state.get('no_ship_in_upper_gate_zone')} "
            f"lower={state.get('no_ship_in_lower_gate_zone')}",
        ),
        (
            "normal",
            f"path_clear entry={state.get('entry_path_clear')} exit={state.get('exit_path_clear')}",
        ),
        (
            "normal",
            f"capacity_available={state.get('chamber_capacity_available')} "
            f"occupied={compact_list(state.get('occupied_berth_slots'))} "
            f"available={compact_list(state.get('available_berth_slots'))}",
        ),
        (
            "normal",
            f"max_parallel entries={state.get('max_parallel_entries')} departures={state.get('max_parallel_departures')}",
        ),
        (
            "normal",
            f"next_enter={short_next_ship(state.get('next_ship_to_enter_weak'))}",
        ),
        (
            "normal",
            f"next_leave={short_next_ship(state.get('next_ship_to_leave_weak'))}",
        ),
        ("section", "Ship Dispatch"),
        (
            "normal",
            f"ship_dispatch_action={state.get('ship_dispatch_action')} "
            f"targets={state.get('ship_dispatch_target_count')} "
            f"conf={state.get('ship_dispatch_confidence')} conflict={state.get('ship_dispatch_conflict')}",
        ),
    ]
    for target in state.get("ship_dispatch_targets") or []:
        lines.append(
            (
                "normal",
                "target "
                f"{short_ship_name(str(target.get('instance_token') or ''), 0)} "
                f"{target.get('dispatch_action')} "
                f"slot={short_slot(target.get('assigned_berth_slot'))} "
                f"occ={short_occlusion(target.get('occlusion_state'))} "
                f"conf={target.get('confidence')}",
            )
        )
    lines.extend(
        [
            ("section", "Future Labels"),
            ("normal", f"future_action_condition={state.get('observed_action')}"),
        ]
    )
    for horizon in HORIZONS:
        future = state.get(f"state_t_plus_{horizon}s") or {}
        if future:
            lines.append(
                (
                    "normal",
                    f"+{horizon}s phase={state.get(f'phase_t_plus_{horizon}s')} "
                    f"upper={future.get('upper_gate_state')} lower={future.get('lower_gate_state')} "
                    f"water={future.get('water_state')} level={future.get('water_level')}",
                )
            )
        else:
            lines.append(("warn", f"+{horizon}s state=None phase={state.get(f'phase_t_plus_{horizon}s')}"))
    future_state = state.get("future_state_after_observed_action") or {}
    if future_state:
        lines.append(
            (
                "muted",
                f"conditioned_future_source={future_state.get('source')} action={future_state.get('conditioning_action')}",
            )
        )
    lines.append(("section", "Ship-Level Added Labels"))
    ships = [ship for ship in record.get("instances_3d") or [] if is_vessel_instance(ship)]
    for idx, ship in enumerate(ships, start=1):
        lines.append(
            (
                "normal",
                f"{short_ship_name(str(ship.get('instance_token') or ''), idx)} "
                f"intent={compact_list([short_intention(x) for x in ship.get('ship_intentions') or []])} "
                f"slot={short_slot(ship.get('assigned_berth_slot'))} "
                f"occ={short_occlusion(ship.get('occlusion_state'))} "
                f"vis={ship.get('visibility_level') or '-'}",
            )
        )
    return lines


def compact_list(values: Any, max_items: int = 6) -> str:
    if values is None:
        return "-"
    if not isinstance(values, list):
        return str(values)
    shown = [str(value) for value in values[:max_items]]
    suffix = "" if len(values) <= max_items else f",...+{len(values)-max_items}"
    return "[" + ",".join(shown) + suffix + "]"


def short_next_ship(value: Any) -> str:
    if not isinstance(value, dict):
        return "-"
    name = short_ship_name(str(value.get("instance_token") or ""), 0)
    side = value.get("side")
    distance = value.get("distance_to_gate_m")
    rank = value.get("rank")
    return f"{name}/rank={rank}/side={side}/dist={distance}"


def wrap_text(text: str, max_chars: int) -> list[str]:
    text = str(text)
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for word in text.split(" "):
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            parts.extend(split_long_token(current, max_chars))
            current = word
    if current:
        parts.extend(split_long_token(current, max_chars))
    return parts


def split_long_token(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]


def added_label_payload(record: dict[str, Any]) -> dict[str, Any]:
    state = lock_state(record)
    sample_fields = [
        "upstream_water_level",
        "downstream_water_level",
        "observed_action",
        "action_start_time",
        "action_end_time",
        "action_target",
        "action_source",
        "action_confidence",
        "operation_phase",
        "ship_operation_phase",
        "no_ship_in_upper_gate_zone",
        "no_ship_in_lower_gate_zone",
        "entry_path_clear",
        "exit_path_clear",
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
        "valid_actions",
        "invalid_actions",
        "violation_reason",
        "state_t_plus_10s",
        "state_t_plus_20s",
        "state_t_plus_30s",
        "phase_t_plus_10s",
        "phase_t_plus_20s",
        "phase_t_plus_30s",
        "future_state_after_observed_action",
        "future_phase_after_observed_action",
        "ship_dispatch_action",
        "ship_dispatch_targets",
        "ship_dispatch_target_count",
        "ship_dispatch_source",
        "ship_dispatch_confidence",
        "ship_dispatch_conflict",
    ]
    ship_fields = [
        "instance_token",
        "annotation_token",
        "category",
        "translation",
        "ship_intentions",
        "attribute_names",
        "assigned_berth_slot",
        "occlusion_state",
        "visibility_level",
    ]
    return {
        "sample_level": {field: state.get(field) for field in sample_fields if field in state},
        "ship_level": [
            {field: ship.get(field) for field in ship_fields if field in ship}
            for ship in record.get("instances_3d") or []
            if is_vessel_instance(ship)
        ],
    }


def write_html_pages(rows: list[dict[str, Any]], output_root: Path, page_size: int) -> None:
    if page_size <= 0 or len(rows) <= page_size:
        (output_root / "index.html").write_text(
            render_index_html(
                rows,
                output_root,
                title="LiDAR BEV added-label review",
                note=(
                    "Each PNG overlays added labels on the LiDAR top-view point cloud. "
                    "The image panel keeps the readable summary; the details block below "
                    "each frame contains the full added-label JSON."
                ),
                nav_html="",
            ),
            encoding="utf-8",
        )
        return

    pages = [rows[index : index + page_size] for index in range(0, len(rows), page_size)]
    links = []
    for page_index, page_rows in enumerate(pages, start=1):
        page_name = f"page_{page_index:03d}.html"
        first = page_rows[0]
        last = page_rows[-1]
        links.append(
            (
                page_name,
                page_index,
                len(page_rows),
                first.get("timestamp_str"),
                last.get("timestamp_str"),
            )
        )
        page_nav = render_page_nav(links=[], current=page_index, total=len(pages))
        (output_root / page_name).write_text(
            render_index_html(
                page_rows,
                output_root,
                title=f"LiDAR BEV added-label review page {page_index:03d}",
                note=f"Frames {((page_index - 1) * page_size) + 1}-{((page_index - 1) * page_size) + len(page_rows)} of {len(rows)}.",
                nav_html=page_nav,
            ),
            encoding="utf-8",
        )

    (output_root / "index.html").write_text(
        render_landing_html(rows, links),
        encoding="utf-8",
    )


def render_landing_html(rows: list[dict[str, Any]], links: list[tuple[str, int, int, Any, Any]]) -> str:
    link_items = "\n".join(
        f'<li><a href="{html.escape(page_name)}">page {page_index:03d}</a> '
        f'({count} frames, {html.escape(str(first_time))} -> {html.escape(str(last_time))})</li>'
        for page_name, page_index, count, first_time, last_time in links
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LiDAR BEV added-label review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 22px; color: #202124; }}
    h1 {{ font-size: 22px; margin-bottom: 6px; }}
    .note {{ color: #555; margin-bottom: 18px; line-height: 1.4; }}
    li {{ margin: 6px 0; }}
    code {{ background: #f4f6f8; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>LiDAR BEV added-label review</h1>
  <div class="note">
    Rendered {len(rows)} frames. Open one page at a time; each page lazy-loads PNGs and includes full added-label JSON per frame.
  </div>
  <ul>
    {link_items}
  </ul>
</body>
</html>
"""


def render_page_nav(
    links: list[tuple[str, int, int, Any, Any]], current: int, total: int
) -> str:
    prev_link = f'<a href="page_{current - 1:03d}.html">prev</a>' if current > 1 else "prev"
    next_link = f'<a href="page_{current + 1:03d}.html">next</a>' if current < total else "next"
    return (
        '<div class="pager">'
        '<a href="index.html">all pages</a> '
        f'| {prev_link} | page {current:03d}/{total:03d} | {next_link}'
        "</div>"
    )


def render_index_html(
    rows: list[dict[str, Any]],
    output_root: Path,
    *,
    title: str,
    note: str,
    nav_html: str,
) -> str:
    body = "\n".join(render_index_row(row, output_root) for row in rows)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 22px; color: #202124; }}
    h1 {{ font-size: 22px; margin-bottom: 6px; }}
    .note {{ color: #555; margin-bottom: 18px; line-height: 1.4; }}
    .pager {{ position: sticky; top: 0; background: rgba(255, 255, 255, 0.96); padding: 8px 0; border-bottom: 1px solid #d9dde2; z-index: 2; }}
    .frame {{ border: 1px solid #d9dde2; border-radius: 6px; margin: 18px 0; padding: 14px; }}
    .meta {{ font-size: 13px; color: #444; margin-bottom: 10px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ccd2d8; background: #111; }}
    code {{ background: #f4f6f8; padding: 1px 4px; border-radius: 3px; }}
    details {{ margin-top: 10px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f7f9fb; padding: 12px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="note">
    {html.escape(note)}
  </div>
  {nav_html}
  {body}
</body>
</html>
"""


def render_index_row(row: dict[str, Any], output_root: Path) -> str:
    image_path = Path(row["image"])
    try:
        rel_image = image_path.relative_to(output_root)
    except ValueError:
        rel_image = image_path
    added_json = json.dumps(row.get("added_labels"), ensure_ascii=False, indent=2)
    reasons = ", ".join(str(item) for item in row.get("selection_reasons") or [])
    return f"""
<div class="frame">
  <div class="meta">
    <b>{html.escape(str(row.get("split")))}</b>
    <code>{html.escape(str(row.get("timestamp_str")))}</code>
    sample=<code>{html.escape(str(row.get("sample_token")))}</code><br/>
    scene=<code>{html.escape(str(row.get("scene_token")))}</code><br/>
    dispatch=<code>{html.escape(str(row.get("dispatch_action")))}</code>
    targets={html.escape(str(row.get("dispatch_target_count")))}
    operation_phase=<code>{html.escape(str(row.get("operation_phase")))}</code>
    ship_phase=<code>{html.escape(str(row.get("ship_operation_phase")))}</code>
    score={html.escape(str(row.get("score")))}
    reasons={html.escape(reasons)}
  </div>
  <img loading="lazy" src="{html.escape(str(rel_image))}" />
  <details>
    <summary>Full added-label JSON</summary>
    <pre>{html.escape(added_json)}</pre>
  </details>
</div>
"""


if __name__ == "__main__":
    main()
