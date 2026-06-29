#!/usr/bin/env python3
"""Build rule-constrained valid/invalid action labels for NavLock frames."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Optional


ACTION_SET = (
    "hold",
    "open_upper_gate",
    "close_upper_gate",
    "open_lower_gate",
    "close_lower_gate",
    "start_filling",
    "start_emptying",
    "stop_filling_emptying",
    "dispatch_enter",
    "dispatch_exit",
)
LABEL_FIELDS = ("valid_actions", "invalid_actions", "violation_reason")
DEFAULT_WATER_TOLERANCE_M = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--water-tolerance-m", type=float, default=DEFAULT_WATER_TOLERANCE_M
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/valid_action_labels/summary.json"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("outputs/valid_action_labels/valid_actions.jsonl"),
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
    direction_by_scene = load_scene_directions(version_root)
    labels = build_valid_action_labels(
        rows,
        direction_by_scene=direction_by_scene,
        water_tolerance_m=args.water_tolerance_m,
    )

    if not args.no_update_sample:
        update_sample_rows(rows, labels)
        sample_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    pkl_report = []
    if not args.no_update_pkl:
        pkl_report = update_info_pkls(args.data_root, labels)

    write_jsonl(args.jsonl_output, rows, labels)
    summary = build_summary(
        rows,
        labels=labels,
        sample_path=sample_path,
        water_tolerance_m=args.water_tolerance_m,
        pkl_report=pkl_report,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote={args.summary_output}")
    print(f"wrote={args.jsonl_output}")
    print(f"num_samples={len(rows)}")
    print(f"valid_action_counts={summary['valid_action_counts']}")
    print(f"invalid_action_counts={summary['invalid_action_counts']}")
    print(f"updated_sample={not args.no_update_sample}")
    print(f"updated_pkl={not args.no_update_pkl}")


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


def build_valid_action_labels(
    rows: list[dict[str, Any]],
    *,
    direction_by_scene: dict[str, str],
    water_tolerance_m: float = DEFAULT_WATER_TOLERANCE_M,
) -> dict[str, dict[str, Any]]:
    labels = {}
    for row in rows:
        reasons = {
            action: action_violation_reasons(
                action,
                row,
                direction=direction_by_scene.get(row.get("scene_token"), "unknown"),
                water_tolerance_m=water_tolerance_m,
            )
            for action in ACTION_SET
        }
        valid_actions = [action for action in ACTION_SET if not reasons[action]]
        invalid_actions = [action for action in ACTION_SET if reasons[action]]
        labels[row["token"]] = {
            "valid_actions": valid_actions,
            "invalid_actions": invalid_actions,
            "violation_reason": {
                action: reasons[action]
                for action in invalid_actions
            },
        }
    return labels


def action_violation_reasons(
    action: str,
    row: dict[str, Any],
    *,
    direction: str,
    water_tolerance_m: float,
) -> list[str]:
    if action == "hold":
        return []
    if action == "open_upper_gate":
        return open_gate_reasons(row, side="upper", water_tolerance_m=water_tolerance_m)
    if action == "open_lower_gate":
        return open_gate_reasons(row, side="lower", water_tolerance_m=water_tolerance_m)
    if action == "close_upper_gate":
        return close_gate_reasons(row, side="upper")
    if action == "close_lower_gate":
        return close_gate_reasons(row, side="lower")
    if action == "start_filling":
        return start_water_reasons(
            row, action="start_filling", water_tolerance_m=water_tolerance_m
        )
    if action == "start_emptying":
        return start_water_reasons(
            row, action="start_emptying", water_tolerance_m=water_tolerance_m
        )
    if action == "stop_filling_emptying":
        return stop_water_reasons(row)
    if action == "dispatch_enter":
        return dispatch_reasons(row, direction=direction, kind="enter")
    if action == "dispatch_exit":
        return dispatch_reasons(row, direction=direction, kind="exit")
    raise ValueError(f"unknown action: {action}")


def open_gate_reasons(
    row: dict[str, Any], *, side: str, water_tolerance_m: float
) -> list[str]:
    reasons = []
    gate_key = f"{side}_gate_state"
    other_gate_key = "lower_gate_state" if side == "upper" else "upper_gate_state"
    side_level_key = "upstream_water_level" if side == "upper" else "downstream_water_level"
    clear_key = f"no_ship_in_{side}_gate_zone"

    if row.get(gate_key) != "closed":
        reasons.append(f"{side}_gate_not_closed")
    if row.get(other_gate_key) != "closed":
        reasons.append(f"other_gate_not_closed")
    if row.get("lock_water_state") != "idle":
        reasons.append("water_state_not_idle")
    water_diff = water_level_diff(row.get("water_level"), row.get(side_level_key))
    if water_diff is None:
        reasons.append(f"{side}_water_level_unavailable")
    elif water_diff > water_tolerance_m:
        reasons.append(f"{side}_water_level_not_equal")
    if row.get(clear_key) is not True:
        reasons.append(f"ship_in_{side}_gate_zone")
    return reasons


def close_gate_reasons(row: dict[str, Any], *, side: str) -> list[str]:
    reasons = []
    gate_key = f"{side}_gate_state"
    clear_key = f"no_ship_in_{side}_gate_zone"
    if row.get(gate_key) != "open":
        reasons.append(f"{side}_gate_not_open")
    if row.get(clear_key) is not True:
        reasons.append(f"ship_in_{side}_gate_zone")
    return reasons


def start_water_reasons(
    row: dict[str, Any], *, action: str, water_tolerance_m: float
) -> list[str]:
    del water_tolerance_m
    reasons = []
    target_state = "filling" if action == "start_filling" else "emptying"
    if row.get("upper_gate_state") != "closed":
        reasons.append("upper_gate_not_closed")
    if row.get("lower_gate_state") != "closed":
        reasons.append("lower_gate_not_closed")
    if not annotated_water_action_matches(row, action=action, target_state=target_state):
        reasons.append(f"{action}_not_annotated")
    if row.get("lock_water_state") not in {"idle", target_state}:
        reasons.append("water_state_conflicts_with_action")
    if row.get("entry_path_clear") is not True:
        reasons.append("entry_path_not_clear")
    if row.get("exit_path_clear") is not True:
        reasons.append("exit_path_not_clear")
    if row.get("all_in_chamber_ships_berthed_or_static") is not True:
        reasons.append("not_all_in_chamber_ships_berthed_or_static")
    if row.get("no_ship_entering_or_leaving_inside_chamber") is not True:
        reasons.append("ship_entering_or_leaving_inside_chamber")
    return reasons


def annotated_water_action_matches(
    row: dict[str, Any], *, action: str, target_state: str
) -> bool:
    if row.get("observed_action") == action:
        return True
    return row.get("lock_water_state") == target_state


def stop_water_reasons(row: dict[str, Any]) -> list[str]:
    if row.get("lock_water_state") in {"filling", "emptying"}:
        return []
    return ["water_state_not_filling_or_emptying"]


def dispatch_reasons(row: dict[str, Any], *, direction: str, kind: str) -> list[str]:
    reasons = []
    entry_side, exit_side = entry_exit_sides(direction)
    side = entry_side if kind == "enter" else exit_side
    if side is None:
        reasons.append("scene_direction_unknown")
        return reasons
    gate_key = f"{side}_gate_state"
    path_key = "entry_path_clear" if kind == "enter" else "exit_path_clear"
    if row.get(gate_key) != "open":
        reasons.append(f"{kind}_gate_not_open")
    if row.get("lock_water_state") != "idle":
        reasons.append("water_state_not_idle")
    if row.get(path_key) is not True:
        reasons.append(f"{path_key}_false")
    if kind == "enter" and row.get("chamber_capacity_available") is not True:
        reasons.append("chamber_capacity_unavailable")
    return reasons


def entry_exit_sides(direction: str) -> tuple[Optional[str], Optional[str]]:
    if direction == "upstream":
        return "lower", "upper"
    if direction == "downstream":
        return "upper", "lower"
    return None, None


def water_level_diff(a: Any, b: Any) -> Optional[float]:
    if not is_number(a) or not is_number(b):
        return None
    return abs(float(a) - float(b))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


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
                **labels[row["token"]],
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def build_summary(
    rows: list[dict[str, Any]],
    *,
    labels: dict[str, dict[str, Any]],
    sample_path: Path,
    water_tolerance_m: float,
    pkl_report: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    num_valid_counts: Counter[int] = Counter()
    for label in labels.values():
        for action in label["valid_actions"]:
            valid_counts[action] += 1
        for action in label["invalid_actions"]:
            invalid_counts[action] += 1
        for action, reasons in label["violation_reason"].items():
            for reason in reasons:
                reason_counts[f"{action}.{reason}"] += 1
        num_valid_counts[len(label["valid_actions"])] += 1
    return {
        "settings": {
            "sample_path": str(sample_path),
            "water_tolerance_m": water_tolerance_m,
            "source": (
                "rule-constrained labels from gate/water/action annotations, "
                "side-level gate-opening checks, gate-zone/path-clear fields, "
                "and chamber queue/capacity labels"
            ),
            "action_set": list(ACTION_SET),
            "scope_note": (
                "dispatch_enter checks chamber_capacity_available; start_filling "
                "and start_emptying use observed_action/lock_water_state labels "
                "instead of water-level direction inference, and check "
                "all_in_chamber_ships_berthed_or_static plus "
                "no_ship_entering_or_leaving_inside_chamber. Queue priority and "
                "next-ship target remain weak planner priors, not hard constraints"
            ),
        },
        "num_samples": len(rows),
        "valid_action_counts": dict(sorted(valid_counts.items())),
        "invalid_action_counts": dict(sorted(invalid_counts.items())),
        "num_valid_actions_distribution": {
            str(num): count for num, count in sorted(num_valid_counts.items())
        },
        "top_violation_reasons": dict(reason_counts.most_common(50)),
        "pkl_update_report": pkl_report,
    }


def clear_label_fields(row: dict[str, Any]) -> None:
    for key in LABEL_FIELDS:
        row.pop(key, None)


if __name__ == "__main__":
    main()
