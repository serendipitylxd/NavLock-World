#!/usr/bin/env python3
"""Build action-conditioned NavLock frame/candidate datasets from sequences."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional


PLANNER_ACTIONS = (
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
STATE_FIELDS = (
    "upper_gate_state",
    "lower_gate_state",
    "water_state",
    "water_level",
    "upstream_water_level",
    "downstream_water_level",
    "observed_action",
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
    "ship_dispatch_action",
    "ship_dispatch_targets",
    "ship_dispatch_target_count",
    "ship_dispatch_source",
    "ship_dispatch_confidence",
    "ship_dispatch_conflict",
)
SHIP_FIELDS = (
    "instance_token",
    "annotation_token",
    "category",
    "translation",
    "velocity",
    "ship_intentions",
    "attribute_names",
    "assigned_berth_slot",
    "occlusion_state",
    "visibility_level",
)
HORIZONS = (10, 20, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated splits, e.g. train or val,test.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/action_conditioned_world_model"),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Output tag. Defaults to split names joined without commas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    if not splits:
        raise SystemExit("--splits must contain at least one split")
    tag = args.tag or "".join(splits)
    sequences = load_sequences(args.data_root, splits)
    frame_rows = build_frame_rows(sequences)
    candidate_rows = build_candidate_rows(frame_rows)

    args.output_root.mkdir(parents=True, exist_ok=True)
    frame_output = args.output_root / f"action_conditioned_{tag}_frames.jsonl"
    candidate_output = args.output_root / f"action_conditioned_{tag}_candidates.jsonl"
    summary_output = args.output_root / f"summary_{tag}.json"
    write_jsonl(frame_output, frame_rows)
    write_jsonl(candidate_output, candidate_rows)
    summary = build_summary(
        frame_rows,
        candidate_rows,
        splits=splits,
        frame_output=frame_output,
        candidate_output=candidate_output,
    )
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote={frame_output}")
    print(f"wrote={candidate_output}")
    print(f"wrote={summary_output}")
    print(f"num_frames={len(frame_rows)}")
    print(f"num_candidates={len(candidate_rows)}")
    print(f"observed_planner_action_counts={summary['observed_planner_action_counts']}")
    print(f"future_coverage={summary['future_coverage']}")


def load_sequences(data_root: Path, splits: list[str]) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for split in splits:
        path = data_root / "navlock_sequences" / f"scene_sequences_{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sequence in payload.get("sequences") or []:
            item = dict(sequence)
            item["split"] = split
            sequences.append(item)
    return sequences


def build_frame_rows(sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        for frame in sequence.get("frames") or []:
            rows.append(build_frame_row(sequence, frame))
    return rows


def build_frame_row(sequence: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    state = frame.get("lock_state") or {}
    observed_action = str(state.get("observed_action") or "hold")
    ship_dispatch_action = str(state.get("ship_dispatch_action") or "hold")
    observed_planner_actions = combined_observed_planner_actions(
        observed_action, ship_dispatch_action
    )
    rule_consistent_planner_actions = rule_consistent_actions(
        observed_planner_actions, state.get("valid_actions") or []
    )
    current_state = {field: state.get(field) for field in STATE_FIELDS if field in state}
    future_targets = build_future_targets(state)
    row = {
        "row_id": frame.get("sample_token"),
        "split": sequence.get("split"),
        "sample_token": frame.get("sample_token"),
        "scene_token": sequence.get("scene_token"),
        "scene_name": sequence.get("scene_name"),
        "timestamp": frame.get("timestamp"),
        "timestamp_str": frame.get("timestamp_str"),
        "frame_index": frame.get("frame_index"),
        "relative_time_sec": frame.get("relative_time_sec"),
        "direction": sequence.get("direction"),
        "operation_date": sequence.get("operation_date"),
        "operation_index": sequence.get("operation_index"),
        "line_index": sequence.get("line_index"),
        "segment_index": sequence.get("segment_index"),
        "sensor": sensor_payload(frame),
        "conditioning": {
            "observed_action": observed_action,
            "ship_dispatch_action": ship_dispatch_action,
            "observed_planner_actions": observed_planner_actions,
            "primary_observed_planner_action": observed_planner_actions[0],
            "rule_consistent_planner_actions": rule_consistent_planner_actions,
            "primary_rule_consistent_planner_action": rule_consistent_planner_actions[0],
            "future_gate_water_conditioning_action": observed_action,
        },
        "current_state": current_state,
        "candidate_actions": build_candidate_actions(
            state,
            observed_action=observed_action,
            ship_dispatch_action=ship_dispatch_action,
            observed_planner_actions=observed_planner_actions,
            rule_consistent_planner_actions=rule_consistent_planner_actions,
        ),
        "future_targets": future_targets,
        "ship_context": [
            {field: inst.get(field) for field in SHIP_FIELDS if field in inst}
            for inst in frame.get("instances_3d") or []
            if is_vessel_instance(inst)
        ],
    }
    return row


def combined_observed_planner_actions(
    observed_action: str, ship_dispatch_action: str
) -> list[str]:
    actions: list[str] = []
    if observed_action and observed_action != "hold":
        actions.append(observed_action)
    if ship_dispatch_action and ship_dispatch_action != "hold":
        actions.append(ship_dispatch_action)
    return actions or ["hold"]


def rule_consistent_actions(actions: list[str], valid_actions: list[str]) -> list[str]:
    valid = set(valid_actions)
    out = [action for action in actions if action in valid]
    if out:
        return out
    return ["hold"] if "hold" in valid or not valid else [sorted(valid)[0]]


def sensor_payload(frame: dict[str, Any]) -> dict[str, Any]:
    lidar = frame.get("lidar") or {}
    images = frame.get("images") or {}
    if isinstance(images, dict):
        image_files = {
            channel: image.get("file_name") or image.get("path")
            for channel, image in images.items()
            if isinstance(image, dict)
        }
    elif isinstance(images, list):
        image_files = {
            str(image.get("channel") or index): image.get("file_name") or image.get("path")
            for index, image in enumerate(images)
            if isinstance(image, dict)
        }
    else:
        image_files = {}
    return {
        "lidar_file": lidar.get("file_name") or lidar.get("path"),
        "lidar_sample_data_token": lidar.get("sample_data_token"),
        "image_files": image_files,
    }


def build_candidate_actions(
    state: dict[str, Any],
    *,
    observed_action: str,
    ship_dispatch_action: str,
    observed_planner_actions: list[str],
    rule_consistent_planner_actions: list[str],
) -> list[dict[str, Any]]:
    valid_actions = set(state.get("valid_actions") or [])
    violation_reason = state.get("violation_reason") or {}
    candidates = []
    for action in PLANNER_ACTIONS:
        candidates.append(
            {
                "action": action,
                "is_valid": action in valid_actions,
                "violation_reason": violation_reason.get(action, []),
                "is_observed_gate_water_action": action == observed_action,
                "is_observed_ship_dispatch_action": action == ship_dispatch_action,
                "is_observed_planner_action": action in observed_planner_actions,
                "is_rule_consistent_planner_action": action
                in rule_consistent_planner_actions,
                "future_gate_water_target_available": action == observed_action,
            }
        )
    return candidates


def build_future_targets(state: dict[str, Any]) -> dict[str, Any]:
    targets: dict[str, Any] = {
        "future_state_after_observed_action": state.get(
            "future_state_after_observed_action"
        ),
        "future_phase_after_observed_action": state.get(
            "future_phase_after_observed_action"
        ),
        "horizons": {},
    }
    for horizon in HORIZONS:
        targets["horizons"][f"t_plus_{horizon}s"] = {
            "state": state.get(f"state_t_plus_{horizon}s"),
            "phase": state.get(f"phase_t_plus_{horizon}s"),
        }
    return targets


def build_candidate_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in frame_rows:
        for candidate in frame["candidate_actions"]:
            future_targets = (
                frame["future_targets"]
                if candidate["future_gate_water_target_available"]
                else None
            )
            rows.append(
                {
                    "row_id": f"{frame['sample_token']}::{candidate['action']}",
                    "split": frame["split"],
                    "sample_token": frame["sample_token"],
                    "scene_token": frame["scene_token"],
                    "timestamp": frame["timestamp"],
                    "timestamp_str": frame["timestamp_str"],
                    "direction": frame["direction"],
                    "current_state": frame["current_state"],
                    "candidate_action": candidate["action"],
                    "is_valid": candidate["is_valid"],
                    "violation_reason": candidate["violation_reason"],
                    "is_observed_planner_action": candidate["is_observed_planner_action"],
                    "is_rule_consistent_planner_action": candidate[
                        "is_rule_consistent_planner_action"
                    ],
                    "is_observed_gate_water_action": candidate[
                        "is_observed_gate_water_action"
                    ],
                    "is_observed_ship_dispatch_action": candidate[
                        "is_observed_ship_dispatch_action"
                    ],
                    "future_gate_water_target_available": candidate[
                        "future_gate_water_target_available"
                    ],
                    "future_targets": future_targets,
                }
            )
    return rows


def is_vessel_instance(item: dict[str, Any]) -> bool:
    category = str(item.get("category") or "").lower()
    if any(marker in category for marker in ("ship", "fleet", "vessel", "tugboat")):
        return True
    if item.get("ship_intentions"):
        return True
    if item.get("assigned_berth_slot"):
        return True
    return False


def build_summary(
    frame_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    splits: list[str],
    frame_output: Path,
    candidate_output: Path,
) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in frame_rows)
    observed_action_counts = Counter(
        row["conditioning"]["observed_action"] for row in frame_rows
    )
    ship_dispatch_counts = Counter(
        row["conditioning"]["ship_dispatch_action"] for row in frame_rows
    )
    planner_counts = Counter(
        row["conditioning"]["primary_observed_planner_action"] for row in frame_rows
    )
    rule_consistent_counts = Counter(
        row["conditioning"]["primary_rule_consistent_planner_action"]
        for row in frame_rows
    )
    valid_candidate_counts = Counter(
        row["candidate_action"] for row in candidate_rows if row["is_valid"]
    )
    observed_planner_valid = sum(
        any(
            candidate["is_observed_planner_action"] and candidate["is_valid"]
            for candidate in row["candidate_actions"]
        )
        for row in frame_rows
    )
    future_coverage = {}
    for horizon in HORIZONS:
        key = f"t_plus_{horizon}s"
        future_coverage[key] = sum(
            1
            for row in frame_rows
            if row["future_targets"]["horizons"][key]["state"] is not None
        )
    return {
        "splits": splits,
        "num_frames": len(frame_rows),
        "num_candidates": len(candidate_rows),
        "split_counts": dict(split_counts),
        "frame_output": str(frame_output),
        "candidate_output": str(candidate_output),
        "planner_actions": list(PLANNER_ACTIONS),
        "observed_action_counts": dict(observed_action_counts),
        "ship_dispatch_action_counts": dict(ship_dispatch_counts),
        "observed_planner_action_counts": dict(planner_counts),
        "rule_consistent_planner_action_counts": dict(rule_consistent_counts),
        "valid_candidate_action_counts": dict(valid_candidate_counts),
        "observed_planner_action_valid_count": observed_planner_valid,
        "observed_planner_action_valid_rate": safe_div(
            observed_planner_valid, len(frame_rows)
        ),
        "future_coverage": future_coverage,
        "notes": [
            "observed_action covers gate/water operation only",
            "ship_dispatch_action is stored separately and may coexist with observed_action=hold",
            "future gate/water targets are available only for the observed_action conditioning branch",
            "non-observed candidate actions have legality labels but no counterfactual future target",
            "rule_consistent_planner_actions filters ongoing observed/dispatch states through the current valid action mask for planner training",
        ],
    }


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
