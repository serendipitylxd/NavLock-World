#!/usr/bin/env python3
"""Evaluate the planner head on fused/deployable world-state inputs."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from tools.build_valid_action_labels import (
    DEFAULT_WATER_TOLERANCE_M,
    action_violation_reasons,
    entry_exit_sides,
)
from tools.train_action_planner_head import (
    PLANNER_ACTIONS,
    evaluate_model,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-candidates",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/action_conditioned_valtest_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--planner-model",
        type=Path,
        default=Path("outputs/action_conditioned_world_model/action_planner_head.pkl"),
    )
    parser.add_argument(
        "--fused-predictions",
        type=Path,
        default=Path(
            "outputs/fused_deployable_baseline/predictions_valtest_fused_legacy_current_gt.jsonl"
        ),
    )
    parser.add_argument(
        "--fused-world-state",
        type=Path,
        default=Path("outputs/fused_deployable_baseline/derived_valtest_legacy_current_gt.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/action_conditioned_world_model"),
    )
    parser.add_argument("--tag", default="valtest_fused_scene100")
    parser.add_argument("--no-hard-mask", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = read_jsonl(args.action_candidates)
    with args.planner_model.open("rb") as handle:
        model = pickle.load(handle)

    fused_by_scene = load_fused_bundle(args.fused_predictions, args.fused_world_state)
    selected_samples = {
        item["sample_token"]
        for item in fused_by_scene.values()
        if item.get("sample_token") is not None
    }
    gt_rows = [copy.deepcopy(row) for row in all_rows if row.get("sample_token") in selected_samples]
    if not gt_rows:
        raise SystemExit("no action candidates matched fused world-state sample tokens")

    fused_rows, replacement_report = replace_rows_with_fused_state(
        all_rows,
        fused_by_scene=fused_by_scene,
        recompute_mask=False,
    )
    fused_eval_rows = [
        copy.deepcopy(row)
        for row in fused_rows
        if row.get("sample_token") in selected_samples
    ]
    deployable_mask_rows, deployable_mask_report = replace_rows_with_fused_state(
        all_rows,
        fused_by_scene=fused_by_scene,
        recompute_mask=True,
    )
    deployable_mask_eval_rows = [
        copy.deepcopy(row)
        for row in deployable_mask_rows
        if row.get("sample_token") in selected_samples
    ]

    hard_mask = not args.no_hard_mask
    gt_eval = evaluate_model(
        model,
        gt_rows,
        hard_mask=hard_mask,
        include_observed_action_features=bool(model.get("include_observed_action_features")),
        dispatch_continuity_override=True,
        history_source_rows=all_rows,
    )
    fused_eval = evaluate_model(
        model,
        fused_eval_rows,
        hard_mask=hard_mask,
        include_observed_action_features=bool(model.get("include_observed_action_features")),
        dispatch_continuity_override=True,
        history_source_rows=fused_rows,
    )
    deployable_mask_eval = evaluate_model(
        model,
        deployable_mask_eval_rows,
        hard_mask=hard_mask,
        include_observed_action_features=bool(model.get("include_observed_action_features")),
        dispatch_continuity_override=True,
        history_source_rows=deployable_mask_rows,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    gt_prediction_output = (
        args.output_root / f"predictions_{args.tag}_gt_input_planner.jsonl"
    )
    fused_prediction_output = (
        args.output_root / f"predictions_{args.tag}_fused_feature_swap_planner.jsonl"
    )
    deployable_prediction_output = (
        args.output_root / f"predictions_{args.tag}_deployable_mask_planner.jsonl"
    )
    summary_output = args.output_root / f"summary_{args.tag}_planner.json"
    candidate_output = (
        args.output_root / f"action_conditioned_{args.tag}_deployable_mask_candidates.jsonl"
    )

    write_jsonl(gt_prediction_output, gt_eval["predictions"])
    write_jsonl(fused_prediction_output, fused_eval["predictions"])
    write_jsonl(deployable_prediction_output, deployable_mask_eval["predictions"])
    write_jsonl(candidate_output, deployable_mask_eval_rows)

    summary = {
        "action_candidates": str(args.action_candidates),
        "planner_model": str(args.planner_model),
        "fused_predictions": str(args.fused_predictions),
        "fused_world_state": str(args.fused_world_state),
        "hard_mask": hard_mask,
        "tag": args.tag,
        "num_all_candidate_rows": len(all_rows),
        "num_fused_scenes": len(fused_by_scene),
        "num_matched_samples": len(selected_samples),
        "num_eval_candidate_rows": len(gt_rows),
        "num_eval_frames": len({row.get("sample_token") for row in gt_rows}),
        "outputs": {
            "summary": str(summary_output),
            "gt_input_predictions": str(gt_prediction_output),
            "fused_feature_swap_predictions": str(fused_prediction_output),
            "deployable_mask_predictions": str(deployable_prediction_output),
            "deployable_mask_candidates": str(candidate_output),
        },
        "evaluation_modes": {
            "gt_input_scene100": {
                "description": "Same 100 fused scene samples, original structured current_state and original candidate legality mask.",
                "summary": gt_eval["summary"],
            },
            "fused_feature_swap_original_mask": {
                "description": (
                    "Gate/water/occupancy/ship features are replaced by fused/deployable "
                    "outputs, while the original candidate legality mask is kept. This "
                    "isolates planner feature sensitivity to fused world-state inputs."
                ),
                "replacement_report": replacement_report,
                "summary": fused_eval["summary"],
            },
            "fused_state_deployable_mask": {
                "description": (
                    "The same fused/deployable feature swap, plus candidate legality "
                    "recomputed from the deployable current state. observed_action is "
                    "not used for recomputed water-action legality, so this is a stricter "
                    "deployment-style mask without operator-command labels."
                ),
                "replacement_report": deployable_mask_report,
                "summary": deployable_mask_eval["summary"],
            },
        },
        "limitations": [
            "The fused baseline artifact covers 100 official val/test scene samples, not all 1787 val/test frames.",
            "History features use the full action-candidate timeline; only the fused scene samples have deployable current-state replacement.",
            "Upstream/downstream side water levels remain from the existing side-water proxy because the fused artifact only stores chamber water_level.",
            "Path-clear and queue features are approximated from fused gate-zone occupancy and fused ship motion when recomputing deployable masks.",
        ],
    }
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote_summary={summary_output}")
    print(f"num_eval_frames={summary['num_eval_frames']}")
    for name, result in summary["evaluation_modes"].items():
        frame = result["summary"]["frame_action_head"]
        print(
            f"{name}: legal={frame['legal_rate']:.3f} "
            f"target={frame['target_set_accuracy']:.3f} "
            f"correct={frame['target_set_hit_count']}/{frame['num_frames']}"
        )


def load_fused_bundle(
    prediction_path: Path, world_state_path: Path
) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(prediction_path):
        scene_token = (row.get("metadata") or {}).get("scene_token")
        if scene_token:
            predictions[str(scene_token)] = row

    bundle: dict[str, dict[str, Any]] = {}
    for world_row in read_jsonl(world_state_path):
        scene_token = str(world_row.get("scene_token"))
        pred_row = predictions.get(scene_token)
        if not pred_row:
            continue
        bundle[scene_token] = {
            "scene_token": scene_token,
            "sample_token": world_row.get("sample_token"),
            "prediction_row": pred_row,
            "world_state_row": world_row,
        }
    return bundle


def replace_rows_with_fused_state(
    rows: list[dict[str, Any]],
    *,
    fused_by_scene: dict[str, dict[str, Any]],
    recompute_mask: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    report = {
        "recompute_mask": recompute_mask,
        "replaced_candidate_rows": 0,
        "replaced_samples": set(),
        "ship_phase_counts": Counter(),
        "valid_action_counts_after_recompute": Counter(),
    }
    for row in rows:
        new_row = copy.deepcopy(row)
        fused = fused_by_scene.get(str(row.get("scene_token")))
        if fused and row.get("sample_token") == fused.get("sample_token"):
            new_row["current_state"] = fused_current_state(
                row.get("current_state") or {},
                fused=fused,
                direction=str(row.get("direction") or "unknown"),
            )
            report["replaced_candidate_rows"] += 1
            report["replaced_samples"].add(str(row.get("sample_token")))
            report["ship_phase_counts"][new_row["current_state"].get("ship_operation_phase")] += 1
        output.append(new_row)

    if recompute_mask:
        recompute_candidate_masks(output)
        for row in output:
            if row.get("sample_token") in report["replaced_samples"] and row.get("is_valid"):
                report["valid_action_counts_after_recompute"][row.get("candidate_action")] += 1

    report["replaced_samples"] = len(report["replaced_samples"])
    report["ship_phase_counts"] = dict(report["ship_phase_counts"])
    report["valid_action_counts_after_recompute"] = dict(
        report["valid_action_counts_after_recompute"]
    )
    return output, report


def fused_current_state(
    original: dict[str, Any],
    *,
    fused: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    state = copy.deepcopy(original)
    pred_json = (fused["prediction_row"].get("prediction_json") or {})
    world_row = fused["world_state_row"]
    current = pred_json.get("current_state") or {}
    for key in ("upper_gate_state", "lower_gate_state", "water_state", "water_level"):
        if current.get(key) is not None:
            state[key] = current.get(key)

    occupancy = (
        (pred_json.get("lock_occupancy") or {}).get("current")
        or (world_row.get("lock_occupancy") or {}).get("current")
        or {}
    )
    apply_occupancy_state(state, occupancy, direction=direction)

    motion_flow = pred_json.get("vessel_motion_flow") or world_row.get(
        "vessel_motion_flow"
    ) or {}
    ship_behavior = pred_json.get("ship_behavior") or {}
    apply_ship_phase_state(
        state,
        motion_items=motion_flow.get("input_window") or [],
        ship_intentions=(ship_behavior.get("ship_intentions") or []),
        direction=direction,
    )
    state["operation_phase"] = infer_operation_phase(state)
    return state


def apply_occupancy_state(
    state: dict[str, Any], occupancy: dict[str, Any], *, direction: str
) -> None:
    berth_slots = occupancy.get("berth_slots") or []
    occupied_slots = [
        str(slot.get("region_id") or slot.get("slot_id"))
        for slot in berth_slots
        if slot.get("occupied")
    ]
    available_slots = [
        str(slot.get("region_id") or slot.get("slot_id"))
        for slot in berth_slots
        if not slot.get("occupied")
    ]
    state["occupied_berth_slots"] = [slot for slot in occupied_slots if slot]
    state["available_berth_slots"] = [slot for slot in available_slots if slot]
    state["num_occupied_berths"] = len(state["occupied_berth_slots"])
    state["chamber_capacity_available"] = bool(state["available_berth_slots"])

    coarse_counts = {
        str(item.get("region_id")): int(item.get("ship_count") or 0)
        for item in occupancy.get("coarse_regions") or []
    }
    state["no_ship_in_upper_gate_zone"] = coarse_counts.get("upper_gate_zone", 0) == 0
    state["no_ship_in_lower_gate_zone"] = coarse_counts.get("lower_gate_zone", 0) == 0
    chamber_count = max(
        int(occupancy.get("num_ships") or 0),
        coarse_counts.get("upper_gate_zone", 0)
        + coarse_counts.get("lower_gate_zone", 0)
        + coarse_counts.get("between_berths", 0),
        len(state["occupied_berth_slots"]),
    )
    state["num_ships_in_chamber"] = chamber_count

    entry_side, exit_side = entry_exit_sides(direction)
    if entry_side:
        state["entry_path_clear"] = coarse_counts.get(f"{entry_side}_gate_zone", 0) == 0
    if exit_side:
        state["exit_path_clear"] = coarse_counts.get(f"{exit_side}_gate_zone", 0) == 0


def apply_ship_phase_state(
    state: dict[str, Any],
    *,
    motion_items: list[dict[str, Any]],
    ship_intentions: list[dict[str, Any]],
    direction: str,
) -> None:
    motion_counts = Counter(str(item.get("motion_state")) for item in motion_items)
    intent_counts = Counter()
    for item in ship_intentions:
        for intent in item.get("ship_intentions") or []:
            intent_counts[str(intent)] += 1

    entering = motion_counts["ship_entering_lock"] + intent_counts["ship_entering_lock"]
    leaving = motion_counts["ship_leaving_lock"] + intent_counts["ship_leaving_lock"]
    moving = motion_counts["ship_moving"]
    static_like = (
        motion_counts["ship_berthed"]
        + motion_counts["ship_static"]
        + motion_counts["object_static"]
        + intent_counts["ship_berthed"]
        + intent_counts["object_static"]
    )
    num_ships = int(state.get("num_ships_in_chamber") or 0)
    if leaving:
        phase = "ship_leaving"
    elif entering:
        phase = "ship_entering"
    elif num_ships == 0:
        phase = "lock_clear"
    elif static_like and not moving:
        phase = "all_ships_berthed"
    else:
        phase = "ship_phase_uncertain"

    state["ship_operation_phase"] = phase
    state["all_in_chamber_ships_berthed_or_static"] = phase in {
        "all_ships_berthed",
        "lock_clear",
    }
    state["no_ship_entering_or_leaving_inside_chamber"] = phase not in {
        "ship_entering",
        "ship_leaving",
        "ship_phase_uncertain",
    }

    entry_side, exit_side = entry_exit_sides(direction)
    next_enter = first_ship_payload(
        motion_items, ship_intentions, target="ship_entering_lock", side=entry_side
    )
    next_leave = first_ship_payload(
        motion_items, ship_intentions, target="ship_leaving_lock", side=exit_side
    )
    state["next_ship_to_enter_weak"] = next_enter
    state["next_ship_to_leave_weak"] = next_leave
    state["queue_rank"] = [next_enter] if next_enter else []
    state["max_parallel_entries"] = min(
        2,
        len(state.get("available_berth_slots") or []),
        1 if next_enter else 0,
    )
    state["max_parallel_departures"] = min(2, 1 if next_leave else 0)


def first_ship_payload(
    motion_items: list[dict[str, Any]],
    ship_intentions: list[dict[str, Any]],
    *,
    target: str,
    side: Optional[str],
) -> Optional[dict[str, Any]]:
    for item in motion_items:
        if item.get("motion_state") == target:
            return {
                "instance_token": item.get("instance_token"),
                "category": item.get("category"),
                "rank": 1,
                "side": side or "unknown",
                "source": "fused_vessel_motion_flow",
                "distance_to_gate_m": 0.0,
                "speed_mps": item.get("end_speed_mps") or 0.0,
            }
    for item in ship_intentions:
        if target in set(item.get("ship_intentions") or []):
            return {
                "instance_token": item.get("instance_token"),
                "category": item.get("category"),
                "rank": 1,
                "side": side or "unknown",
                "source": "fused_ship_behavior",
                "distance_to_gate_m": 0.0,
                "speed_mps": 0.0,
            }
    return None


def infer_operation_phase(state: dict[str, Any]) -> str:
    water_state = state.get("water_state")
    if water_state in {"filling", "emptying"}:
        return str(water_state)
    upper = state.get("upper_gate_state")
    lower = state.get("lower_gate_state")
    if upper in {"opening", "closing"}:
        return f"upper_gate_{upper}"
    if lower in {"opening", "closing"}:
        return f"lower_gate_{lower}"
    if upper == "open" and lower != "open":
        return "upper_gate_open_idle"
    if lower == "open" and upper != "open":
        return "lower_gate_open_idle"
    if upper == "closed" and lower == "closed":
        return "chamber_closed_idle"
    return "operation_phase_uncertain"


def recompute_candidate_masks(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_token"))].append(row)

    for sample_rows in grouped.values():
        representative = sample_rows[0]
        state = representative.get("current_state") or {}
        valid_row = dict(state)
        valid_row["lock_water_state"] = state.get("water_state")
        valid_row["observed_action"] = None
        direction = str(representative.get("direction") or "unknown")
        reasons_by_action = {
            action: action_violation_reasons(
                action,
                valid_row,
                direction=direction,
                water_tolerance_m=DEFAULT_WATER_TOLERANCE_M,
            )
            for action in PLANNER_ACTIONS
        }
        valid_actions = [
            action for action in PLANNER_ACTIONS if not reasons_by_action[action]
        ]
        invalid_actions = [
            action for action in PLANNER_ACTIONS if reasons_by_action[action]
        ]
        state["valid_actions"] = valid_actions
        state["invalid_actions"] = invalid_actions
        state["violation_reason"] = {
            action: reasons_by_action[action]
            for action in invalid_actions
        }
        for row in sample_rows:
            action = row["candidate_action"]
            row["current_state"] = state
            row["is_valid"] = action in set(valid_actions)
            row["violation_reason"] = reasons_by_action[action]


if __name__ == "__main__":
    main()
