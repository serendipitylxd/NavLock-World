#!/usr/bin/env python3
"""Evaluate planner inputs rebuilt from full-frame deployable perception state.

This is the bridge from action-conditioned rows to deployable ship/world-state
features for every val/test frame. Gate/water state remains the existing
telemetry/state field from the action-conditioned rows; ship occupancy, motion,
ship phase, path-clear approximation, and weak enter/leave queues are rebuilt
from Hydro3DNet + optional RTMDet recovery.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from navlock_world.lock_world_state import load_lock_chamber_bounds, load_scene_berths
from tools.analyze_rtmdet_hydro_2d_support import load_rtmdet_ship_boxes
from tools.derive_world_state_from_hydro3dnet_tracks import (
    derive_prediction_from_hydro_tracks,
    load_hydro_predictions,
)
from tools.evaluate_planner_on_fused_world_state import (
    fused_current_state,
    recompute_candidate_masks,
)
from tools.train_action_planner_head import evaluate_model, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits", default="val,test")
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
        "--output-root",
        type=Path,
        default=Path("outputs/action_conditioned_world_model"),
    )
    parser.add_argument("--tag", default="valtest_full_deployable")
    parser.add_argument("--scene-json", type=Path, default=None)
    parser.add_argument(
        "--lock-boundary-map",
        type=Path,
        default=Path("data/maps/huaiyin_lock_boundary.json"),
    )
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--track-distance-m", type=float, default=40.0)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument(
        "--no-rtmdet-recovery",
        action="store_true",
        help="Disable RTMDet multi-camera and open-gate recovery.",
    )
    parser.add_argument("--rtmdet-score-threshold", type=float, default=0.30)
    parser.add_argument("--support-iou-threshold", type=float, default=0.30)
    parser.add_argument("--recovery-min-cameras", type=int, default=4)
    parser.add_argument("--recovery-max-ray-residual-m", type=float, default=10.0)
    parser.add_argument("--recovery-cluster-distance-m", type=float, default=20.0)
    parser.add_argument("--recovery-existing-distance-m", type=float, default=20.0)
    parser.add_argument("--recovery-chamber-margin-m", type=float, default=0.0)
    parser.add_argument("--open-gate-min-cameras", type=int, default=3)
    parser.add_argument("--open-gate-zone-length-m", type=float, default=70.0)
    parser.add_argument("--open-gate-max-candidates", type=int, default=1)
    parser.add_argument(
        "--no-temporal-dispatch-stitch",
        action="store_true",
        help=(
            "Disable temporal continuity for deployable ship_entering/"
            "ship_leaving planner features."
        ),
    )
    parser.add_argument(
        "--dispatch-stitch-max-gap-sec",
        type=float,
        default=35.0,
        help="Maximum gap for carrying enter/exit dispatch phase evidence.",
    )
    parser.add_argument(
        "--dispatch-stitch-min-speed-mps",
        type=float,
        default=0.05,
        help="Minimum next-ship speed treated as active dispatch evidence.",
    )
    parser.add_argument(
        "--disable-berth-motion-prior",
        action="store_true",
        help=(
            "Disable ideal-berth slot matching and berth-settled motion cues when "
            "deriving deployable ship/world-state inputs. This is intended for "
            "module ablation only."
        ),
    )
    parser.add_argument("--no-hard-mask", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    if not splits:
        raise SystemExit("--splits must contain at least one split")

    all_rows = read_jsonl(args.action_candidates)
    with args.planner_model.open("rb") as handle:
        model = pickle.load(handle)

    print("loading sequences/perception inputs")
    sequences = load_sequences(args.data_root, splits)
    scene_json = args.scene_json or (args.data_root / "v1.0-trainval" / "scene.json")
    berths_by_scene = load_scene_berths(scene_json)
    state_berths_by_scene = (
        {scene_token: [] for scene_token in berths_by_scene}
        if args.disable_berth_motion_prior
        else berths_by_scene
    )
    lock_chamber_bounds = load_lock_chamber_bounds(args.lock_boundary_map)
    hydro_predictions = load_hydro_predictions_for_splits(splits)
    rtmdet_by_path = None
    use_rtmdet = not args.no_rtmdet_recovery
    if use_rtmdet:
        rtmdet_by_path = load_rtmdet_for_splits(splits, args.rtmdet_score_threshold)

    print("deriving full-frame deployable world state")
    world_state_by_sample = derive_full_frame_world_state(
        sequences,
        berths_by_scene=state_berths_by_scene,
        hydro_predictions=hydro_predictions,
        data_root=args.data_root,
        lock_chamber_bounds=lock_chamber_bounds,
        rtmdet_by_path=rtmdet_by_path,
        use_rtmdet=use_rtmdet,
        args=args,
    )
    if not args.no_temporal_dispatch_stitch:
        print("applying temporal dispatch stitch")
        stitch_report = apply_temporal_dispatch_stitch(
            world_state_by_sample,
            sequences=sequences,
            max_gap_sec=args.dispatch_stitch_max_gap_sec,
            min_speed_mps=args.dispatch_stitch_min_speed_mps,
        )
    else:
        stitch_report = {"enabled": False}
    deployable_feature_rows, replacement_report = replace_candidate_states(
        all_rows,
        world_state_by_sample=world_state_by_sample,
        recompute_mask=False,
    )
    deployable_mask_rows, deployable_mask_report = replace_candidate_states(
        all_rows,
        world_state_by_sample=world_state_by_sample,
        recompute_mask=True,
    )

    hard_mask = not args.no_hard_mask
    include_observed = bool(model.get("include_observed_action_features"))
    gt_eval = evaluate_model(
        model,
        all_rows,
        hard_mask=hard_mask,
        include_observed_action_features=include_observed,
        dispatch_continuity_override=True,
        history_source_rows=all_rows,
    )
    feature_eval = evaluate_model(
        model,
        deployable_feature_rows,
        hard_mask=hard_mask,
        include_observed_action_features=include_observed,
        dispatch_continuity_override=True,
        history_source_rows=deployable_feature_rows,
    )
    mask_eval = evaluate_model(
        model,
        deployable_mask_rows,
        hard_mask=hard_mask,
        include_observed_action_features=include_observed,
        dispatch_continuity_override=True,
        history_source_rows=deployable_mask_rows,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    world_state_output = args.output_root / f"deployable_world_state_{args.tag}.jsonl"
    candidate_output = args.output_root / f"action_conditioned_{args.tag}_candidates.jsonl"
    gt_prediction_output = args.output_root / f"predictions_{args.tag}_gt_input_planner.jsonl"
    feature_prediction_output = (
        args.output_root / f"predictions_{args.tag}_feature_swap_planner.jsonl"
    )
    mask_prediction_output = (
        args.output_root / f"predictions_{args.tag}_deployable_mask_planner.jsonl"
    )
    summary_output = args.output_root / f"summary_{args.tag}_planner.json"
    write_jsonl(world_state_output, sorted_world_states(world_state_by_sample.values()))
    write_jsonl(candidate_output, deployable_mask_rows)
    write_jsonl(gt_prediction_output, gt_eval["predictions"])
    write_jsonl(feature_prediction_output, feature_eval["predictions"])
    write_jsonl(mask_prediction_output, mask_eval["predictions"])

    summary = {
        "tag": args.tag,
        "splits": splits,
        "action_candidates": str(args.action_candidates),
        "planner_model": str(args.planner_model),
        "num_candidate_rows": len(all_rows),
        "num_frames": len({row.get("sample_token") for row in all_rows}),
        "num_deployable_world_state_frames": len(world_state_by_sample),
        "outputs": {
            "summary": str(summary_output),
            "deployable_world_state": str(world_state_output),
            "deployable_mask_candidates": str(candidate_output),
            "gt_input_predictions": str(gt_prediction_output),
            "feature_swap_predictions": str(feature_prediction_output),
            "deployable_mask_predictions": str(mask_prediction_output),
        },
        "settings": {
            "window_size": args.window_size,
            "score_threshold": args.score_threshold,
            "track_distance_m": args.track_distance_m,
            "use_rtmdet_recovery": use_rtmdet,
            "rtmdet_score_threshold": args.rtmdet_score_threshold if use_rtmdet else None,
            "recovery_min_cameras": args.recovery_min_cameras if use_rtmdet else None,
            "recover_open_gate_new_ships": use_rtmdet,
            "hard_mask": hard_mask,
            "disable_berth_motion_prior": bool(args.disable_berth_motion_prior),
            "temporal_dispatch_stitch": not args.no_temporal_dispatch_stitch,
            "dispatch_stitch_max_gap_sec": args.dispatch_stitch_max_gap_sec,
            "dispatch_stitch_min_speed_mps": args.dispatch_stitch_min_speed_mps,
        },
        "temporal_dispatch_stitch_report": stitch_report,
        "evaluation_modes": {
            "gt_structured_sanity_check": {
                "description": "Original structured action rows. This is only a sanity check, not a deployable planner result.",
                "summary": gt_eval["summary"],
            },
            "deployable_feature_swap_original_mask": {
                "description": "All-frame deployable ship/occupancy/motion feature replacement while keeping the original candidate legality mask.",
                "replacement_report": replacement_report,
                "summary": feature_eval["summary"],
            },
            "deployable_state_recomputed_mask": {
                "description": "All-frame deployable feature replacement plus candidate legality recomputed from deployable current state.",
                "replacement_report": deployable_mask_report,
                "summary": mask_eval["summary"],
            },
        },
        "limitations": [
            "Gate/water state is still the existing telemetry/state field from action-conditioned rows; full-frame VLM gate/water was not generated.",
            "Ship/occupancy/motion/phase/queue/path-clear features are rebuilt from Hydro3DNet + RTMDet recovery.",
            "When disable_berth_motion_prior is true, ideal berth boxes are not used for slot occupancy or berth-settled motion labels.",
            "Target actions remain the action-conditioned labels and are used only for evaluation.",
        ],
    }
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote_summary={summary_output}")
    for name, result in summary["evaluation_modes"].items():
        frame = result["summary"]["frame_action_head"]
        print(
            f"{name}: legal={frame['legal_rate']:.3f} "
            f"target={frame['target_set_accuracy']:.3f} "
            f"correct={frame['target_set_hit_count']}/{frame['num_frames']}"
        )


def load_sequences(data_root: Path, splits: list[str]) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for split in splits:
        payload = json.loads(
            (data_root / "navlock_sequences" / f"scene_sequences_{split}.json").read_text(
                encoding="utf-8"
            )
        )
        for sequence in payload.get("sequences") or []:
            item = dict(sequence)
            item["split"] = split
            sequences.append(item)
    return sequences


def load_hydro_predictions_for_splits(
    splits: list[str],
) -> dict[str, dict[Any, dict[str, Any]]]:
    by_token: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    for split in splits:
        loaded = load_hydro_predictions(
            Path("outputs") / "hydro3dnet_navlock" / f"{split}_predictions.json"
        )
        by_token.update(loaded["by_token"])
        by_index.update(loaded["by_index"])
    return {"by_token": by_token, "by_index": by_index}


def load_rtmdet_for_splits(
    splits: list[str], score_threshold: float
) -> dict[str, list[dict[str, Any]]]:
    boxes: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        boxes.update(
            load_rtmdet_ship_boxes(
                Path("outputs") / "mmdet2d" / "navlock_rtmdet_s" / f"{split}_predictions.pkl",
                score_threshold,
            )
        )
    return boxes


def derive_full_frame_world_state(
    sequences: list[dict[str, Any]],
    *,
    berths_by_scene: dict[str, list[dict[str, Any]]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    data_root: Path,
    lock_chamber_bounds: dict[str, float],
    rtmdet_by_path: Optional[dict[str, list[dict[str, Any]]]],
    use_rtmdet: bool,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    world_state_by_sample: dict[str, dict[str, Any]] = {}
    total = sum(len(sequence.get("frames") or []) for sequence in sequences)
    done = 0
    for sequence in sequences:
        frames = sequence.get("frames") or []
        for frame_index, frame in enumerate(frames):
            start = max(0, frame_index - max(1, args.window_size) + 1)
            window_frames = frames[start : frame_index + 1]
            window_sequence = {
                "scene_token": sequence.get("scene_token"),
                "has_prediction_target": False,
                "frames": window_frames,
                "prediction_input_frame_indices": list(range(len(window_frames))),
            }
            pred = derive_prediction_from_hydro_tracks(
                window_sequence,
                berths_by_scene.get(sequence.get("scene_token"), []),
                hydro_predictions,
                data_root=data_root,
                lock_chamber_bounds=lock_chamber_bounds,
                rtmdet_by_path=rtmdet_by_path,
                score_threshold=args.score_threshold,
                track_distance_m=args.track_distance_m,
                recover_rtmdet_multicamera=use_rtmdet,
                support_iou_threshold=args.support_iou_threshold,
                recovery_min_cameras=args.recovery_min_cameras,
                recovery_max_ray_residual_m=args.recovery_max_ray_residual_m,
                recovery_cluster_distance_m=args.recovery_cluster_distance_m,
                recovery_existing_distance_m=args.recovery_existing_distance_m,
                recovery_chamber_margin_m=args.recovery_chamber_margin_m,
                recover_open_gate_new_ships=use_rtmdet,
                open_gate_min_cameras=args.open_gate_min_cameras,
                open_gate_zone_length_m=args.open_gate_zone_length_m,
                open_gate_max_candidates=args.open_gate_max_candidates,
                recovery_current_frame_only=True,
                future_motion_mode="settle_aware",
                eval_token_map=False,
                eval_open_gate_new_ship_tokens=False,
            )
            pred["split"] = sequence.get("split")
            pred["timestamp"] = frame.get("timestamp")
            pred["timestamp_str"] = frame.get("timestamp_str")
            pred["track_source"]["window_size"] = len(window_frames)
            pred["track_source"]["disable_berth_motion_prior"] = bool(
                args.disable_berth_motion_prior
            )
            sample_token = frame.get("sample_token")
            if sample_token:
                world_state_by_sample[str(sample_token)] = pred
            done += 1
            if done == total or done % 100 == 0:
                print(f"  derived {done}/{total}", file=sys.stderr, flush=True)
    return world_state_by_sample


def replace_candidate_states(
    rows: list[dict[str, Any]],
    *,
    world_state_by_sample: dict[str, dict[str, Any]],
    recompute_mask: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    report = {
        "recompute_mask": recompute_mask,
        "replaced_candidate_rows": 0,
        "replaced_samples": set(),
        "ship_phase_counts": Counter(),
    }
    for row in rows:
        new_row = copy.deepcopy(row)
        state = world_state_by_sample.get(str(row.get("sample_token")))
        if state:
            new_row["current_state"] = deployable_current_state(row, state)
            report["replaced_candidate_rows"] += 1
            report["replaced_samples"].add(str(row.get("sample_token")))
            report["ship_phase_counts"][new_row["current_state"].get("ship_operation_phase")] += 1
        output.append(new_row)

    if recompute_mask:
        recompute_candidate_masks(output)
    report["replaced_samples"] = len(report["replaced_samples"])
    report["ship_phase_counts"] = dict(report["ship_phase_counts"])
    if recompute_mask:
        valid_counts = Counter(
            row.get("candidate_action")
            for row in output
            if row.get("sample_token") in world_state_by_sample and row.get("is_valid")
        )
        report["valid_action_counts_after_recompute"] = dict(valid_counts)
    return output, report


def deployable_current_state(
    row: dict[str, Any], world_state: dict[str, Any]
) -> dict[str, Any]:
    original = copy.deepcopy(row.get("current_state") or {})
    current_state = {
        key: original.get(key)
        for key in ("upper_gate_state", "lower_gate_state", "water_state", "water_level")
        if key in original
    }
    fused = {
        "prediction_row": {
            "prediction_json": {
                "current_state": current_state,
                "lock_occupancy": world_state.get("lock_occupancy"),
                "vessel_motion_flow": world_state.get("vessel_motion_flow"),
                "ship_behavior": {"ship_intentions": []},
            }
        },
        "world_state_row": world_state,
    }
    state = fused_current_state(
        original,
        fused=fused,
        direction=str(row.get("direction") or "unknown"),
    )
    apply_planner_feature_stitch(state, world_state.get("planner_feature_stitch") or {})
    strip_non_deployable_state_labels(state)
    return state


def apply_planner_feature_stitch(
    state: dict[str, Any], stitch: dict[str, Any]
) -> None:
    if not stitch:
        return
    phase = stitch.get("ship_operation_phase")
    if phase not in {"ship_entering", "ship_leaving"}:
        return
    state["ship_operation_phase"] = phase
    state["all_in_chamber_ships_berthed_or_static"] = False
    state["no_ship_entering_or_leaving_inside_chamber"] = False
    if phase == "ship_entering":
        next_ship = stitch.get("next_ship_to_enter_weak")
        if isinstance(next_ship, dict):
            state["next_ship_to_enter_weak"] = copy.deepcopy(next_ship)
            state["queue_rank"] = [copy.deepcopy(next_ship)]
            state["max_parallel_entries"] = max(
                1,
                int(state.get("max_parallel_entries") or 0),
            )
        state["next_ship_to_leave_weak"] = None
        state["max_parallel_departures"] = 0
    elif phase == "ship_leaving":
        next_ship = stitch.get("next_ship_to_leave_weak")
        if isinstance(next_ship, dict):
            state["next_ship_to_leave_weak"] = copy.deepcopy(next_ship)
            state["max_parallel_departures"] = max(
                1,
                int(state.get("max_parallel_departures") or 0),
            )
        state["next_ship_to_enter_weak"] = None
        state["queue_rank"] = []
        state["max_parallel_entries"] = 0


def apply_temporal_dispatch_stitch(
    world_state_by_sample: dict[str, dict[str, Any]],
    *,
    sequences: list[dict[str, Any]],
    max_gap_sec: float,
    min_speed_mps: float,
) -> dict[str, Any]:
    report = {
        "enabled": True,
        "max_gap_sec": max_gap_sec,
        "min_speed_mps": min_speed_mps,
        "enter_frames_stitched": 0,
        "exit_frames_stitched": 0,
        "enter_evidence_frames": 0,
        "exit_evidence_frames": 0,
        "blocked_by_gate_or_water": 0,
        "cleared_by_gate_or_water": 0,
        "top_scenes": Counter(),
    }
    for sequence in sequences:
        direction = str(sequence.get("direction") or "unknown")
        entry_side, exit_side = planner_entry_exit_sides(direction)
        memory: dict[str, Optional[dict[str, Any]]] = {
            "enter": None,
            "exit": None,
        }
        frames = sorted(
            sequence.get("frames") or [],
            key=lambda frame: (
                int(frame.get("timestamp") or 0),
                str(frame.get("sample_token") or ""),
            ),
        )
        for frame in frames:
            sample_token = str(frame.get("sample_token") or "")
            world_state = world_state_by_sample.get(sample_token)
            if not world_state:
                continue
            state = planner_state_from_world_state(
                frame.get("lock_state") or {},
                world_state,
                direction=direction,
            )
            timestamp = int(frame.get("timestamp") or 0)
            enter_evidence = dispatch_evidence_from_state(
                state,
                kind="enter",
                min_speed_mps=min_speed_mps,
            )
            exit_evidence = dispatch_evidence_from_state(
                state,
                kind="exit",
                min_speed_mps=min_speed_mps,
            )
            if enter_evidence:
                memory["enter"] = {
                    "timestamp": timestamp,
                    "next_ship": copy.deepcopy(enter_evidence),
                }
                report["enter_evidence_frames"] += 1
            if exit_evidence:
                memory["exit"] = {
                    "timestamp": timestamp,
                    "next_ship": copy.deepcopy(exit_evidence),
                }
                report["exit_evidence_frames"] += 1

            active_kind = None
            active_side = None
            active_memory = None
            if dispatch_memory_active(
                state,
                memory["exit"],
                side=exit_side,
                timestamp=timestamp,
                max_gap_sec=max_gap_sec,
            ):
                active_kind = "exit"
                active_side = exit_side
                active_memory = memory["exit"]
            elif dispatch_memory_active(
                state,
                memory["enter"],
                side=entry_side,
                timestamp=timestamp,
                max_gap_sec=max_gap_sec,
            ):
                active_kind = "enter"
                active_side = entry_side
                active_memory = memory["enter"]

            if active_kind is None:
                if memory["enter"] or memory["exit"]:
                    report["blocked_by_gate_or_water"] += 1
                if state.get("water_state") not in {"idle", None} or (
                    entry_side
                    and exit_side
                    and state.get(f"{entry_side}_gate_state") != "open"
                    and state.get(f"{exit_side}_gate_state") != "open"
                ):
                    memory["enter"] = None
                    memory["exit"] = None
                    report["cleared_by_gate_or_water"] += 1
                continue

            next_ship = copy.deepcopy((active_memory or {}).get("next_ship") or {})
            if next_ship:
                next_ship["source"] = f"temporal_dispatch_stitch:{next_ship.get('source')}"
                next_ship["side"] = active_side or next_ship.get("side") or "unknown"
            apply_stitched_dispatch_to_world_state(
                world_state,
                kind=active_kind,
                next_ship=next_ship,
            )
            if active_kind == "enter":
                report["enter_frames_stitched"] += 1
            else:
                report["exit_frames_stitched"] += 1
            report["top_scenes"][str(sequence.get("scene_token"))] += 1
    report["top_scenes"] = dict(report["top_scenes"].most_common(20))
    return report


def planner_state_from_world_state(
    lock_state: dict[str, Any],
    world_state: dict[str, Any],
    *,
    direction: str,
) -> dict[str, Any]:
    row = {
        "current_state": {
            "upper_gate_state": lock_state.get("upper_gate_state"),
            "lower_gate_state": lock_state.get("lower_gate_state"),
            "water_state": lock_state.get("water_state"),
            "water_level": lock_state.get("water_level"),
        },
        "direction": direction,
    }
    return deployable_current_state(row, world_state)


def dispatch_evidence_from_state(
    state: dict[str, Any],
    *,
    kind: str,
    min_speed_mps: float,
) -> Optional[dict[str, Any]]:
    phase = state.get("ship_operation_phase")
    key = "next_ship_to_enter_weak" if kind == "enter" else "next_ship_to_leave_weak"
    target_phase = "ship_entering" if kind == "enter" else "ship_leaving"
    next_ship = state.get(key)
    if phase != target_phase or not isinstance(next_ship, dict):
        return None
    speed = float(next_ship.get("speed_mps") or 0.0)
    if speed < min_speed_mps:
        return None
    return next_ship


def dispatch_memory_active(
    state: dict[str, Any],
    memory: Optional[dict[str, Any]],
    *,
    side: Optional[str],
    timestamp: int,
    max_gap_sec: float,
) -> bool:
    if not memory or side is None:
        return False
    if state.get("water_state") != "idle":
        return False
    if state.get(f"{side}_gate_state") != "open":
        return False
    last_timestamp = int(memory.get("timestamp") or 0)
    if last_timestamp <= 0:
        return False
    return (timestamp - last_timestamp) / 1_000_000.0 <= max_gap_sec


def apply_stitched_dispatch_to_world_state(
    world_state: dict[str, Any],
    *,
    kind: str,
    next_ship: dict[str, Any],
) -> None:
    planner = world_state.setdefault("planner_feature_stitch", {})
    planner["ship_operation_phase"] = (
        "ship_entering" if kind == "enter" else "ship_leaving"
    )
    planner["next_ship_to_enter_weak"] = next_ship if kind == "enter" else None
    planner["next_ship_to_leave_weak"] = next_ship if kind == "exit" else None
    planner["source"] = "temporal_dispatch_stitch"


def strip_non_deployable_state_labels(state: dict[str, Any]) -> None:
    state["observed_action"] = None
    state["action_target"] = None
    state["action_source"] = None
    state["action_confidence"] = None
    state["ship_dispatch_action"] = "hold"
    state["ship_dispatch_targets"] = []
    state["ship_dispatch_target_count"] = 0
    state["ship_dispatch_source"] = None
    state["ship_dispatch_confidence"] = None
    state["ship_dispatch_conflict"] = None


def sorted_world_states(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("split")),
            int(row.get("timestamp") or 0),
            str(row.get("sample_token")),
        ),
    )


def planner_entry_exit_sides(direction: str) -> tuple[Optional[str], Optional[str]]:
    if direction == "upstream":
        return "lower", "upper"
    if direction == "downstream":
        return "upper", "lower"
    return None, None


if __name__ == "__main__":
    main()
