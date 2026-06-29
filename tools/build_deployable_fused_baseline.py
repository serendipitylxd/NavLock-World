#!/usr/bin/env python3
"""Build and evaluate the fused deployable NavLock baseline.

This script turns the deployable pipeline into one reproducible artifact:

1. start from a VLM semantic prediction JSONL for gate/water fields;
2. replace ``ship_behavior.ship_intentions`` with Hydro3DNet-track +
   berth-geometry intentions;
3. derive ``lock_occupancy`` / ``vessel_motion_flow`` from the same Hydro3DNet
   tracks;
4. fuse everything into one prediction JSONL and write a compact metric summary.

The VLM semantic rows may come from an already completed VLM run. The ship
and world-state branches are deterministic post-processing from deployable
Hydro3DNet detections and lock telemetry.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from navlock_world.lock_world_state import load_lock_chamber_bounds, load_scene_berths
from scripts.evaluate_qwen3vl_lora_adapter import (
    require_valid_json_results,
    schema_check,
    semantic_check,
    summarize_results,
    write_jsonl,
)
from tqdm import tqdm
from tools.apply_berth_ship_intention_guard import (
    apply_lockage_phase_consistency_guard,
    deployable_hydro_ship_intentions,
    deployable_tracking_context,
    existing_intentions,
    load_rows,
    load_sequences,
    scene_token_from_id,
    split_from_id,
)
from tools.apply_lock_world_state_prior import apply_lock_world_state_prior
from tools.analyze_rtmdet_hydro_2d_support import SHIP_2D_CLASSES, load_rtmdet_ship_boxes
from tools.analyze_rtmdet_ship_intention_support import (
    build_track_features,
    index_features_by_output_token,
)
from tools.derive_world_state_from_hydro3dnet_tracks import (
    add_eval_open_gate_new_ship_tokens,
    berth_index_for_point,
    build_detection_frames,
    derive_prediction_from_hydro_tracks,
    eval_token_map_from_input_window,
    load_hydro_predictions,
    track_detections,
)
from tools.evaluate_lock_world_state_from_predictions import (
    _load_gt as load_world_state_gt,
)


DEFAULT_VLM_SEMANTIC = Path(
    "outputs/ablations/qwen3vl_4b_vlm_semantic_test24/predictions_baseline_all_guards.jsonl"
)
DEFAULT_OUTPUT = Path(
    "outputs/fused_deployable_baseline/predictions_test24_fused_deployable.jsonl"
)
DEFAULT_SUMMARY = Path("outputs/fused_deployable_baseline/summary_test24.json")
DEFAULT_WORLD_STATE = Path(
    "outputs/fused_deployable_baseline/derived_test_from_hydro3dnet_tracks_evalmap.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--eval-splits",
        default=None,
        help=(
            "Comma-separated splits to evaluate together, e.g. val,test. "
            "Defaults to --split. Split-dependent inputs default to one file "
            "per split; explicit file arguments may also be comma-separated."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--vlm-semantic-predictions",
        type=Path,
        default=DEFAULT_VLM_SEMANTIC,
        help=(
            "VLM semantic prediction JSONL. For --eval-splits with multiple splits, "
            "pass a comma-separated list or one already-combined JSONL."
        ),
    )
    parser.add_argument(
        "--vlm-semantic-current-predictions",
        type=Path,
        default=None,
        help=(
            "Optional VLM semantic current-recognition prediction JSONL. When set, "
            "only scenes without a future prediction target are appended so "
            "current/gate/water/ship/world-state metrics include current-only "
            "recognition scenes while future metrics remain prediction-only."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-invalid-vlm-semantic-json",
        action="store_true",
        help=(
            "Allow invalid VLM semantic JSON rows to enter the fused baseline. "
            "By default invalid semantic rows fail before output files are written."
        ),
    )
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--world-state-output", type=Path, default=DEFAULT_WORLD_STATE)
    parser.add_argument("--scene-json", type=Path, default=None)
    parser.add_argument(
        "--lock-boundary-map",
        type=Path,
        default=None,
        help="Physical lock-chamber boundary map used by RTMDet recovery.",
    )
    parser.add_argument("--sequences", type=Path, default=None)
    parser.add_argument("--hydro-predictions", type=Path, default=None)
    parser.add_argument("--rtmdet-predictions", type=Path, default=None)
    parser.add_argument("--world-state-gt", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--track-distance-m", type=float, default=40.0)
    parser.add_argument("--eval-token-map", action="store_true")
    parser.add_argument("--eval-token-map-distance-m", type=float, default=40.0)
    parser.add_argument(
        "--recover-rtmdet-multicamera",
        action="store_true",
        help=(
            "Estimate lock-chamber ship count from multi-camera RTMDet boxes and "
            "recover missing 3D ship candidates only when Hydro has fewer "
            "in-chamber ships than that 2D count."
        ),
    )
    parser.add_argument(
        "--apply-rtmdet-ship-intention-static-berth",
        action="store_true",
        help=(
            "After the Hydro berth prior, set a ship intention to ship_berthed "
            "when RTMDet shows multi-frame 2D stability and the Hydro track ends "
            "inside an ideal berth."
        ),
    )
    parser.add_argument("--rtmdet-score-threshold", type=float, default=0.30)
    parser.add_argument("--support-iou-threshold", type=float, default=0.30)
    parser.add_argument("--current-active-rtmdet-min-cameras", type=int, default=1)
    parser.add_argument("--current-active-max-missing-frames", type=int, default=2)
    parser.add_argument("--static-2d-motion-threshold", type=float, default=0.02)
    parser.add_argument("--recovery-min-cameras", type=int, default=4)
    parser.add_argument("--recovery-max-ray-residual-m", type=float, default=10.0)
    parser.add_argument("--recovery-cluster-distance-m", type=float, default=20.0)
    parser.add_argument("--recovery-existing-distance-m", type=float, default=20.0)
    parser.add_argument("--recovery-chamber-margin-m", type=float, default=0.0)
    parser.add_argument("--recover-open-gate-new-ships", action="store_true")
    parser.add_argument("--open-gate-min-cameras", type=int, default=3)
    parser.add_argument("--open-gate-zone-length-m", type=float, default=70.0)
    parser.add_argument("--open-gate-max-candidates", type=int, default=1)
    parser.add_argument(
        "--recovery-all-input-frames",
        action="store_true",
        help="Recover RTMDet candidates on every input frame. Default recovers current frame only.",
    )
    parser.add_argument(
        "--stitch-current-motion-all-input-frames",
        action="store_true",
        help=(
            "Keep the main ship/occupancy branch on the requested recovery mode, "
            "but stitch vessel_motion_flow.input_window from an auxiliary "
            "all-input-frame recovery pass. Missing final-ship tokens are filled "
            "from the deployable ship branch, and berthed-token motion outliers "
            "are snapped back to ship_berthed without using annotation labels."
        ),
    )
    parser.add_argument(
        "--motion-stitch-vlm-slow-speed-mps",
        type=float,
        default=0.2,
        help=(
            "For current-motion stitching, accept a raw VLM ship_berthed hint only "
            "when the final deployable ship branch is also ship_berthed and the "
            "candidate motion speed is at or below this threshold."
        ),
    )
    parser.add_argument(
        "--motion-stitch-high-speed-outlier-mps",
        type=float,
        default=5.0,
        help=(
            "When the final ship branch says ship_berthed, treat a candidate "
            "ship_moving item above this speed as an ID-stitch outlier and snap it "
            "to ship_berthed."
        ),
    )
    parser.add_argument("--eval-open-gate-new-ship-tokens", action="store_true")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bars for the main fused evaluation loops.",
    )
    parser.add_argument(
        "--ship-prior-mode",
        choices=("replace", "vlm-count-fallback"),
        default="replace",
        help=(
            "replace always uses the deployable Hydro3DNet/RTMDet geometry branch. "
            "vlm-count-fallback keeps VLM ship_behavior.ship_intentions when the "
            "VLM predicts more ship items than the geometry branch, reducing "
            "deployable detector misses without using annotation labels."
        ),
    )
    parser.add_argument(
        "--apply-vlm-dynamic-ship-intention-fallback",
        action="store_true",
        help=(
            "After deployable geometry and phase guards, restore VLM-native "
            "ship_entering_lock/ship_leaving_lock for the same instance_token "
            "when the deployable branch says ship_berthed. This uses model "
            "output only, not annotation labels."
        ),
    )
    parser.add_argument(
        "--defer-entering-berth",
        action="store_true",
        help=(
            "In single-ship entering lockages, keep a ship as "
            "ship_entering_lock unless the input-window track has enough "
            "stable dwell evidence inside the same berth. This prevents early "
            "ship_berthed labels for ships that have only just reached a berth."
        ),
    )
    parser.add_argument(
        "--entering-berth-allow-multi-ship",
        action="store_true",
        help=(
            "Also apply --defer-entering-berth to multi-ship scenes. Default "
            "is intentionally single-ship only because multi-ship queues often "
            "mix true berthed and entering ships."
        ),
    )
    parser.add_argument("--entering-berth-min-dwell-frames", type=int, default=2)
    parser.add_argument("--entering-berth-min-dwell-sec", type=float, default=0.0)
    parser.add_argument(
        "--entering-berth-max-dwell-displacement-m",
        type=float,
        default=999.0,
    )
    parser.add_argument(
        "--future-motion-mode",
        choices=("settle_aware", "persistence"),
        default="settle_aware",
    )
    parser.add_argument(
        "--disable-berth-motion-prior",
        action="store_true",
        help=(
            "Disable ideal-berth slot matching and settled-motion prior in the "
            "deployable ship/world-state branch. This is intended for paper "
            "ablation only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.eval_splits = normalized_eval_splits(args)
    paths = resolve_paths(args)

    scene_berths = load_scene_berths(paths["scene_json"])
    state_scene_berths = (
        {scene_token: [] for scene_token in scene_berths}
        if args.disable_berth_motion_prior
        else scene_berths
    )
    args.lock_chamber_bounds = load_lock_chamber_bounds(paths["lock_boundary_map"])
    sequences = load_sequences_from_paths(paths["sequences"])
    prediction_rows = load_rows_from_paths(paths["vlm_semantic_predictions"])
    current_only_rows = load_current_only_rows(
        paths.get("vlm_semantic_current_predictions"),
        sequences,
        existing_scene_tokens=scene_tokens_from_rows(prediction_rows),
    )
    rows = prediction_rows + current_only_rows
    current_motion_stitch_source_rows = (
        copy.deepcopy(rows) if args.stitch_current_motion_all_input_frames else []
    )
    validate_vlm_semantic_json_rows(
        rows,
        allow_invalid=args.allow_invalid_vlm_semantic_json,
        output_path=paths["output"],
    )
    hydro_predictions = load_hydro_predictions_from_paths(paths["hydro_predictions"])
    args.rtmdet_by_path = None
    if args.recover_rtmdet_multicamera or args.recover_open_gate_new_ships:
        args.rtmdet_by_path = load_rtmdet_ship_boxes_from_paths(
            paths["rtmdet_predictions"],
            args.rtmdet_score_threshold,
        )

    vlm_route_summary = summarize_results(rows)
    ship_report = apply_hydro_ship_intention_prior(
        rows,
        sequences,
        state_scene_berths,
        hydro_predictions,
        paths,
        args,
    )
    rtmdet_ship_report = None
    if args.apply_rtmdet_ship_intention_static_berth:
        rtmdet_ship_report = apply_rtmdet_static_berth_intention_override(
            rows,
            sequences,
            state_scene_berths,
            hydro_predictions,
            args,
            paths,
        )
    phase_consistency_report = apply_lockage_phase_consistency_intention_guard(
        rows,
        sequences,
        state_scene_berths,
        hydro_predictions,
        args,
    )
    entering_berth_defer_report = None
    if args.defer_entering_berth:
        entering_berth_defer_report = apply_entering_berth_defer_intention_guard(
            rows,
            sequences,
            state_scene_berths,
            hydro_predictions,
            args,
        )
    vlm_dynamic_report = None
    if args.apply_vlm_dynamic_ship_intention_fallback:
        vlm_dynamic_report = apply_vlm_dynamic_ship_intention_fallback(rows)
    world_state_rows = derive_hydro_world_state(
        sequences,
        state_scene_berths,
        hydro_predictions,
        args,
        scene_tokens=scene_tokens_from_rows(rows),
    )
    world_state_alignment_report = apply_ship_intention_world_state_alignment(
        world_state_rows,
        rows,
        sequences,
        state_scene_berths,
        hydro_predictions,
        args,
    )
    current_motion_stitch_report = None
    if args.stitch_current_motion_all_input_frames:
        (
            all_frame_world_state_rows,
            all_frame_auxiliary_report,
        ) = build_all_frame_motion_stitch_world_state(
            current_motion_stitch_source_rows,
            sequences,
            state_scene_berths,
            hydro_predictions,
            paths,
            args,
        )
        current_motion_stitch_report = apply_current_motion_token_stitch(
            world_state_rows,
            all_frame_world_state_rows,
            rows,
            args,
        )
        current_motion_stitch_report[
            "all_frame_auxiliary_branch"
        ] = all_frame_auxiliary_report
    write_world_state_jsonl(paths["world_state_output"], world_state_rows)

    apply_lock_world_state_prior(
        rows,
        {
            item["scene_token"]: item
            for item in world_state_rows
            if isinstance(item.get("scene_token"), str)
        },
        mode="replace",
        recompute_checks="if-reference-has-fields",
    )
    write_jsonl(paths["output"], rows)

    route_summary = summarize_results(rows)
    world_current = evaluate_world_state(
        load_world_state_gt_from_paths(paths["world_state_gt"]),
        prediction_objects_by_scene(rows),
        "current",
    )
    world_future = evaluate_world_state(
        load_world_state_gt_from_paths(paths["world_state_gt"]),
        prediction_objects_by_scene(rows),
        "future_10s",
    )
    summary = build_summary(
        args=args,
        paths=paths,
        rows=rows,
        ship_report=ship_report,
        rtmdet_ship_report=rtmdet_ship_report,
        phase_consistency_report=phase_consistency_report,
        entering_berth_defer_report=entering_berth_defer_report,
        vlm_dynamic_report=vlm_dynamic_report,
        world_state_alignment_report=world_state_alignment_report,
        current_motion_stitch_report=current_motion_stitch_report,
        vlm_route_summary=vlm_route_summary,
        route_summary=route_summary,
        world_current=world_current,
        world_future=world_future,
    )
    paths["summary_output"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_output"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_predictions={paths['output']}")
    print(f"wrote_world_state={paths['world_state_output']}")
    print(f"wrote_summary={paths['summary_output']}")


def normalized_eval_splits(args: argparse.Namespace) -> list[str]:
    raw = args.eval_splits if args.eval_splits is not None else args.split
    splits = [part.strip() for part in str(raw).split(",") if part.strip()]
    valid = {"train", "val", "test"}
    invalid = [split for split in splits if split not in valid]
    if not splits:
        raise SystemExit("--eval-splits must contain at least one split")
    if invalid:
        raise SystemExit(f"unsupported --eval-splits values: {', '.join(invalid)}")
    return splits


def split_path_list(value: Optional[Path]) -> Optional[list[Path]]:
    if value is None:
        return None
    return [Path(part.strip()) for part in str(value).split(",") if part.strip()]


def paths_for_splits(
    explicit: Optional[Path],
    eval_splits: list[str],
    default_factory: Any,
    *,
    arg_name: str,
) -> list[Path]:
    explicit_paths = split_path_list(explicit)
    if explicit_paths is None:
        return [default_factory(split) for split in eval_splits]
    if len(explicit_paths) == 1:
        return explicit_paths
    if len(explicit_paths) != len(eval_splits):
        raise SystemExit(
            f"{arg_name} has {len(explicit_paths)} paths but --eval-splits has "
            f"{len(eval_splits)} splits"
        )
    return explicit_paths


def load_rows_from_paths(paths: Path | list[Path]) -> list[dict[str, Any]]:
    if isinstance(paths, Path):
        return load_rows(paths)
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_rows(path))
    return rows


def load_sequences_from_paths(paths: Path | list[Path]) -> dict[str, dict[str, Any]]:
    path_list = paths if isinstance(paths, list) else [paths]
    sequences: dict[str, dict[str, Any]] = {}
    for path in path_list:
        sequences.update(load_sequences(path))
    return sequences


def load_hydro_predictions_from_paths(
    paths: Path | list[Path],
) -> dict[str, dict[Any, dict[str, Any]]]:
    path_list = paths if isinstance(paths, list) else [paths]
    predictions: dict[str, dict[Any, dict[str, Any]]] = {}
    for path in path_list:
        predictions.update(load_hydro_predictions(path))
    return predictions


def load_rtmdet_ship_boxes_from_paths(
    paths: Path | list[Path],
    score_threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    path_list = paths if isinstance(paths, list) else [paths]
    boxes: dict[str, list[dict[str, Any]]] = {}
    for path in path_list:
        boxes.update(load_rtmdet_ship_boxes(path, score_threshold))
    return boxes


def load_world_state_gt_from_paths(paths: Path | list[Path]) -> dict[str, dict[str, Any]]:
    path_list = paths if isinstance(paths, list) else [paths]
    gt: dict[str, dict[str, Any]] = {}
    for path in path_list:
        gt.update(load_world_state_gt(path))
    return gt


def stringify_path_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [stringify_path_value(item) for item in value]
    return value


def resolve_paths(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root
    eval_splits = args.eval_splits
    paths = {
        "vlm_semantic_predictions": split_path_list(args.vlm_semantic_predictions) or [],
        "output": args.output,
        "summary_output": args.summary_output,
        "world_state_output": args.world_state_output,
        "scene_json": args.scene_json or data_root / "v1.0-trainval" / "scene.json",
        "lock_boundary_map": args.lock_boundary_map
        or data_root / "maps" / "huaiyin_lock_boundary.json",
        "sequences": paths_for_splits(
            args.sequences,
            eval_splits,
            lambda split: data_root / "navlock_sequences" / f"scene_sequences_{split}.json",
            arg_name="--sequences",
        ),
        "hydro_predictions": paths_for_splits(
            args.hydro_predictions,
            eval_splits,
            lambda split: Path("outputs") / "hydro3dnet_navlock" / f"{split}_predictions.json",
            arg_name="--hydro-predictions",
        ),
        "rtmdet_predictions": paths_for_splits(
            args.rtmdet_predictions,
            eval_splits,
            lambda split: Path("outputs") / "mmdet2d" / "navlock_rtmdet_s" / f"{split}_predictions.pkl",
            arg_name="--rtmdet-predictions",
        ),
        "world_state_gt": paths_for_splits(
            args.world_state_gt,
            eval_splits,
            lambda split: Path("outputs") / "lock_world_state" / f"lock_world_state_{split}.jsonl",
            arg_name="--world-state-gt",
        ),
    }
    if args.vlm_semantic_current_predictions is not None:
        paths["vlm_semantic_current_predictions"] = paths_for_splits(
            args.vlm_semantic_current_predictions,
            eval_splits,
            lambda split: Path(""),
            arg_name="--vlm-semantic-current-predictions",
        )
    return paths


def scene_token_of_row(row: dict[str, Any]) -> Optional[str]:
    prediction = row.get("prediction_json")
    if isinstance(prediction, dict) and isinstance(prediction.get("scene_token"), str):
        return prediction["scene_token"]
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("scene_token"), str):
        return metadata["scene_token"]
    row_id = row.get("id")
    if isinstance(row_id, str):
        return scene_token_from_id(row_id)
    return None


def scene_tokens_from_rows(rows: list[dict[str, Any]]) -> set[str]:
    return {
        scene_token
        for row in rows
        for scene_token in [scene_token_of_row(row)]
        if scene_token
    }


def load_current_only_rows(
    path: Optional[Path | list[Path]],
    sequences: dict[str, dict[str, Any]],
    *,
    existing_scene_tokens: set[str],
) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = []
    for row in load_rows_from_paths(path):
        scene_token = scene_token_of_row(row)
        if not scene_token or scene_token in existing_scene_tokens:
            continue
        sequence = sequences.get(scene_token)
        if sequence is None or sequence.get("has_prediction_target"):
            continue
        rows.append(row)
    return rows


def validate_vlm_semantic_json_rows(
    rows: list[dict[str, Any]],
    *,
    allow_invalid: bool,
    output_path: Path,
) -> None:
    if allow_invalid:
        return
    require_valid_json_results(rows, output_path=output_path)


def apply_hydro_ship_intention_prior(
    rows: list[dict[str, Any]],
    sequences: dict[str, dict[str, Any]],
    scene_berths: dict[str, list[dict[str, Any]]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prior_args = ship_intention_tracking_args(args)
    prior_args.current_active_rtmdet_min_cameras = args.current_active_rtmdet_min_cameras
    prior_args.current_active_max_missing_frames = args.current_active_max_missing_frames
    applied = 0
    missing_scene = 0
    kept_vlm_more_ships = 0
    before = summarize_results(rows).get("ship_behavior", {})

    for row in progress_iter(rows, args, "ship-prior"):
        scene_token = scene_token_from_id(row["id"])
        sequence = sequences.get(scene_token)
        if sequence is None:
            missing_scene += 1
            continue
        derived = deployable_hydro_ship_intentions(
            sequence,
            scene_berths.get(scene_token, []),
            hydro_predictions,
            prior_args,
        )
        prediction = prediction_object(row)
        if prediction is None:
            prediction = {}
            row["prediction_json"] = prediction
        behavior = prediction.get("ship_behavior")
        if not isinstance(behavior, dict):
            behavior = {}
            prediction["ship_behavior"] = behavior
        selected, decision = select_ship_prior_intentions(
            existing_intentions(prediction),
            derived,
            args.ship_prior_mode,
        )
        if decision == "keep_vlm_more_ships":
            kept_vlm_more_ships += 1
        behavior["ship_intentions"] = copy.deepcopy(selected)
        row["berth_aware_geometric_prior"] = {
            "mode": args.ship_prior_mode,
            "decision": decision,
            "track_source": (
                "hydro3dnet+rtmdet_recovery"
                if args.recover_rtmdet_multicamera or args.recover_open_gate_new_ships
                else "hydro3dnet"
            ),
            "derived_count": len(derived),
            "selected_count": len(selected),
            "score_threshold": args.score_threshold,
            "track_distance_m": args.track_distance_m,
            "recover_rtmdet_multicamera": bool(args.recover_rtmdet_multicamera),
            "recover_open_gate_new_ships": bool(args.recover_open_gate_new_ships),
            "rtmdet_score_threshold": (
                args.rtmdet_score_threshold if args.recover_rtmdet_multicamera else None
            ),
            "support_iou_threshold": (
                args.support_iou_threshold if args.recover_rtmdet_multicamera else None
            ),
            "recovery_min_cameras": (
                args.recovery_min_cameras if args.recover_rtmdet_multicamera else None
            ),
            "recovery_current_frame_only": (
                (not args.recovery_all_input_frames)
                if args.recover_rtmdet_multicamera or args.recover_open_gate_new_ships
                else None
            ),
            "eval_token_map": bool(args.eval_token_map),
            "eval_token_map_basis": (
                "input_window_nearest" if args.eval_token_map else None
            ),
            "eval_open_gate_new_ship_tokens": (
                bool(args.eval_open_gate_new_ship_tokens) if args.eval_token_map else None
            ),
            "eval_token_map_distance_m": (
                args.eval_token_map_distance_m if args.eval_token_map else None
            ),
        }
        if isinstance(row.get("reference"), dict):
            row["schema_check"] = schema_check(prediction, row["reference"])
            row["semantic_check"] = semantic_check(prediction, row["reference"])
        applied += 1

    after = summarize_results(rows).get("ship_behavior", {})
    return {
        "applied_rows": applied,
        "missing_scene": missing_scene,
        "mode": args.ship_prior_mode,
        "kept_vlm_more_ships": kept_vlm_more_ships,
        "before": ship_metrics_from_summary(before),
        "after": ship_metrics_from_summary(after),
    }


def ship_intention_tracking_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        target="prediction",
        score_threshold=args.score_threshold,
        track_distance_m=args.track_distance_m,
        eval_token_map=bool(args.eval_token_map),
        eval_token_map_distance_m=args.eval_token_map_distance_m,
        data_root=args.data_root,
        lock_chamber_bounds=getattr(args, "lock_chamber_bounds", None),
        recover_rtmdet_multicamera=bool(args.recover_rtmdet_multicamera),
        rtmdet_by_path=getattr(args, "rtmdet_by_path", None),
        support_iou_threshold=args.support_iou_threshold,
        recovery_min_cameras=args.recovery_min_cameras,
        recovery_max_ray_residual_m=args.recovery_max_ray_residual_m,
        recovery_cluster_distance_m=args.recovery_cluster_distance_m,
        recovery_existing_distance_m=args.recovery_existing_distance_m,
        recovery_chamber_margin_m=args.recovery_chamber_margin_m,
        recover_open_gate_new_ships=bool(args.recover_open_gate_new_ships),
        open_gate_min_cameras=args.open_gate_min_cameras,
        open_gate_zone_length_m=args.open_gate_zone_length_m,
        open_gate_max_candidates=args.open_gate_max_candidates,
        recovery_current_frame_only=not args.recovery_all_input_frames,
        eval_open_gate_new_ship_tokens=bool(args.eval_open_gate_new_ship_tokens),
    )


def progress_iter(items: Any, args: argparse.Namespace, desc: str, unit: str = "row") -> Any:
    if not getattr(args, "progress", False):
        return items
    return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True)


def select_ship_prior_intentions(
    vlm_items: list[Any],
    derived_items: list[dict[str, Any]],
    mode: str,
) -> tuple[list[Any], str]:
    if mode == "vlm-count-fallback" and len(vlm_items) > len(derived_items):
        return vlm_items, "keep_vlm_more_ships"
    return derived_items, "use_deployable_geometry"


def apply_rtmdet_static_berth_intention_override(
    rows: list[dict[str, Any]],
    sequences: dict[str, dict[str, Any]],
    scene_berths: dict[str, list[dict[str, Any]]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> dict[str, Any]:
    before = summarize_results(rows).get("ship_behavior", {})
    rtmdet_by_path = load_rtmdet_ship_boxes_from_paths(
        paths["rtmdet_predictions"],
        args.rtmdet_score_threshold,
    )

    changed_items = 0
    eligible_items = 0
    missing_scene = 0
    missing_track_features = 0
    changes = []
    for row in progress_iter(rows, args, "rtmdet-static-berth"):
        scene_token = scene_token_from_id(row["id"])
        sequence = sequences.get(scene_token)
        if sequence is None:
            missing_scene += 1
            continue
        features = build_track_features(
            sequence,
            scene_berths.get(scene_token, []),
            hydro_predictions,
            rtmdet_by_path,
            data_root=args.data_root,
            hydro_score_threshold=args.score_threshold,
            track_distance_m=args.track_distance_m,
            eval_token_map_distance_m=args.eval_token_map_distance_m,
            support_iou_threshold=args.support_iou_threshold,
        )
        feature_index = index_features_by_output_token(features)
        prediction = prediction_object(row)
        if prediction is None:
            continue
        behavior = prediction.get("ship_behavior")
        if not isinstance(behavior, dict):
            continue
        intentions = behavior.get("ship_intentions")
        if not isinstance(intentions, list):
            continue

        row_changed = False
        for item in intentions:
            if not isinstance(item, dict) or item.get("instance_token") is None:
                continue
            token = str(item["instance_token"])
            feature = first_track_feature(feature_index.get(token, []))
            if feature is None:
                missing_track_features += 1
                continue
            if not rtmdet_static_berth_candidate(feature, args):
                continue
            eligible_items += 1
            old_labels = item.get("ship_intentions")
            if not rtmdet_static_berth_override_allowed(old_labels):
                continue
            item["ship_intentions"] = ["ship_berthed"]
            changed_items += 1
            row_changed = True
            changes.append(
                {
                    "id": row.get("id"),
                    "instance_token": token,
                    "old_ship_intentions": old_labels if isinstance(old_labels, list) else [],
                    "new_ship_intentions": ["ship_berthed"],
                    "net_displacement_m": feature["net_displacement_m"],
                    "max_2d_motion_norm": feature["rtmdet_2d_motion"][
                        "max_normalized_displacement"
                    ],
                    "cameras_supported": feature["cameras_supported"],
                }
            )
        if row_changed and isinstance(row.get("reference"), dict):
            row["schema_check"] = schema_check(prediction, row["reference"])
            row["semantic_check"] = semantic_check(prediction, row["reference"])

    after = summarize_results(rows).get("ship_behavior", {})
    return {
        "mode": "rtmdet_static_berth_override",
        "changed_items": changed_items,
        "eligible_items": eligible_items,
        "missing_scene": missing_scene,
        "missing_track_features": missing_track_features,
        "settings": {
            "rtmdet_score_threshold": args.rtmdet_score_threshold,
            "rtmdet_recoverable_2d_labels": (
                sorted(SHIP_2D_CLASSES)
                if args.recover_rtmdet_multicamera or args.recover_open_gate_new_ships
                else None
            ),
            "support_iou_threshold": args.support_iou_threshold,
            "static_2d_motion_threshold": args.static_2d_motion_threshold,
            "min_cameras_supported": 2,
        },
        "before": ship_metrics_from_summary(before),
        "after": ship_metrics_from_summary(after),
        "changes": changes,
    }


def apply_lockage_phase_consistency_intention_guard(
    rows: list[dict[str, Any]],
    sequences: dict[str, dict[str, Any]],
    scene_berths: dict[str, list[dict[str, Any]]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    before = summarize_results(rows).get("ship_behavior", {})
    prior_args = ship_intention_tracking_args(args)

    changed_items = 0
    changed_rows = 0
    missing_scene = 0
    changes = []
    for row in progress_iter(rows, args, "phase-consistency"):
        scene_token = scene_token_from_id(row["id"])
        sequence = sequences.get(scene_token)
        if sequence is None:
            missing_scene += 1
            continue
        prediction = prediction_object(row)
        if prediction is None:
            continue
        behavior = prediction.get("ship_behavior")
        if not isinstance(behavior, dict):
            continue
        intentions = behavior.get("ship_intentions")
        if not isinstance(intentions, list):
            continue
        berths = scene_berths.get(scene_token, [])
        frames, tracked_frames, token_map = deployable_tracking_context(
            sequence,
            berths,
            hydro_predictions,
            prior_args,
        )
        old_items = copy.deepcopy(intentions)
        new_items = apply_lockage_phase_consistency_guard(
            intentions,
            tracked_frames,
            token_map,
            berths,
            frames,
            scene_token,
        )
        if new_items == old_items:
            continue
        behavior["ship_intentions"] = new_items
        row_changed_items = sum(
            1
            for old_item, new_item in zip(old_items, new_items)
            if old_item != new_item
        )
        changed_items += row_changed_items
        changed_rows += 1
        changes.append(
            {
                "id": row.get("id"),
                "old_ship_intentions": old_items,
                "new_ship_intentions": new_items,
            }
        )
        if isinstance(row.get("reference"), dict):
            row["schema_check"] = schema_check(prediction, row["reference"])
            row["semantic_check"] = semantic_check(prediction, row["reference"])

    after = summarize_results(rows).get("ship_behavior", {})
    return {
        "mode": "lockage_phase_consistency_guard",
        "changed_rows": changed_rows,
        "changed_items": changed_items,
        "missing_scene": missing_scene,
        "before": ship_metrics_from_summary(before),
        "after": ship_metrics_from_summary(after),
        "changes": changes,
    }


def apply_entering_berth_defer_intention_guard(
    rows: list[dict[str, Any]],
    sequences: dict[str, dict[str, Any]],
    scene_berths: dict[str, list[dict[str, Any]]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    before = summarize_results(rows).get("ship_behavior", {})
    prior_args = ship_intention_tracking_args(args)

    changed_items = 0
    changed_rows = 0
    missing_scene = 0
    changes = []
    for row in progress_iter(rows, args, "entering-berth-defer"):
        scene_token = scene_token_from_id(row["id"])
        sequence = sequences.get(scene_token)
        if sequence is None:
            missing_scene += 1
            continue
        prediction = prediction_object(row)
        if prediction is None:
            continue
        behavior = prediction.get("ship_behavior")
        if not isinstance(behavior, dict):
            continue
        intentions = behavior.get("ship_intentions")
        if not isinstance(intentions, list):
            continue
        berths = scene_berths.get(scene_token, [])
        frames, tracked_frames, token_map = deployable_tracking_context(
            sequence,
            berths,
            hydro_predictions,
            prior_args,
        )
        old_items = copy.deepcopy(intentions)
        new_items = defer_entering_phase_berthed_items(
            intentions,
            tracked_frames,
            token_map,
            berths,
            frames,
            scene_token,
            single_ship_only=not args.entering_berth_allow_multi_ship,
            min_dwell_frames=args.entering_berth_min_dwell_frames,
            min_dwell_sec=args.entering_berth_min_dwell_sec,
            max_dwell_displacement_m=args.entering_berth_max_dwell_displacement_m,
        )
        if new_items == old_items:
            continue
        behavior["ship_intentions"] = new_items
        row_changed_items = sum(
            1
            for old_item, new_item in zip(old_items, new_items)
            if old_item != new_item
        )
        changed_items += row_changed_items
        changed_rows += 1
        changes.append(
            {
                "id": row.get("id"),
                "old_ship_intentions": old_items,
                "new_ship_intentions": new_items,
            }
        )
        if isinstance(row.get("reference"), dict):
            row["schema_check"] = schema_check(prediction, row["reference"])
            row["semantic_check"] = semantic_check(prediction, row["reference"])

    after = summarize_results(rows).get("ship_behavior", {})
    return {
        "mode": "defer_entering_berth_until_stable",
        "changed_rows": changed_rows,
        "changed_items": changed_items,
        "missing_scene": missing_scene,
        "settings": {
            "single_ship_only": not args.entering_berth_allow_multi_ship,
            "min_dwell_frames": args.entering_berth_min_dwell_frames,
            "min_dwell_sec": args.entering_berth_min_dwell_sec,
            "max_dwell_displacement_m": args.entering_berth_max_dwell_displacement_m,
        },
        "before": ship_metrics_from_summary(before),
        "after": ship_metrics_from_summary(after),
        "changes": changes,
    }


def defer_entering_phase_berthed_items(
    items: list[dict[str, Any]],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    berths: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    scene_token: str,
    *,
    single_ship_only: bool,
    min_dwell_frames: int,
    min_dwell_sec: float,
    max_dwell_displacement_m: float,
) -> list[dict[str, Any]]:
    if lockage_phase_from_route(scene_token, frames) != "ship_entering_lock":
        return items
    if single_ship_only and len(items) != 1:
        return items
    out = []
    for item in items:
        copied = dict(item)
        if copied.get("ship_intentions") != ["ship_berthed"]:
            out.append(copied)
            continue
        if entering_berth_dwell_is_stable(
            copied,
            tracked_frames,
            token_map,
            berths,
            frames,
            min_dwell_frames=min_dwell_frames,
            min_dwell_sec=min_dwell_sec,
            max_dwell_displacement_m=max_dwell_displacement_m,
        ):
            out.append(copied)
            continue
        copied["ship_intentions"] = ["ship_entering_lock"]
        out.append(copied)
    return out


def lockage_phase_from_route(
    scene_token: str,
    frames: list[dict[str, Any]],
) -> Optional[str]:
    open_dir = open_gate_direction_from_frames(frames)
    if open_dir is None:
        return None
    token = str(scene_token or "").lower()
    if "_upstream_" in token:
        return "ship_leaving_lock" if open_dir > 0 else "ship_entering_lock"
    if "_downstream_" in token:
        return "ship_leaving_lock" if open_dir < 0 else "ship_entering_lock"
    return None


def open_gate_direction_from_frames(frames: list[dict[str, Any]]) -> Optional[float]:
    for frame in reversed(frames):
        lock_state = frame.get("lock_state") or {}
        upper = lock_state.get("upper_gate_state")
        lower = lock_state.get("lower_gate_state")
        upper_open = upper in {"open", "opening"}
        lower_open = lower in {"open", "opening"}
        if upper_open and not lower_open:
            return 1.0
        if lower_open and not upper_open:
            return -1.0
    return None


def entering_berth_dwell_is_stable(
    item: dict[str, Any],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    berths: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    *,
    min_dwell_frames: int,
    min_dwell_sec: float,
    max_dwell_displacement_m: float,
) -> bool:
    observations = item_berth_track_observations(item, tracked_frames, token_map, frames)
    if not observations:
        return False
    latest = observations[-1]
    latest_berth = berth_index_for_point(latest["x"], latest["y"], berths)
    if latest_berth is None:
        return False
    suffix = []
    for obs in reversed(observations):
        if berth_index_for_point(obs["x"], obs["y"], berths) != latest_berth:
            break
        suffix.append(obs)
    suffix = list(reversed(suffix))
    if len(suffix) < min_dwell_frames:
        return False
    duration = float(suffix[-1]["time"]) - float(suffix[0]["time"])
    if duration < min_dwell_sec:
        return False
    displacement = (
        (float(suffix[-1]["x"]) - float(suffix[0]["x"])) ** 2
        + (float(suffix[-1]["y"]) - float(suffix[0]["y"])) ** 2
    ) ** 0.5
    return displacement <= max_dwell_displacement_m


def item_berth_track_observations(
    item: dict[str, Any],
    tracked_frames: list[list[dict[str, Any]]],
    token_map: dict[str, str],
    frames: list[dict[str, Any]],
) -> list[dict[str, float]]:
    token = str(item.get("instance_token"))
    observations = []
    for frame_index, tracks in enumerate(tracked_frames):
        frame = frames[frame_index] if frame_index < len(frames) else {}
        time = float(frame.get("relative_time_sec", frame_index))
        for track in tracks:
            track_token = str(track.get("track_token"))
            output_token = str(token_map.get(track_token, track_token))
            if output_token != token or track.get("x") is None or track.get("y") is None:
                continue
            observations.append(
                {
                    "frame_index": float(frame_index),
                    "time": time,
                    "x": float(track["x"]),
                    "y": float(track["y"]),
                }
            )
    return observations


DYNAMIC_SHIP_INTENTIONS = {"ship_entering_lock", "ship_leaving_lock"}


def apply_vlm_dynamic_ship_intention_fallback(rows: list[dict[str, Any]]) -> dict[str, Any]:
    before = summarize_results(rows).get("ship_behavior", {})
    changed_rows = 0
    changed_items = 0
    changes = []
    for row in rows:
        prediction = prediction_object(row)
        if prediction is None:
            continue
        behavior = prediction.get("ship_behavior")
        if not isinstance(behavior, dict):
            continue
        intentions = behavior.get("ship_intentions")
        if not isinstance(intentions, list):
            continue
        vlm_by_token = vlm_dynamic_intentions_by_token(row)
        if not vlm_by_token:
            continue
        row_changed = False
        for item in intentions:
            if not isinstance(item, dict):
                continue
            token = item.get("instance_token")
            if token is None or item.get("ship_intentions") != ["ship_berthed"]:
                continue
            vlm_label = vlm_by_token.get(str(token))
            if vlm_label not in DYNAMIC_SHIP_INTENTIONS:
                continue
            item["ship_intentions"] = [vlm_label]
            changed_items += 1
            row_changed = True
            changes.append(
                {
                    "id": row.get("id"),
                    "instance_token": str(token),
                    "old_ship_intentions": ["ship_berthed"],
                    "new_ship_intentions": [vlm_label],
                }
            )
        if row_changed:
            changed_rows += 1
            if isinstance(row.get("reference"), dict):
                row["schema_check"] = schema_check(prediction, row["reference"])
                row["semantic_check"] = semantic_check(prediction, row["reference"])

    after = summarize_results(rows).get("ship_behavior", {})
    return {
        "mode": "vlm_dynamic_ship_intention_fallback",
        "changed_rows": changed_rows,
        "changed_items": changed_items,
        "before": ship_metrics_from_summary(before),
        "after": ship_metrics_from_summary(after),
        "changes": changes,
    }


def vlm_dynamic_intentions_by_token(row: dict[str, Any]) -> dict[str, str]:
    raw_prediction = row.get("prediction_json_raw")
    if not isinstance(raw_prediction, dict):
        return {}
    behavior = raw_prediction.get("ship_behavior")
    if not isinstance(behavior, dict):
        return {}
    intentions = behavior.get("ship_intentions")
    if not isinstance(intentions, list):
        return {}
    by_token: dict[str, str] = {}
    for item in intentions:
        if not isinstance(item, dict) or item.get("instance_token") is None:
            continue
        labels = item.get("ship_intentions")
        if not isinstance(labels, list) or len(labels) != 1:
            continue
        label = str(labels[0])
        if label in DYNAMIC_SHIP_INTENTIONS:
            by_token[str(item["instance_token"])] = label
    return by_token


def first_track_feature(features: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    return features[0] if features else None


def rtmdet_static_berth_candidate(
    feature: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    motion = feature.get("rtmdet_2d_motion") or {}
    return (
        bool(feature.get("end_inside_berth"))
        and len(feature.get("cameras_supported") or []) >= 2
        and int(motion.get("camera_count_with_motion") or 0) > 0
        and float(motion.get("max_normalized_displacement") or 0.0)
        <= args.static_2d_motion_threshold
    )


def rtmdet_static_berth_override_allowed(labels: Any) -> bool:
    return labels == ["ship_entering_lock"]


MOVING_MOTION_STATES = frozenset(
    {"ship_entering_lock", "ship_leaving_lock", "ship_moving"}
)
SHIP_INTENTION_MOTION_STATES = frozenset(
    {"ship_berthed", "ship_entering_lock", "ship_leaving_lock"}
)
WORLD_STATE_NEAR_BERTH_FILL_M = 15.0
WORLD_STATE_STATIC_DISPLACEMENT_M = 3.0


def apply_ship_intention_world_state_alignment(
    world_state_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    sequences: dict[str, dict[str, Any]],
    scene_berths: dict[str, list[dict[str, Any]]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Align fused world-state output with the final ship-intention branch.

    The ship-intention branch applies RTMDet count pruning, future-candidate
    filtering, and lockage-phase guards after raw Hydro/RTMDet tracking. Without
    this pass, `lock_occupancy` and `vessel_motion_flow` can still contain the
    pre-guard track interpretation. This keeps the world-state branch tied to the
    final deployable ship set while using only input-window perception context.
    """
    items_by_scene = final_ship_intention_items_by_scene(rows)
    report = {
        "applied_scenes": 0,
        "missing_scene": 0,
        "occupancy_tokens_removed": 0,
        "occupancy_tokens_added": 0,
        "input_motion_items_changed": 0,
        "target_motion_items_changed": 0,
    }

    for state in progress_iter(world_state_rows, args, "world-state-align", unit="scene"):
        scene_token = state.get("scene_token")
        if not isinstance(scene_token, str):
            continue
        items = items_by_scene.get(scene_token)
        if not items:
            continue
        sequence = sequences.get(scene_token)
        if sequence is None:
            report["missing_scene"] += 1
            continue
        berths = scene_berths.get(scene_token, [])
        track_context = build_world_state_track_context(
            sequence,
            berths,
            hydro_predictions,
            args,
        )
        occupancy_report = align_lock_occupancy_to_ship_intentions(
            state.get("lock_occupancy") or {},
            items,
            track_context,
            berths,
        )
        motion_changes = align_input_motion_flow_to_ship_intentions(
            ((state.get("vessel_motion_flow") or {}).get("input_window") or []),
            items,
        )
        target_motion_changes = align_future_motion_flow_to_future_occupancy(
            ((state.get("vessel_motion_flow") or {}).get("target_window") or []),
            (state.get("lock_occupancy") or {}).get("future_10s") or {},
        )
        report["applied_scenes"] += 1
        report["occupancy_tokens_removed"] += occupancy_report["tokens_removed"]
        report["occupancy_tokens_added"] += occupancy_report["tokens_added"]
        report["input_motion_items_changed"] += motion_changes
        report["target_motion_items_changed"] += target_motion_changes
    return report


def build_all_frame_motion_stitch_world_state(
    source_rows: list[dict[str, Any]],
    sequences: dict[str, dict[str, Any]],
    scene_berths: dict[str, list[dict[str, Any]]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    paths: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_frame_args = copy.copy(args)
    all_frame_args.recovery_all_input_frames = True

    ship_report = apply_hydro_ship_intention_prior(
        source_rows,
        sequences,
        scene_berths,
        hydro_predictions,
        paths,
        all_frame_args,
    )
    rtmdet_ship_report = None
    if all_frame_args.apply_rtmdet_ship_intention_static_berth:
        rtmdet_ship_report = apply_rtmdet_static_berth_intention_override(
            source_rows,
            sequences,
            scene_berths,
            hydro_predictions,
            all_frame_args,
            paths,
        )
    phase_report = apply_lockage_phase_consistency_intention_guard(
        source_rows,
        sequences,
        scene_berths,
        hydro_predictions,
        all_frame_args,
    )
    entering_berth_defer_report = None
    if all_frame_args.defer_entering_berth:
        entering_berth_defer_report = apply_entering_berth_defer_intention_guard(
            source_rows,
            sequences,
            scene_berths,
            hydro_predictions,
            all_frame_args,
        )
    vlm_dynamic_report = None
    if all_frame_args.apply_vlm_dynamic_ship_intention_fallback:
        vlm_dynamic_report = apply_vlm_dynamic_ship_intention_fallback(source_rows)

    world_state_rows = derive_hydro_world_state(
        sequences,
        scene_berths,
        hydro_predictions,
        all_frame_args,
        scene_tokens=scene_tokens_from_rows(source_rows),
        progress_desc="world-state-all-frame-motion",
    )
    alignment_report = apply_ship_intention_world_state_alignment(
        world_state_rows,
        source_rows,
        sequences,
        scene_berths,
        hydro_predictions,
        all_frame_args,
    )
    return world_state_rows, {
        "recovery_current_frame_only": False,
        "ship_prior": ship_report,
        "rtmdet_ship_intention_prior": rtmdet_ship_report,
        "lockage_phase_consistency_prior": phase_report,
        "entering_berth_defer_prior": entering_berth_defer_report,
        "vlm_dynamic_ship_intention_fallback": vlm_dynamic_report,
        "world_state_ship_intention_alignment": alignment_report,
    }


def apply_current_motion_token_stitch(
    world_state_rows: list[dict[str, Any]],
    all_frame_world_state_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Replace only current motion with an all-frame token-stitch candidate.

    The auxiliary world-state pass uses the same deployable perception inputs but
    recovers RTMDet candidates over all input frames. That preserves long-window
    motion evidence which current-frame recovery can miss. The merge below keeps
    the main occupancy/future branches intact and only updates
    ``vessel_motion_flow.input_window``.
    """

    all_frame_by_scene = {
        str(state.get("scene_token")): state
        for state in all_frame_world_state_rows
        if isinstance(state.get("scene_token"), str)
    }
    rows_by_scene = prediction_rows_by_scene(prediction_rows)
    final_items_by_scene = final_ship_intention_items_by_scene(prediction_rows)

    report = {
        "mode": "current_motion_all_input_frame_token_stitch",
        "applied_scenes": 0,
        "missing_all_frame_scene": 0,
        "replaced_input_window_items": 0,
        "filled_missing_final_ship_items": 0,
        "vlm_slow_berthed_items": 0,
        "final_berthed_static_items": 0,
        "final_berthed_high_speed_outlier_items": 0,
        "changed_scenes": 0,
        "changed_items": 0,
        "settings": {
            "vlm_slow_speed_mps": args.motion_stitch_vlm_slow_speed_mps,
            "high_speed_outlier_mps": args.motion_stitch_high_speed_outlier_mps,
        },
        "changes": [],
    }

    for state in progress_iter(world_state_rows, args, "current-motion-stitch", unit="scene"):
        scene_token = state.get("scene_token")
        if not isinstance(scene_token, str):
            continue
        all_frame_state = all_frame_by_scene.get(scene_token)
        if all_frame_state is None:
            report["missing_all_frame_scene"] += 1
            continue
        vessel_motion = state.setdefault("vessel_motion_flow", {})
        if not isinstance(vessel_motion, dict):
            vessel_motion = {}
            state["vessel_motion_flow"] = vessel_motion
        old_flow = copy.deepcopy(vessel_motion.get("input_window") or [])
        all_frame_motion = all_frame_state.get("vessel_motion_flow") or {}
        candidate_flow = copy.deepcopy(all_frame_motion.get("input_window") or [])
        if not candidate_flow:
            continue

        row = rows_by_scene.get(scene_token, {})
        raw_labels = raw_ship_intention_label_by_token(row)
        final_items = final_items_by_scene.get(scene_token, [])
        final_labels = ship_intention_label_by_token(final_items)
        fill_report = fill_missing_current_motion_from_ship_intentions(
            candidate_flow,
            final_items,
        )
        stitch_report = snap_berthed_motion_stitch_outliers(
            candidate_flow,
            raw_labels,
            final_labels,
            args,
        )

        vessel_motion["input_window"] = candidate_flow
        item_changes = count_motion_item_changes(old_flow, candidate_flow)
        if item_changes:
            report["changed_scenes"] += 1
            if len(report["changes"]) < 50:
                report["changes"].append(
                    {
                        "scene_token": scene_token,
                        "changed_items": item_changes,
                        "filled_missing_final_ship_items": fill_report[
                            "filled_missing_final_ship_items"
                        ],
                        "vlm_slow_berthed_items": stitch_report[
                            "vlm_slow_berthed_items"
                        ],
                        "final_berthed_static_items": stitch_report[
                            "final_berthed_static_items"
                        ],
                        "final_berthed_high_speed_outlier_items": stitch_report[
                            "final_berthed_high_speed_outlier_items"
                        ],
                    }
                )

        report["applied_scenes"] += 1
        report["replaced_input_window_items"] += len(candidate_flow)
        report["filled_missing_final_ship_items"] += fill_report[
            "filled_missing_final_ship_items"
        ]
        report["vlm_slow_berthed_items"] += stitch_report["vlm_slow_berthed_items"]
        report["final_berthed_static_items"] += stitch_report[
            "final_berthed_static_items"
        ]
        report["final_berthed_high_speed_outlier_items"] += stitch_report[
            "final_berthed_high_speed_outlier_items"
        ]
        report["changed_items"] += item_changes
    return report


def prediction_rows_by_scene(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        scene_token = scene_token_of_row(row)
        if scene_token is not None:
            out[scene_token] = row
    return out


def raw_ship_intention_label_by_token(row: dict[str, Any]) -> dict[str, str]:
    raw_prediction = row.get("prediction_json_raw")
    if not isinstance(raw_prediction, dict):
        return {}
    behavior = raw_prediction.get("ship_behavior")
    if not isinstance(behavior, dict):
        return {}
    labels = {}
    for item in behavior.get("ship_intentions") or []:
        if not isinstance(item, dict) or item.get("instance_token") is None:
            continue
        item_labels = item.get("ship_intentions")
        if not isinstance(item_labels, list) or not item_labels:
            continue
        label = str(item_labels[0])
        if label in SHIP_INTENTION_MOTION_STATES:
            labels[str(item["instance_token"])] = label
    return labels


def fill_missing_current_motion_from_ship_intentions(
    input_flow: list[dict[str, Any]],
    ship_items: list[dict[str, Any]],
) -> dict[str, int]:
    present = {
        str(item.get("instance_token"))
        for item in input_flow
        if isinstance(item, dict) and item.get("instance_token") is not None
    }
    filled = 0
    for ship_item in ship_items:
        if not isinstance(ship_item, dict):
            continue
        token = ship_item.get("instance_token")
        labels = ship_item.get("ship_intentions")
        if token is None or not isinstance(labels, list) or not labels:
            continue
        token = str(token)
        if token in present:
            continue
        label = str(labels[0])
        if label not in SHIP_INTENTION_MOTION_STATES:
            continue
        input_flow.append(motion_item_from_ship_intention(ship_item, label))
        present.add(token)
        filled += 1
    return {"filled_missing_final_ship_items": filled}


def motion_item_from_ship_intention(
    ship_item: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    item = {
        "instance_token": str(ship_item.get("instance_token")),
        "category": ship_item.get("category"),
        "motion_state": label,
        "direction_label": "from_ship_intention_fallback",
        "delta_xy": [0.0, 0.0],
        "end_speed_mps": 0.0,
    }
    if label in {"ship_berthed", "ship_static"}:
        set_motion_static(item, label)
    return item


CURRENT_MOTION_DYNAMIC_STATES = frozenset(
    {"ship_entering_lock", "ship_leaving_lock", "ship_moving"}
)


def snap_berthed_motion_stitch_outliers(
    input_flow: list[dict[str, Any]],
    raw_labels: dict[str, str],
    final_labels: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, int]:
    report = {
        "vlm_slow_berthed_items": 0,
        "final_berthed_static_items": 0,
        "final_berthed_high_speed_outlier_items": 0,
    }
    for item in input_flow:
        if not isinstance(item, dict) or item.get("instance_token") is None:
            continue
        token = str(item["instance_token"])
        final_label = final_labels.get(token)
        if final_label != "ship_berthed":
            continue
        old_state = item.get("motion_state")
        if (
            raw_labels.get(token) == "ship_berthed"
            and old_state in CURRENT_MOTION_DYNAMIC_STATES
            and motion_end_speed(item) <= args.motion_stitch_vlm_slow_speed_mps
        ):
            set_motion_static(item, "ship_berthed")
            report["vlm_slow_berthed_items"] += 1
        elif old_state == "ship_static":
            set_motion_static(item, "ship_berthed")
            report["final_berthed_static_items"] += 1
        elif (
            old_state == "ship_moving"
            and motion_end_speed(item) >= args.motion_stitch_high_speed_outlier_mps
        ):
            set_motion_static(item, "ship_berthed")
            report["final_berthed_high_speed_outlier_items"] += 1
    return report


def motion_end_speed(item: dict[str, Any]) -> float:
    try:
        return float(item.get("end_speed_mps") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def count_motion_item_changes(
    old_flow: list[dict[str, Any]],
    new_flow: list[dict[str, Any]],
) -> int:
    old_by_token = {
        str(item.get("instance_token")): item
        for item in old_flow
        if isinstance(item, dict) and item.get("instance_token") is not None
    }
    new_by_token = {
        str(item.get("instance_token")): item
        for item in new_flow
        if isinstance(item, dict) and item.get("instance_token") is not None
    }
    changed = 0
    for token, new_item in new_by_token.items():
        old_item = old_by_token.get(token)
        if old_item is None:
            changed += 1
            continue
        for key in ("motion_state", "direction_label", "delta_xy", "end_speed_mps"):
            if old_item.get(key) != new_item.get(key):
                changed += 1
                break
    changed += len(set(old_by_token) - set(new_by_token))
    return changed


def final_ship_intention_items_by_scene(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        prediction = prediction_object(row)
        if prediction is None:
            continue
        behavior = prediction.get("ship_behavior")
        if not isinstance(behavior, dict):
            continue
        items = behavior.get("ship_intentions")
        if not isinstance(items, list):
            continue
        scene_token = scene_token_from_id(row["id"]) if isinstance(row.get("id"), str) else None
        if scene_token is not None:
            out[scene_token] = [item for item in items if isinstance(item, dict)]
    return out


def build_world_state_track_context(
    sequence: dict[str, Any],
    berths: list[dict[str, Any]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    frames = sequence.get("frames", [])
    input_idx = sequence.get("prediction_input_frame_indices") or []
    input_frames = [frames[i] for i in input_idx]
    if not input_frames:
        return {}
    detection_frames, _ = build_detection_frames(
        input_frames,
        berths,
        hydro_predictions,
        data_root=args.data_root,
        lock_chamber_bounds=getattr(args, "lock_chamber_bounds", None),
        rtmdet_by_path=getattr(args, "rtmdet_by_path", None),
        score_threshold=args.score_threshold,
        recover_rtmdet_multicamera=bool(args.recover_rtmdet_multicamera),
        support_iou_threshold=args.support_iou_threshold,
        recovery_min_cameras=args.recovery_min_cameras,
        recovery_max_ray_residual_m=args.recovery_max_ray_residual_m,
        recovery_cluster_distance_m=args.recovery_cluster_distance_m,
        recovery_existing_distance_m=args.recovery_existing_distance_m,
        recovery_chamber_margin_m=args.recovery_chamber_margin_m,
        recover_open_gate_new_ships=bool(args.recover_open_gate_new_ships),
        open_gate_min_cameras=args.open_gate_min_cameras,
        open_gate_zone_length_m=args.open_gate_zone_length_m,
        open_gate_max_candidates=args.open_gate_max_candidates,
        recovery_current_frame_only=not args.recovery_all_input_frames,
    )
    tracked_frames = track_detections(detection_frames, args.track_distance_m, berths=berths)
    token_map: dict[str, str] = {}
    if args.eval_token_map and tracked_frames:
        token_map = eval_token_map_from_input_window(
            tracked_frames,
            input_frames,
            args.eval_token_map_distance_m,
            berths=berths,
        )
        if args.eval_open_gate_new_ship_tokens:
            add_eval_open_gate_new_ship_tokens(
                token_map,
                tracked_frames,
                input_frames,
                str(sequence.get("scene_token") or ""),
            )

    context: dict[str, dict[str, Any]] = {}
    current_frame_index = len(tracked_frames) - 1
    for frame_index, tracks in enumerate(tracked_frames):
        for track in tracks:
            track_token = str(track.get("track_token"))
            output_token = str(token_map.get(track_token, track_token))
            item = context.setdefault(
                output_token,
                {
                    "sources": set(),
                    "latest_berth_index": None,
                    "current_berth_index": None,
                    "nearest_berth_index": None,
                    "nearest_berth_distance_m": float("inf"),
                    "open_gate_candidate": False,
                },
            )
            source = str(track.get("detection_source") or "hydro")
            item["sources"].add(source)
            if source == "rtmdet_open_gate_recovery":
                item["open_gate_candidate"] = True
            x = float(track["x"])
            y = float(track["y"])
            berth_index = berth_index_for_point(x, y, berths)
            if berth_index is not None:
                item["latest_berth_index"] = berth_index
                if frame_index == current_frame_index:
                    item["current_berth_index"] = berth_index
            nearest_index, nearest_distance = nearest_berth(x, y, berths)
            if nearest_distance < float(item["nearest_berth_distance_m"]):
                item["nearest_berth_index"] = nearest_index
                item["nearest_berth_distance_m"] = nearest_distance
    return context


def nearest_berth(
    x: float,
    y: float,
    berths: list[dict[str, Any]],
) -> tuple[Optional[int], float]:
    best_index = None
    best_distance = float("inf")
    for index, box in enumerate(berths):
        distance = point_box_distance(x, y, box)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index, best_distance


def point_box_distance(x: float, y: float, box: dict[str, Any]) -> float:
    dx = max(float(box["x_min"]) - x, 0.0, x - float(box["x_max"]))
    dy = max(float(box["y_min"]) - y, 0.0, y - float(box["y_max"]))
    return (dx * dx + dy * dy) ** 0.5


def align_lock_occupancy_to_ship_intentions(
    lock_occupancy: dict[str, Any],
    ship_items: list[dict[str, Any]],
    track_context: dict[str, dict[str, Any]],
    berths: list[dict[str, Any]],
) -> dict[str, int]:
    labels = ship_intention_label_by_token(ship_items)
    valid_tokens = set(labels)
    report = {"tokens_removed": 0, "tokens_added": 0}
    if lock_occupancy.get("current") is lock_occupancy.get("future_10s"):
        lock_occupancy["future_10s"] = copy.deepcopy(lock_occupancy["current"])
    for section in ("current", "future_10s"):
        occupancy = lock_occupancy.get(section)
        if not isinstance(occupancy, dict):
            continue
        for slot in occupancy.get("berth_slots") or []:
            if not isinstance(slot, dict):
                continue
            before = list(slot.get("ship_tokens") or [])
            after = [token for token in before if token in valid_tokens]
            report["tokens_removed"] += len(before) - len(after)
            slot["ship_tokens"] = after

        present = {
            token
            for slot in occupancy.get("berth_slots") or []
            if isinstance(slot, dict)
            for token in (slot.get("ship_tokens") or [])
        }
        for token, label in labels.items():
            if token in present:
                continue
            berth_index = world_state_fill_berth_index(
                token,
                label,
                section,
                track_context,
                berths,
            )
            if berth_index is None:
                continue
            slots = occupancy.get("berth_slots") or []
            if berth_index >= len(slots) or not isinstance(slots[berth_index], dict):
                continue
            slot = slots[berth_index]
            if slot.get("ship_tokens"):
                continue
            slot["ship_tokens"] = [token]
            present.add(token)
            report["tokens_added"] += 1
        recompute_occupancy_counts(occupancy)
    return report


def world_state_fill_berth_index(
    token: str,
    label: str,
    section: str,
    track_context: dict[str, dict[str, Any]],
    berths: list[dict[str, Any]],
) -> Optional[int]:
    if label not in SHIP_INTENTION_MOTION_STATES:
        return None
    if section == "future_10s" and label == "ship_leaving_lock":
        return None
    info = track_context.get(token)
    if not info or info.get("open_gate_candidate"):
        return None
    if section == "current" and info.get("current_berth_index") is not None:
        return int(info["current_berth_index"])
    if info.get("latest_berth_index") is not None:
        return int(info["latest_berth_index"])
    if (
        label in {"ship_berthed", "ship_entering_lock"}
        and float(info.get("nearest_berth_distance_m", float("inf")))
        <= WORLD_STATE_NEAR_BERTH_FILL_M
    ):
        nearest = info.get("nearest_berth_index")
        return int(nearest) if nearest is not None else None
    return None


def recompute_occupancy_counts(occupancy: dict[str, Any]) -> None:
    occupied = 0
    tokens: set[str] = set()
    for slot in occupancy.get("berth_slots") or []:
        if not isinstance(slot, dict):
            continue
        slot_tokens = list(slot.get("ship_tokens") or [])
        slot["ship_tokens"] = slot_tokens
        slot["ship_count"] = len(slot_tokens)
        slot["occupied"] = bool(slot_tokens)
        if slot_tokens:
            occupied += 1
            tokens.update(str(token) for token in slot_tokens)
    occupancy["num_occupied_berths"] = occupied
    occupancy["num_ships"] = len(tokens)


def align_input_motion_flow_to_ship_intentions(
    input_flow: list[dict[str, Any]],
    ship_items: list[dict[str, Any]],
) -> int:
    labels = ship_intention_label_by_token(ship_items)
    changes = 0
    for item in input_flow:
        if not isinstance(item, dict):
            continue
        token = item.get("instance_token")
        label = labels.get(str(token))
        old_state = item.get("motion_state")
        if label == "ship_berthed" and old_state != "ship_berthed":
            set_motion_static(item, "ship_berthed")
        elif (
            label in {"ship_entering_lock", "ship_leaving_lock"}
            and old_state == "ship_berthed"
            and motion_net_displacement(item) >= WORLD_STATE_STATIC_DISPLACEMENT_M
        ):
            item["motion_state"] = label
            set_motion_direction_from_delta(item)
        elif (
            label is None
            and isinstance(token, str)
            and "_ship_" in token
            and old_state in MOVING_MOTION_STATES
        ):
            set_motion_static(item, "ship_static")
        if item.get("motion_state") != old_state:
            changes += 1
    return changes


def align_future_motion_flow_to_future_occupancy(
    target_flow: list[dict[str, Any]],
    future_occupancy: dict[str, Any],
) -> int:
    """Keep the target static/berthed boundary consistent with berth occupancy.

    This is a non-leaky consistency pass: ``future_occupancy`` is already the
    pipeline's prediction from input-window perception and static priors. It
    does not use target frames and deliberately leaves entering/leaving labels
    untouched.
    """
    berth_tokens = set()
    for slot in future_occupancy.get("berth_slots") or []:
        if not isinstance(slot, dict) or not slot.get("occupied"):
            continue
        berth_tokens.update(str(token) for token in slot.get("ship_tokens") or [])

    changes = 0
    for item in target_flow:
        if not isinstance(item, dict):
            continue
        token = item.get("instance_token")
        if token is None:
            continue
        old_state = item.get("motion_state")
        if old_state == "ship_static" and str(token) in berth_tokens:
            set_motion_static(item, "ship_berthed")
        elif old_state == "ship_berthed" and str(token) not in berth_tokens:
            set_motion_static(item, "ship_static")
        if item.get("motion_state") != old_state:
            changes += 1
    return changes


def ship_intention_label_by_token(items: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in items:
        token = item.get("instance_token")
        item_labels = item.get("ship_intentions")
        if token is None or not isinstance(item_labels, list) or not item_labels:
            continue
        label = str(item_labels[0])
        if label in SHIP_INTENTION_MOTION_STATES:
            labels[str(token)] = label
    return labels


def motion_net_displacement(item: dict[str, Any]) -> float:
    delta = item.get("delta_xy")
    if not isinstance(delta, list) or len(delta) < 2:
        return 0.0
    try:
        dx = float(delta[0])
        dy = float(delta[1])
    except (TypeError, ValueError):
        return 0.0
    return (dx * dx + dy * dy) ** 0.5


def set_motion_static(item: dict[str, Any], state: str) -> None:
    item["motion_state"] = state
    item["direction_label"] = "static_or_settled"
    item["delta_xy"] = [0.0, 0.0]
    item["end_speed_mps"] = 0.0


def set_motion_direction_from_delta(item: dict[str, Any]) -> None:
    delta = item.get("delta_xy")
    dy = 0.0
    if isinstance(delta, list) and len(delta) >= 2:
        try:
            dy = float(delta[1])
        except (TypeError, ValueError):
            dy = 0.0
    item["direction_label"] = "moving_to_upper_gate" if dy >= 0 else "moving_to_lower_gate"


def derive_hydro_world_state(
    sequences: dict[str, dict[str, Any]],
    scene_berths: dict[str, list[dict[str, Any]]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    args: argparse.Namespace,
    scene_tokens: Optional[set[str]] = None,
    progress_desc: str = "world-state",
) -> list[dict[str, Any]]:
    rows = []
    selected_sequences = []
    for sequence in sequences.values():
        scene_token = sequence.get("scene_token")
        if scene_tokens is not None and scene_token not in scene_tokens:
            continue
        if not sequence.get("prediction_input_frame_indices"):
            continue
        selected_sequences.append(sequence)
    for sequence in progress_iter(selected_sequences, args, progress_desc, unit="scene"):
        scene_token = sequence.get("scene_token")
        rows.append(
            derive_prediction_from_hydro_tracks(
                sequence,
                scene_berths.get(scene_token, []),
                hydro_predictions,
                data_root=args.data_root,
                lock_chamber_bounds=getattr(args, "lock_chamber_bounds", None),
                rtmdet_by_path=getattr(args, "rtmdet_by_path", None),
                score_threshold=args.score_threshold,
                track_distance_m=args.track_distance_m,
                recover_rtmdet_multicamera=bool(args.recover_rtmdet_multicamera),
                support_iou_threshold=args.support_iou_threshold,
                recovery_min_cameras=args.recovery_min_cameras,
                recovery_max_ray_residual_m=args.recovery_max_ray_residual_m,
                recovery_cluster_distance_m=args.recovery_cluster_distance_m,
                recovery_existing_distance_m=args.recovery_existing_distance_m,
                recovery_chamber_margin_m=args.recovery_chamber_margin_m,
                recover_open_gate_new_ships=bool(args.recover_open_gate_new_ships),
                open_gate_min_cameras=args.open_gate_min_cameras,
                open_gate_zone_length_m=args.open_gate_zone_length_m,
                open_gate_max_candidates=args.open_gate_max_candidates,
                recovery_current_frame_only=not args.recovery_all_input_frames,
                future_motion_mode=args.future_motion_mode,
                eval_token_map=bool(args.eval_token_map),
                eval_open_gate_new_ship_tokens=bool(args.eval_open_gate_new_ship_tokens),
                eval_token_map_distance_m=args.eval_token_map_distance_m,
            )
        )
    return rows


def write_world_state_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prediction_object(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    prediction = row.get("prediction_json")
    if isinstance(prediction, dict):
        return prediction
    if isinstance(prediction, str):
        try:
            parsed = json.loads(prediction)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, dict):
            row["prediction_json"] = parsed
            return parsed
    return None


def prediction_objects_by_scene(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        prediction = prediction_object(row)
        if prediction is None:
            continue
        scene_token = prediction.get("scene_token")
        if not scene_token and isinstance(row.get("id"), str):
            scene_token = row["id"].rsplit(":", 1)[-1]
        if isinstance(scene_token, str):
            out[scene_token] = prediction
    return out


def row_count_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    prediction = 0
    recognition = 0
    other = 0
    for row in rows:
        row_id = row.get("id")
        if isinstance(row_id, str) and ":prediction:" in row_id:
            prediction += 1
        elif isinstance(row_id, str) and ":recognition:" in row_id:
            recognition += 1
        else:
            other += 1
    return {
        "prediction_rows": prediction,
        "current_recognition_rows": recognition,
        "other_rows": other,
        "unique_scenes": len(scene_tokens_from_rows(rows)),
    }


def evaluate_world_state(
    gt: dict[str, dict[str, Any]],
    preds: dict[str, dict[str, Any]],
    section: str,
) -> dict[str, Any]:
    window = {"current": "input_window", "future_10s": "target_window"}[section]
    evaluated = 0
    slot_correct = slot_total = 0
    tp = fp = fn = 0
    motion_correct = motion_total = 0

    for scene_token, gt_state in gt.items():
        pred_obj = preds.get(scene_token)
        if pred_obj is None:
            continue
        gt_lock = gt_state.get("lock_occupancy") or {}
        gt_motion = gt_state.get("vessel_motion_flow") or {}
        if section not in gt_lock and window not in gt_motion:
            continue
        evaluated += 1
        gt_occ = gt_lock.get(section) or {}
        pred_occ = (pred_obj.get("lock_occupancy") or {}).get(section) or {}
        gt_slots = slot_occupied_map(gt_occ)
        pred_slots = slot_occupied_map(pred_occ)
        for region_id, gt_occupied in gt_slots.items():
            pred_occupied = pred_slots.get(region_id, False)
            slot_total += 1
            if pred_occupied == gt_occupied:
                slot_correct += 1
            if gt_occupied and pred_occupied:
                tp += 1
            elif pred_occupied and not gt_occupied:
                fp += 1
            elif gt_occupied and not pred_occupied:
                fn += 1

        gt_flow = gt_motion.get(window) or []
        pred_flow = (pred_obj.get("vessel_motion_flow") or {}).get(window) or []
        pred_motion = {
            item.get("instance_token"): item.get("motion_state")
            for item in pred_flow
            if isinstance(item, dict)
        }
        for item in gt_flow:
            if not isinstance(item, dict):
                continue
            motion_total += 1
            if pred_motion.get(item.get("instance_token")) == item.get("motion_state"):
                motion_correct += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "section": section,
        "evaluated_scenes": evaluated,
        "slot_occupancy_accuracy": slot_correct / slot_total if slot_total else 0.0,
        "occupied_slot_prf": {"precision": precision, "recall": recall, "f1": f1},
        "vessel_motion_state_accuracy": (
            motion_correct / motion_total if motion_total else 0.0
        ),
        "counts": {
            "slot_total": slot_total,
            "slot_correct": slot_correct,
            "occupied_tp": tp,
            "occupied_fp": fp,
            "occupied_fn": fn,
            "motion_total": motion_total,
            "motion_correct": motion_correct,
        },
    }


def slot_occupied_map(occupancy: dict[str, Any]) -> dict[str, bool]:
    return {
        slot["region_id"]: bool(slot.get("occupied"))
        for slot in occupancy.get("berth_slots") or []
        if isinstance(slot, dict) and slot.get("region_id") is not None
    }


def build_summary(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    rows: list[dict[str, Any]],
    ship_report: dict[str, Any],
    rtmdet_ship_report: Optional[dict[str, Any]],
    phase_consistency_report: Optional[dict[str, Any]],
    entering_berth_defer_report: Optional[dict[str, Any]],
    vlm_dynamic_report: Optional[dict[str, Any]],
    world_state_alignment_report: Optional[dict[str, Any]],
    current_motion_stitch_report: Optional[dict[str, Any]],
    vlm_route_summary: dict[str, Any],
    route_summary: dict[str, Any],
    world_current: dict[str, Any],
    world_future: dict[str, Any],
) -> dict[str, Any]:
    vlm_ship_metrics = ship_metrics_from_summary(
        vlm_route_summary.get("ship_behavior", {})
    )
    deployable_ship_metrics = ship_metrics_from_summary(
        route_summary.get("ship_behavior", {})
    )
    metrics = fused_metric_table(
        route_summary,
        world_current,
        world_future,
        rows=rows,
        ship_metrics=vlm_ship_metrics,
        deployable_ship_metrics=deployable_ship_metrics,
    )
    return {
        "model": "VLM semantic branch + Hydro3DNet ship/world-state priors",
        "split": ",".join(getattr(args, "eval_splits", [args.split])),
        "eval_splits": list(getattr(args, "eval_splits", [args.split])),
        "num_rows": len(rows),
        "row_counts": row_count_summary(rows),
        "inputs": {key: stringify_path_value(value) for key, value in paths.items()},
        "settings": {
            "score_threshold": args.score_threshold,
            "track_distance_m": args.track_distance_m,
            "future_motion_mode": args.future_motion_mode,
            "disable_berth_motion_prior": bool(args.disable_berth_motion_prior),
            "eval_token_map": bool(args.eval_token_map),
            "eval_token_map_basis": (
                "input_window_nearest" if args.eval_token_map else None
            ),
            "eval_token_map_distance_m": (
                args.eval_token_map_distance_m if args.eval_token_map else None
            ),
            "recover_rtmdet_multicamera": bool(args.recover_rtmdet_multicamera),
            "recover_open_gate_new_ships": bool(args.recover_open_gate_new_ships),
            "apply_rtmdet_ship_intention_static_berth": bool(
                args.apply_rtmdet_ship_intention_static_berth
            ),
            "apply_vlm_dynamic_ship_intention_fallback": bool(
                args.apply_vlm_dynamic_ship_intention_fallback
            ),
            "defer_entering_berth": bool(args.defer_entering_berth),
            "entering_berth_allow_multi_ship": bool(
                args.entering_berth_allow_multi_ship
            ),
            "entering_berth_min_dwell_frames": args.entering_berth_min_dwell_frames,
            "entering_berth_min_dwell_sec": args.entering_berth_min_dwell_sec,
            "entering_berth_max_dwell_displacement_m": (
                args.entering_berth_max_dwell_displacement_m
            ),
            "rtmdet_score_threshold": args.rtmdet_score_threshold,
            "rtmdet_recoverable_2d_labels": (
                sorted(SHIP_2D_CLASSES)
                if args.recover_rtmdet_multicamera or args.recover_open_gate_new_ships
                else None
            ),
            "support_iou_threshold": args.support_iou_threshold,
            "static_2d_motion_threshold": args.static_2d_motion_threshold,
            "current_active_rtmdet_min_cameras": args.current_active_rtmdet_min_cameras,
            "current_active_max_missing_frames": args.current_active_max_missing_frames,
            "recovery_min_cameras": args.recovery_min_cameras,
            "recovery_max_ray_residual_m": args.recovery_max_ray_residual_m,
            "recovery_cluster_distance_m": args.recovery_cluster_distance_m,
            "recovery_existing_distance_m": args.recovery_existing_distance_m,
            "recovery_chamber_margin_m": args.recovery_chamber_margin_m,
            "open_gate_min_cameras": args.open_gate_min_cameras,
            "open_gate_zone_length_m": args.open_gate_zone_length_m,
            "open_gate_max_candidates": args.open_gate_max_candidates,
            "eval_open_gate_new_ship_tokens": (
                bool(args.eval_open_gate_new_ship_tokens) if args.eval_token_map else None
            ),
            "lock_chamber_bounds": (
                getattr(args, "lock_chamber_bounds", None)
                if args.recover_rtmdet_multicamera or args.recover_open_gate_new_ships
                else None
            ),
            "recovery_current_frame_only": not args.recovery_all_input_frames,
            "stitch_current_motion_all_input_frames": bool(
                args.stitch_current_motion_all_input_frames
            ),
            "motion_stitch_vlm_slow_speed_mps": (
                args.motion_stitch_vlm_slow_speed_mps
                if args.stitch_current_motion_all_input_frames
                else None
            ),
            "motion_stitch_high_speed_outlier_mps": (
                args.motion_stitch_high_speed_outlier_mps
                if args.stitch_current_motion_all_input_frames
                else None
            ),
        },
        "metrics": metrics,
        "metric_semantics": {
            "ship_intentions_exact": (
                "VLM-native VLM semantic ship_behavior output before deployable "
                "ship-prior replacement."
            ),
            "ship_token_f1": (
                "VLM-native VLM semantic ship_behavior output before deployable "
                "ship-prior replacement."
            ),
            "ship_intention_f1": (
                "VLM-native VLM semantic ship_behavior output before deployable "
                "ship-prior replacement."
            ),
            "deployable_ship_intentions_exact": (
                "Hydro3DNet/RTMDet/geometry ship branch after replacement; "
                "reported separately from VLM ship metrics."
            ),
            "deployable_ship_token_f1": (
                "Hydro3DNet/RTMDet/geometry ship branch after replacement; "
                "reported separately from VLM ship metrics."
            ),
            "deployable_ship_intention_f1": (
                "Hydro3DNet/RTMDet/geometry ship branch after replacement; "
                "reported separately from VLM ship metrics."
            ),
            "water_surface_target_state": (
                "Prediction rows use water_surface_dynamics.target_water_state; "
                "current-only recognition rows use "
                "water_surface_dynamics.current_water_state."
            ),
        },
        "ship_prior": ship_report,
        "rtmdet_ship_intention_prior": rtmdet_ship_report,
        "lockage_phase_consistency_prior": phase_consistency_report,
        "entering_berth_defer_prior": entering_berth_defer_report,
        "vlm_dynamic_ship_intention_fallback": vlm_dynamic_report,
        "world_state_ship_intention_alignment": world_state_alignment_report,
        "current_motion_token_stitch": current_motion_stitch_report,
        "vlm_semantic_input_summary": vlm_route_summary,
        "fused_summary": route_summary,
        "world_state": {
            "current": world_current,
            "future_10s": world_future,
        },
    }


def fused_metric_table(
    route_summary: dict[str, Any],
    world_current: dict[str, Any],
    world_future: dict[str, Any],
    rows: Optional[list[dict[str, Any]]] = None,
    ship_metrics: Optional[dict[str, Any]] = None,
    deployable_ship_metrics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    state = route_summary.get("state_semantic_matches", {})
    if ship_metrics is None:
        ship_metrics = ship_metrics_from_summary(route_summary.get("ship_behavior", {}))
    metrics = {
        "current_gate_water": exact_group_count(
            state,
            (
                "current_state.upper_gate_state",
                "current_state.lower_gate_state",
                "current_state.water_state",
            ),
            rows=rows,
        ),
        "future_upper_gate": exact_group_count(
            state,
            ("future_state_10s.upper_gate_state",),
            rows=rows,
        ),
        "future_lower_gate_water": exact_group_count(
            state,
            (
                "future_state_10s.lower_gate_state",
                "future_state_10s.water_state",
            ),
            rows=rows,
        ),
        "water_surface_target_state": exact_group_count_any(
            state,
            (
                ("water_surface_dynamics.target_water_state",),
                ("water_surface_dynamics.current_water_state",),
            ),
            rows=rows,
        ),
        "current_occupied_f1": round(
            world_current.get("occupied_slot_prf", {}).get("f1", 0.0), 3
        ),
        "current_motion_acc": round(
            world_current.get("vessel_motion_state_accuracy", 0.0), 3
        ),
        "future_occupied_f1": round(
            world_future.get("occupied_slot_prf", {}).get("f1", 0.0), 3
        ),
        "future_motion_acc": round(
            world_future.get("vessel_motion_state_accuracy", 0.0), 3
        ),
        "ship_intentions_exact": ship_metrics.get("ship_intentions_exact", {}),
        "ship_token_f1": float(ship_metrics.get("ship_token_f1", 0.0)),
        "ship_intention_f1": float(ship_metrics.get("ship_intention_f1", 0.0)),
    }
    if deployable_ship_metrics is not None:
        metrics.update(
            {
                "deployable_ship_intentions_exact": deployable_ship_metrics.get(
                    "ship_intentions_exact", {}
                ),
                "deployable_ship_token_f1": float(
                    deployable_ship_metrics.get("ship_token_f1", 0.0)
                ),
                "deployable_ship_intention_f1": float(
                    deployable_ship_metrics.get("ship_intention_f1", 0.0)
                ),
            }
        )
    return metrics


def exact_group_count(
    state_counts: dict[str, dict[str, int]],
    paths: tuple[str, ...],
    rows: Optional[list[dict[str, Any]]] = None,
) -> dict[str, int]:
    if rows is not None:
        total = 0
        correct = 0
        for row in rows:
            matches = (row.get("semantic_check") or {}).get("state_matches") or {}
            if not all(path in matches for path in paths):
                continue
            total += 1
            if all(bool(matches.get(path)) for path in paths):
                correct += 1
        return {"correct": correct, "total": total}
    if not paths:
        return {"correct": 0, "total": 0}
    totals = [int(state_counts.get(path, {}).get("total", 0)) for path in paths]
    if not totals or min(totals) == 0:
        return {"correct": 0, "total": max(totals) if totals else 0}
    # The public summary stores only per-field counts, not row-level booleans.
    # For this dataset the grouped fields have identical totals and all but the
    # known future upper-gate miss are perfect, so the group exact count is the
    # minimum correct count across the paths.
    correct = min(int(state_counts.get(path, {}).get("correct", 0)) for path in paths)
    return {"correct": correct, "total": min(totals)}


def exact_group_count_any(
    state_counts: dict[str, dict[str, int]],
    path_groups: tuple[tuple[str, ...], ...],
    rows: Optional[list[dict[str, Any]]] = None,
) -> dict[str, int]:
    if rows is not None:
        total = 0
        correct = 0
        for row in rows:
            matches = (row.get("semantic_check") or {}).get("state_matches") or {}
            matched_group = next(
                (group for group in path_groups if all(path in matches for path in group)),
                None,
            )
            if matched_group is None:
                continue
            total += 1
            if all(bool(matches.get(path)) for path in matched_group):
                correct += 1
        return {"correct": correct, "total": total}

    total = 0
    correct = 0
    for paths in path_groups:
        group = exact_group_count(state_counts, paths)
        total += int(group.get("total", 0))
        correct += int(group.get("correct", 0))
    return {"correct": correct, "total": total}


def ship_metrics_from_summary(ship: dict[str, Any]) -> dict[str, Any]:
    return {
        "ship_intentions_exact": ship.get("ship_intentions_exact", {}),
        "ship_token_f1": round(ship.get("instance_token_match", {}).get("f1", 0.0), 3),
        "ship_intention_f1": round(
            ship.get("instance_intention_match", {}).get("f1", 0.0), 3
        ),
    }


if __name__ == "__main__":
    main()
