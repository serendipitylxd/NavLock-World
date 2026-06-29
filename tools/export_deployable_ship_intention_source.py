#!/usr/bin/env python3
"""Export deployable ship-intention context for VLM semantic prompt rebuilding.

The VLM semantic branch prompt keeps ``ship_behavior_context`` as input. Main-baseline
prompts must source that context from the deployable perception/geometry branch,
not annotation-backed scene instances. This tool writes a minimal JSONL with
``prediction_json.ship_behavior.ship_intentions`` for every scene that has an
input window, including current-only scenes without a future prediction target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from navlock_world.lock_world_state import load_lock_chamber_bounds, load_scene_berths
from tools.analyze_rtmdet_hydro_2d_support import load_rtmdet_ship_boxes
from tools.apply_berth_ship_intention_guard import (
    apply_lockage_phase_consistency_guard,
    deployable_hydro_ship_intentions,
    deployable_tracking_context,
    load_sequences,
)
from tools.analyze_rtmdet_ship_intention_support import (
    build_track_features,
    index_features_by_output_token,
)
from tools.derive_world_state_from_hydro3dnet_tracks import load_hydro_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--granularity",
        choices=("scene", "frame"),
        default="scene",
        help=(
            "scene exports one source row per scene. frame exports one source "
            "row per recognition frame and keys it by sample_token."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-json", type=Path, default=None)
    parser.add_argument("--sequences", type=Path, default=None)
    parser.add_argument("--hydro-predictions", type=Path, default=None)
    parser.add_argument("--rtmdet-predictions", type=Path, default=None)
    parser.add_argument(
        "--lock-boundary-map",
        type=Path,
        default=Path("data/maps/huaiyin_lock_boundary.json"),
    )
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--track-distance-m", type=float, default=40.0)
    parser.add_argument("--eval-token-map", action="store_true")
    parser.add_argument("--eval-token-map-distance-m", type=float, default=40.0)
    parser.add_argument("--recover-rtmdet-multicamera", action="store_true")
    parser.add_argument("--rtmdet-score-threshold", type=float, default=0.30)
    parser.add_argument("--support-iou-threshold", type=float, default=0.30)
    parser.add_argument("--current-active-rtmdet-min-cameras", type=int, default=1)
    parser.add_argument("--current-active-max-missing-frames", type=int, default=2)
    parser.add_argument("--recovery-min-cameras", type=int, default=4)
    parser.add_argument("--recovery-max-ray-residual-m", type=float, default=10.0)
    parser.add_argument("--recovery-cluster-distance-m", type=float, default=20.0)
    parser.add_argument("--recovery-existing-distance-m", type=float, default=20.0)
    parser.add_argument("--recovery-chamber-margin-m", type=float, default=0.0)
    parser.add_argument("--recover-open-gate-new-ships", action="store_true")
    parser.add_argument("--open-gate-min-cameras", type=int, default=3)
    parser.add_argument("--open-gate-zone-length-m", type=float, default=70.0)
    parser.add_argument("--open-gate-max-candidates", type=int, default=1)
    parser.add_argument("--apply-rtmdet-ship-intention-static-berth", action="store_true")
    parser.add_argument("--static-2d-motion-threshold", type=float, default=0.02)
    parser.add_argument("--recovery-all-input-frames", action="store_true")
    parser.add_argument("--eval-open-gate-new-ship-tokens", action="store_true")
    parser.add_argument(
        "--source-name",
        default="hydro3dnet_rtmdet_geometry",
        help="Metadata label written to each exported row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    scene_json = args.scene_json or data_root / "v1.0-trainval" / "scene.json"
    sequence_file = args.sequences or (
        data_root / "navlock_sequences" / f"scene_sequences_{args.split}.json"
    )
    hydro_predictions_file = args.hydro_predictions or (
        Path("outputs") / "hydro3dnet_navlock" / f"{args.split}_predictions.json"
    )
    rtmdet_predictions_file = args.rtmdet_predictions or (
        Path("outputs") / "mmdet2d" / "navlock_rtmdet_s" / f"{args.split}_predictions.pkl"
    )

    sequences = load_sequences(sequence_file)
    scene_berths = load_scene_berths(scene_json)
    hydro_predictions = load_hydro_predictions(hydro_predictions_file)
    rtmdet_by_path = None
    if args.recover_rtmdet_multicamera or args.recover_open_gate_new_ships:
        rtmdet_by_path = load_rtmdet_ship_boxes(
            rtmdet_predictions_file,
            args.rtmdet_score_threshold,
        )
    prior_args = SimpleNamespace(
        target="recognition" if args.granularity == "frame" else "prediction",
        score_threshold=args.score_threshold,
        track_distance_m=args.track_distance_m,
        eval_token_map=bool(args.eval_token_map),
        eval_token_map_distance_m=args.eval_token_map_distance_m,
        data_root=data_root,
        lock_chamber_bounds=load_lock_chamber_bounds(args.lock_boundary_map),
        recover_rtmdet_multicamera=bool(args.recover_rtmdet_multicamera),
        rtmdet_by_path=rtmdet_by_path,
        support_iou_threshold=args.support_iou_threshold,
        current_active_rtmdet_min_cameras=args.current_active_rtmdet_min_cameras,
        current_active_max_missing_frames=args.current_active_max_missing_frames,
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

    rows: list[dict[str, Any]] = []
    static_berth_changed_items = 0
    for sequence in sequences.values():
        if args.granularity == "frame":
            frame_rows, changed = frame_level_rows_for_sequence(
                sequence,
                scene_berths,
                hydro_predictions,
                rtmdet_by_path,
                prior_args,
                args,
            )
            rows.extend(frame_rows)
            static_berth_changed_items += changed
            continue
        if not sequence.get("prediction_input_frame_indices"):
            continue
        scene_token = sequence.get("scene_token")
        berths = scene_berths.get(scene_token, [])
        items = deployable_hydro_ship_intentions(
            sequence,
            berths,
            hydro_predictions,
            prior_args,
        )
        if args.apply_rtmdet_ship_intention_static_berth and rtmdet_by_path:
            static_berth_changed_items += apply_rtmdet_static_berth_override(
                items,
                sequence,
                berths,
                hydro_predictions,
                rtmdet_by_path,
                args,
            )
        frames, tracked_frames, token_map = deployable_tracking_context(
            sequence,
            berths,
            hydro_predictions,
            prior_args,
        )
        items = apply_lockage_phase_consistency_guard(
            items,
            tracked_frames,
            token_map,
            berths,
            frames,
            str(scene_token or ""),
        )
        rows.append(
            {
                "id": f"{args.split}:recognition:{scene_token}",
                "split": args.split,
                "scene_token": scene_token,
                "has_prediction_target": bool(sequence.get("has_prediction_target")),
                "prediction_json": {
                    "ship_behavior": {
                        "ship_intentions": items,
                    }
                },
                "ship_intention_context_source": args.source_name,
                "source_settings": {
                    "apply_rtmdet_ship_intention_static_berth": bool(
                        args.apply_rtmdet_ship_intention_static_berth
                    ),
                    "static_2d_motion_threshold": args.static_2d_motion_threshold,
                    "current_active_rtmdet_min_cameras": args.current_active_rtmdet_min_cameras,
                    "current_active_max_missing_frames": args.current_active_max_missing_frames,
                },
            }
        )

    write_jsonl(args.output, rows)
    report = {
        "output": str(args.output),
        "split": args.split,
        "granularity": args.granularity,
        "rows": len(rows),
        "prediction_target_rows": sum(
            1 for row in rows if row.get("has_prediction_target")
        ),
        "current_only_rows": sum(
            1 for row in rows if not row.get("has_prediction_target")
        ),
        "source_name": args.source_name,
        "static_berth_changed_items": static_berth_changed_items,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


def frame_level_rows_for_sequence(
    sequence: dict[str, Any],
    scene_berths: dict[str, list[dict[str, Any]]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    rtmdet_by_path: dict[str, list[dict[str, Any]]] | None,
    prior_args: SimpleNamespace,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    static_berth_changed_items = 0
    scene_token = sequence.get("scene_token")
    frames = sequence.get("frames") or []
    recognition_indices = sequence.get("recognition_frame_indices") or []
    berths = scene_berths.get(scene_token, [])
    for current_index in recognition_indices:
        if not isinstance(current_index, int) or current_index < 0 or current_index >= len(frames):
            continue
        frame = frames[current_index]
        context_indices = [index for index in recognition_indices if index <= current_index]
        current_sequence = dict(sequence)
        current_sequence["recognition_frame_indices"] = context_indices
        current_sequence["has_prediction_target"] = False
        items = deployable_hydro_ship_intentions(
            current_sequence,
            berths,
            hydro_predictions,
            prior_args,
        )
        if args.apply_rtmdet_ship_intention_static_berth and rtmdet_by_path:
            static_berth_changed_items += apply_rtmdet_static_berth_override(
                items,
                current_sequence,
                berths,
                hydro_predictions,
                rtmdet_by_path,
                args,
            )
        context_frames, tracked_frames, token_map = deployable_tracking_context(
            current_sequence,
            berths,
            hydro_predictions,
            prior_args,
        )
        items = apply_lockage_phase_consistency_guard(
            items,
            tracked_frames,
            token_map,
            berths,
            context_frames,
            str(scene_token or ""),
        )
        sample_token = frame.get("sample_token")
        rows.append(
            {
                "id": f"{args.split}:recognition_frame:{scene_token}:{sample_token}",
                "split": args.split,
                "scene_token": scene_token,
                "sample_token": sample_token,
                "timestamp": frame.get("timestamp"),
                "timestamp_str": frame.get("timestamp_str"),
                "current_frame_index": current_index,
                "has_prediction_target": bool(sequence.get("has_prediction_target")),
                "prediction_json": {
                    "ship_behavior": {
                        "ship_intentions": items,
                    }
                },
                "ship_intention_context_source": args.source_name,
                "source_settings": {
                    "granularity": "frame",
                    "apply_rtmdet_ship_intention_static_berth": bool(
                        args.apply_rtmdet_ship_intention_static_berth
                    ),
                    "static_2d_motion_threshold": args.static_2d_motion_threshold,
                    "current_active_rtmdet_min_cameras": args.current_active_rtmdet_min_cameras,
                    "current_active_max_missing_frames": args.current_active_max_missing_frames,
                },
            }
        )
    return rows, static_berth_changed_items


def apply_rtmdet_static_berth_override(
    items: list[dict[str, Any]],
    sequence: dict[str, Any],
    berths: list[dict[str, Any]],
    hydro_predictions: dict[str, dict[Any, dict[str, Any]]],
    rtmdet_by_path: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> int:
    features = build_track_features(
        sequence,
        berths,
        hydro_predictions,
        rtmdet_by_path,
        data_root=args.data_root,
        hydro_score_threshold=args.score_threshold,
        track_distance_m=args.track_distance_m,
        eval_token_map_distance_m=args.eval_token_map_distance_m,
        support_iou_threshold=args.support_iou_threshold,
    )
    feature_index = index_features_by_output_token(features)
    changed = 0
    for item in items:
        token = item.get("instance_token")
        if token is None:
            continue
        feature = first_track_feature(feature_index.get(str(token), []))
        if feature is None or not rtmdet_static_berth_candidate(feature, args):
            continue
        if item.get("ship_intentions") != ["ship_entering_lock"]:
            continue
        item["ship_intentions"] = ["ship_berthed"]
        changed += 1
    return changed


def first_track_feature(features: list[dict[str, Any]]) -> dict[str, Any] | None:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
