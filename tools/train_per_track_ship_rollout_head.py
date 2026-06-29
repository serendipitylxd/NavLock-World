#!/usr/bin/env python3
"""Train a per-track deployable ship rollout head.

This is the ship-level follow-up to the aggregate deployable ship rollout head.
Deployable Hydro3DNet/RTMDet tracks do not share annotation instance IDs, so
training uses a current-frame spatial/category match between deployable track
items and dense future label ships. At inference time each deployable track
predicts its future coarse region, berth slot, and motion state; a deterministic
berth-assignment step then enforces one ship per berth slot before aggregating
back to the existing dense rollout metrics.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools.evaluate_deployable_dense_ship_rollout import (
    COARSE_REGIONS,
    HORIZONS,
    coarse_count_counter,
    metrics_for_mode,
    occupied_slot_set,
    read_jsonl,
    set_counts,
    target_motion_counter,
    write_jsonl,
)
from tools.train_action_planner_head import as_float, as_int, float_or_zero


PER_TRACK_MODE = "per_track_learned_rollout"
PER_TRACK_CALIBRATED_MODE = "per_track_calibrated_rollout"
PER_TRACK_HYBRID_MODE = "per_track_berth_hybrid_rollout"
NONE_BERTH = "__none__"
UNKNOWN = "unknown"
ASSIGNMENT_FILL_THRESHOLDS = tuple(round(value / 100.0, 2) for value in range(0, 101, 5))
ASSIGNMENT_KEEP_THRESHOLDS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
DEFAULT_MOTION_LABELS = (
    "ship_berthed",
    "ship_entering_lock",
    "ship_leaving_lock",
    "ship_static",
    "ship_moving",
    "unknown",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-world-state",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/"
            "deployable_world_state_train_full_deployable.jsonl"
        ),
    )
    parser.add_argument(
        "--train-dense-labels",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/dense_ship_future_labels_train.jsonl"
        ),
    )
    parser.add_argument(
        "--eval-world-state",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/"
            "deployable_world_state_valtest_full_deployable.jsonl"
        ),
    )
    parser.add_argument(
        "--eval-dense-labels",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/dense_ship_future_labels_valtest.jsonl"
        ),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/per_track_ship_rollout_head.pkl"
        ),
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/"
            "predictions_valtest_per_track_ship_rollout_head.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/"
            "summary_valtest_per_track_ship_rollout_head.json"
        ),
    )
    parser.add_argument("--no-history-features", action="store_true")
    parser.add_argument(
        "--rich-history-features",
        action="store_true",
        help="Include track age, dwell, gap, speed, and delta-history features. "
        "Kept as an ablation because it can overfit Hydro track noise.",
    )
    parser.add_argument("--match-min-score", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_world = read_jsonl(args.train_world_state)
    train_labels = read_jsonl(args.train_dense_labels)
    eval_world = read_jsonl(args.eval_world_state)
    eval_labels = read_jsonl(args.eval_dense_labels)

    model = train_per_track_model(
        train_world,
        train_labels,
        use_history_features=not args.no_history_features,
        use_temporal_history_features=args.rich_history_features,
        match_min_score=args.match_min_score,
        max_iter=args.max_iter,
    )
    world_by_sample = {str(row.get("sample_token")): row for row in eval_world}
    predictions = build_predictions(eval_labels, world_by_sample, model)
    summary = build_summary(
        train_world,
        train_labels,
        eval_world,
        eval_labels,
        predictions,
        model=model,
        train_world_state=args.train_world_state,
        train_dense_labels=args.train_dense_labels,
        eval_world_state=args.eval_world_state,
        eval_dense_labels=args.eval_dense_labels,
        output_model=args.output_model,
        prediction_output=args.prediction_output,
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with args.output_model.open("wb") as handle:
        pickle.dump(model, handle)
    write_jsonl(args.prediction_output, predictions)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote_model={args.output_model}")
    print(f"wrote_predictions={args.prediction_output}")
    print(f"wrote_summary={args.summary_output}")
    print(
        "matched_eval_frames="
        f"{summary['matched_eval_frames']}/{summary['num_eval_label_frames']}"
    )
    print(
        "train_track_matches="
        f"{summary['track_training_rows']}/{summary['track_match_candidates']}"
    )
    for horizon, metrics in summary["rollout_metrics"][PER_TRACK_MODE].items():
        print(
            f"{horizon}: berth_f1={metrics['berth_occupied_f1']:.3f} "
            f"coarse_f1={metrics['coarse_region_count_f1']:.3f} "
            f"motion_f1={metrics['motion_count_f1']:.3f}"
        )


def train_per_track_model(
    world_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    *,
    use_history_features: bool = True,
    use_temporal_history_features: bool = False,
    match_min_score: float = 2.0,
    max_iter: int = 3000,
) -> dict[str, Any]:
    world_by_sample = {str(row.get("sample_token")): row for row in world_rows}
    history_by_sample = (
        build_track_history_features(world_rows) if use_history_features else {}
    )
    berth_slot_ids = collect_berth_slot_ids(world_rows, label_rows)
    coarse_region_ids = collect_coarse_region_ids(world_rows, label_rows)
    motion_labels = collect_motion_labels(world_rows, label_rows)

    examples_by_horizon: dict[str, list[dict[str, Any]]] = {}
    match_candidates_by_horizon: dict[str, int] = {}
    for horizon in HORIZONS:
        horizon_key = horizon_name(horizon)
        examples, candidate_count = build_training_examples(
            label_rows,
            world_by_sample,
            horizon_key,
            history_by_sample=history_by_sample,
            use_temporal_history_features=use_temporal_history_features,
            match_min_score=match_min_score,
        )
        examples_by_horizon[horizon_key] = examples
        match_candidates_by_horizon[horizon_key] = candidate_count

    all_features = [
        example["features"]
        for examples in examples_by_horizon.values()
        for example in examples
    ]
    if not all_features:
        raise SystemExit("no matched per-track training examples")
    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit(all_features)

    heads: dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_key = horizon_name(horizon)
        examples = examples_by_horizon[horizon_key]
        if not examples:
            heads[horizon_key] = empty_horizon_heads()
            continue
        x = vectorizer.transform([example["features"] for example in examples])
        heads[horizon_key] = {
            "num_train_rows": len(examples),
            "future_region": fit_classifier(
                x,
                [example["future_region"] for example in examples],
                max_iter=max_iter,
            ),
            "future_berthed": fit_binary_classifier(
                x,
                [example["future_berth_slot"] != NONE_BERTH for example in examples],
                max_iter=max_iter,
            ),
            "future_berth_slot": fit_classifier(
                x,
                [example["future_berth_slot"] for example in examples],
                max_iter=max_iter,
            ),
            "future_motion_state": fit_classifier(
                x,
                [example["future_motion_state"] for example in examples],
                max_iter=max_iter,
            ),
            "match_score_counts": {
                str(key): int(value)
                for key, value in Counter(
                    int(example["match_score"]) for example in examples
                ).items()
            },
        }

    model = {
        "vectorizer": vectorizer,
        "heads": heads,
        "horizons": list(HORIZONS),
        "berth_slot_ids": berth_slot_ids,
        "coarse_region_ids": coarse_region_ids,
        "motion_labels": motion_labels,
        "use_history_features": use_history_features,
        "use_temporal_history_features": use_temporal_history_features,
        "match_min_score": match_min_score,
        "num_train_world_state_frames": len(world_rows),
        "num_train_label_frames": len(label_rows),
        "track_training_rows": sum(len(v) for v in examples_by_horizon.values()),
        "track_match_candidates": sum(match_candidates_by_horizon.values()),
        "horizon_train_rows": {
            key: len(value) for key, value in examples_by_horizon.items()
        },
        "horizon_match_candidates": match_candidates_by_horizon,
        "num_features": len(vectorizer.feature_names_),
        "feature_policy": {
            "input": "per-track deployable Hydro3DNet/RTMDet world state",
            "target": "spatially matched dense future ship labels",
            "identity": "training uses spatial/category matching; evaluation remains aggregate spatial/motion metrics",
            "history_features": "basic previous-frame history by default; rich temporal identity/dwell features are an explicit ablation",
            "berth_assignment": "one track per berth slot, with train-only horizon calibration for learned berth fills",
        },
    }
    model["assignment_configs"] = tune_assignment_configs(
        label_rows,
        world_by_sample,
        model,
        history_by_sample=history_by_sample,
    )
    return model


def build_training_examples(
    label_rows: list[dict[str, Any]],
    world_by_sample: dict[str, dict[str, Any]],
    horizon_key: str,
    *,
    history_by_sample: dict[str, dict[str, dict[str, Any]]],
    use_temporal_history_features: bool,
    match_min_score: float,
) -> tuple[list[dict[str, Any]], int]:
    examples: list[dict[str, Any]] = []
    candidate_count = 0
    for label in label_rows:
        target = horizon_target(label, horizon_key)
        if not isinstance(target, dict):
            continue
        ships = [ship for ship in target.get("matched_ships") or [] if isinstance(ship, dict)]
        if not ships:
            continue
        world = world_by_sample.get(str(label.get("sample_token")))
        if world is None:
            continue
        track_states = extract_track_states(world)
        if not track_states:
            continue
        matches = match_tracks_to_label_ships(
            track_states,
            ships,
            min_score=match_min_score,
        )
        candidate_count += min(len(track_states), len(ships))
        history_for_sample = history_by_sample.get(str(label.get("sample_token"))) or {}
        for track, ship, score in matches:
            examples.append(
                {
                    "features": track_features(
                        track,
                        world,
                        history_features=history_for_sample.get(track.token),
                        use_temporal_history_features=use_temporal_history_features,
                    ),
                    "future_region": normalize_label(ship.get("future_region")),
                    "future_berth_slot": normalize_berth_label(
                        ship.get("future_berth_slot")
                    ),
                    "future_motion_state": normalize_label(
                        ship.get("target_motion_state")
                    ),
                    "match_score": score,
                }
            )
    return examples, candidate_count


def build_predictions(
    label_rows: list[dict[str, Any]],
    world_by_sample: dict[str, dict[str, Any]],
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    history_by_sample = (
        build_track_history_features(list(world_by_sample.values()))
        if model.get("use_history_features")
        else {}
    )
    predictions = []
    for label in label_rows:
        sample_token = str(label.get("sample_token"))
        world = world_by_sample.get(sample_token)
        if world is None:
            continue
        track_states = extract_track_states(world)
        history_for_sample = history_by_sample.get(sample_token) or {}
        modes: dict[str, dict[str, Any]] = {
            "deployable_persistence": {
                horizon_name(h): deployable_persistence(world) for h in HORIZONS
            },
            "dispatch_aware_rollout": {
                horizon_name(h): dispatch_aware_rollout(world) for h in HORIZONS
            },
        }
        per_track_horizons = {}
        calibrated_horizons = {}
        hybrid_horizons = {}
        for horizon in model.get("horizons") or HORIZONS:
            horizon_key = horizon_name(int(horizon))
            per_track_pred = predict_horizon(
                model,
                world,
                track_states,
                horizon_key,
                history_for_sample=history_for_sample,
            )
            calibrated_pred = predict_horizon(
                model,
                world,
                track_states,
                horizon_key,
                history_for_sample=history_for_sample,
                assignment_config=assignment_config_for_horizon(model, horizon_key),
            )
            per_track_horizons[horizon_key] = per_track_pred
            calibrated_horizons[horizon_key] = calibrated_pred
            hybrid_horizons[horizon_key] = per_track_berth_hybrid(
                calibrated_pred,
                modes["deployable_persistence"][horizon_key],
            )
        modes[PER_TRACK_MODE] = per_track_horizons
        modes[PER_TRACK_CALIBRATED_MODE] = calibrated_horizons
        modes[PER_TRACK_HYBRID_MODE] = hybrid_horizons
        predictions.append(
            {
                "sample_token": sample_token,
                "split": label.get("split"),
                "scene_token": label.get("scene_token"),
                "timestamp": label.get("timestamp"),
                "timestamp_str": label.get("timestamp_str"),
                "rollout_modes": modes,
            }
        )
    return predictions


def build_summary(
    train_world_rows: list[dict[str, Any]],
    train_label_rows: list[dict[str, Any]],
    eval_world_rows: list[dict[str, Any]],
    eval_label_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    train_world_state: Path,
    train_dense_labels: Path,
    eval_world_state: Path,
    eval_dense_labels: Path,
    output_model: Path,
    prediction_output: Path,
) -> dict[str, Any]:
    pred_by_sample = {str(row.get("sample_token")): row for row in predictions}
    modes = [
        "deployable_persistence",
        "dispatch_aware_rollout",
        PER_TRACK_MODE,
        PER_TRACK_CALIBRATED_MODE,
        PER_TRACK_HYBRID_MODE,
    ]
    eval_world_by_sample = {str(row.get("sample_token")): row for row in eval_world_rows}
    return {
        "train_world_state": str(train_world_state),
        "train_dense_labels": str(train_dense_labels),
        "eval_world_state": str(eval_world_state),
        "eval_dense_labels": str(eval_dense_labels),
        "output_model": str(output_model),
        "prediction_output": str(prediction_output),
        "num_train_world_state_frames": len(train_world_rows),
        "num_train_label_frames": len(train_label_rows),
        "num_eval_label_frames": len(eval_label_rows),
        "matched_eval_frames": len(predictions),
        "num_features": int(model.get("num_features") or 0),
        "use_history_features": bool(model.get("use_history_features")),
        "use_temporal_history_features": bool(
            model.get("use_temporal_history_features")
        ),
        "match_min_score": float(model.get("match_min_score") or 0.0),
        "track_training_rows": int(model.get("track_training_rows") or 0),
        "track_match_candidates": int(model.get("track_match_candidates") or 0),
        "horizon_train_rows": model.get("horizon_train_rows") or {},
        "horizon_match_candidates": model.get("horizon_match_candidates") or {},
        "assignment_calibration": model.get("assignment_configs") or {},
        "rollout_metrics": {
            mode: metrics_for_mode(eval_label_rows, pred_by_sample, mode)
            for mode in modes
        },
        "rollout_error_breakdown": {
            mode: rollout_error_breakdown(
                eval_label_rows,
                pred_by_sample,
                mode,
                world_by_sample=eval_world_by_sample,
            )
            for mode in (PER_TRACK_MODE, PER_TRACK_CALIBRATED_MODE)
        },
        "notes": [
            "Per-track rollout trains on spatial/category matched deployable tracks because Hydro3DNet/RTMDet track IDs and annotation instance tokens are not shared.",
            "The per-track head predicts future coarse region, berth slot, and motion state for each current deployable track.",
            "Berth assignment enforces at most one predicted track per berth slot; if multiple tracks want the same berth, the highest calibrated berth probability wins.",
            "per_track_calibrated_rollout applies train-only horizon-specific fill/keep thresholds before berth assignment.",
            "Evaluation is still aggregate spatial/motion-count because benchmark identity is not deployable.",
            "per_track_berth_hybrid_rollout keeps deployable-persistence berth occupancy while using calibrated per-track learned coarse and motion predictions.",
        ],
    }


class TrackState:
    def __init__(
        self,
        *,
        token: str,
        category: str = UNKNOWN,
        current_region: str = UNKNOWN,
        current_berth_slot: Optional[str] = None,
        motion_state: str = UNKNOWN,
        direction_label: str = UNKNOWN,
        start_region: str = UNKNOWN,
        end_region: str = UNKNOWN,
        end_speed_mps: float = 0.0,
        delta_xy: Optional[list[Any]] = None,
    ) -> None:
        self.token = token
        self.category = category
        self.current_region = current_region
        self.current_berth_slot = current_berth_slot
        self.motion_state = motion_state
        self.direction_label = direction_label
        self.start_region = start_region
        self.end_region = end_region
        self.end_speed_mps = end_speed_mps
        self.delta_xy = delta_xy or [0.0, 0.0]


def extract_track_states(world: dict[str, Any]) -> list[TrackState]:
    by_token: dict[str, TrackState] = {}
    occupancy = (world.get("lock_occupancy") or {}).get("current") or {}
    for region in occupancy.get("coarse_regions") or []:
        region_id = normalize_label(region.get("region_id"))
        for token in region.get("ship_tokens") or []:
            token = str(token)
            by_token.setdefault(token, TrackState(token=token)).current_region = region_id
    for slot in occupancy.get("berth_slots") or []:
        slot_id = normalize_berth_label(slot.get("region_id"))
        if slot_id == NONE_BERTH:
            continue
        for token in slot.get("ship_tokens") or []:
            token = str(token)
            state = by_token.setdefault(token, TrackState(token=token))
            state.current_berth_slot = slot_id
            if state.current_region == UNKNOWN:
                state.current_region = "between_berths"
    for item in (world.get("vessel_motion_flow") or {}).get("input_window") or []:
        token = str(item.get("instance_token") or "")
        if not token:
            continue
        state = by_token.setdefault(token, TrackState(token=token))
        state.category = normalize_label(item.get("category"))
        state.motion_state = normalize_label(item.get("motion_state"))
        state.direction_label = normalize_label(item.get("direction_label"))
        state.start_region = normalize_label(item.get("start_region"))
        state.end_region = normalize_label(item.get("end_region"))
        if state.current_region == UNKNOWN and state.end_region != UNKNOWN:
            state.current_region = state.end_region
        state.end_speed_mps = float_or_zero(item.get("end_speed_mps"))
        delta = item.get("delta_xy")
        if isinstance(delta, list) and len(delta) >= 2:
            state.delta_xy = delta
    return sorted(by_token.values(), key=lambda state: state.token)


def match_tracks_to_label_ships(
    tracks: list[TrackState],
    ships: list[dict[str, Any]],
    *,
    min_score: float,
) -> list[tuple[TrackState, dict[str, Any], float]]:
    candidates = []
    for track in tracks:
        for ship in ships:
            score = track_ship_match_score(track, ship)
            candidates.append((score, track.token, str(ship.get("instance_token")), track, ship))
    candidates.sort(reverse=True)
    used_tracks: set[str] = set()
    used_ships: set[str] = set()
    matches = []
    for score, track_token, ship_token, track, ship in candidates:
        if score < min_score:
            break
        if track_token in used_tracks or ship_token in used_ships:
            continue
        used_tracks.add(track_token)
        used_ships.add(ship_token)
        matches.append((track, ship, score))
    return matches


def track_ship_match_score(track: TrackState, ship: dict[str, Any]) -> float:
    score = 0.0
    if normalize_category(track.category) == normalize_category(ship.get("category")):
        score += 3.0
    if track.current_berth_slot and track.current_berth_slot == normalize_berth_label(
        ship.get("current_berth_slot")
    ):
        score += 3.0
    if track.current_region == normalize_label(ship.get("current_region")):
        score += 2.0
    if track.motion_state == normalize_label(ship.get("current_motion_state")):
        score += 1.0
    if track.current_region == UNKNOWN and normalize_label(ship.get("current_region")) == UNKNOWN:
        score -= 1.0
    return score


def track_features(
    track: TrackState,
    world: dict[str, Any],
    *,
    history_features: Optional[dict[str, Any]] = None,
    use_temporal_history_features: bool = False,
) -> dict[str, Any]:
    occupancy = (world.get("lock_occupancy") or {}).get("current") or {}
    stitch = world.get("planner_feature_stitch") or {}
    track_source = world.get("track_source") or {}
    recovery = track_source.get("rtmdet_recovery_summary") or {}
    dx, dy = track_delta_xy(track)
    feat: dict[str, Any] = {
        "scene_direction": direction_from_scene(world.get("scene_token")),
        "track_category": normalize_category(track.category),
        "track_category_raw": track.category,
        "track_current_region": track.current_region,
        "track_current_berth_slot": track.current_berth_slot or NONE_BERTH,
        "track_motion_state": track.motion_state,
        "track_direction_label": track.direction_label,
        "track_start_region": track.start_region,
        "track_end_region": track.end_region,
        "track_speed_mps": track.end_speed_mps,
        "track_delta_x": dx,
        "track_delta_y": dy,
        "track_abs_delta_x": abs(dx),
        "track_abs_delta_y": abs(dy),
        "track_in_berth": track.current_berth_slot is not None,
        "ship_operation_phase": stitch.get("ship_operation_phase") or "missing",
        "num_ships": float_or_zero(occupancy.get("num_ships")),
        "num_occupied_berths": float_or_zero(occupancy.get("num_occupied_berths")),
        "window_size": float_or_zero(track_source.get("window_size")),
        "rtmdet_recovered_detections": float_or_zero(recovery.get("recovered_detections")),
        "rtmdet_recovered_frames": float_or_zero(recovery.get("recovered_frames")),
    }
    for slot in occupancy.get("berth_slots") or []:
        slot_id = normalize_berth_label(slot.get("region_id"))
        if slot_id != NONE_BERTH:
            feat[f"berth_{slot_id}_count"] = int(slot.get("ship_count") or 0)
            feat[f"berth_{slot_id}_occupied"] = bool(slot.get("occupied"))
    for region in occupancy.get("coarse_regions") or []:
        region_id = normalize_label(region.get("region_id"))
        feat[f"coarse_{region_id}_count"] = int(region.get("ship_count") or 0)
    for label, count in motion_counts_from_world(world).items():
        feat[f"motion_{label}_count"] = count
    add_track_history_features(
        feat,
        history_features,
        use_temporal_history_features=use_temporal_history_features,
    )
    return feat


def build_track_history_features(
    world_rows: Iterable[dict[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    by_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in world_rows:
        by_scene[(str(row.get("split")), str(row.get("scene_token")))].append(row)
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for rows in by_scene.values():
        rows.sort(key=lambda row: (int(row.get("timestamp") or 0), str(row.get("sample_token"))))
        previous_tracks: dict[str, TrackState] = {}
        previous_timestamp: Optional[int] = None
        track_memory: dict[str, dict[str, Any]] = {}
        for row in rows:
            timestamp = as_int(row.get("timestamp"))
            per_track = {}
            current_tracks = {track.token: track for track in extract_track_states(row)}
            for token, track in current_tracks.items():
                prev = previous_tracks.get(token)
                memory = track_memory.setdefault(
                    token,
                    {
                        "first_timestamp": timestamp,
                        "seen_frame_count": 0,
                        "last_timestamp": None,
                        "last_track": None,
                        "current_region": None,
                        "region_start_timestamp": timestamp,
                        "region_frame_count": 0,
                        "current_berth_slot": None,
                        "berth_start_timestamp": timestamp,
                        "berth_frame_count": 0,
                        "current_motion_state": None,
                        "motion_start_timestamp": timestamp,
                        "motion_frame_count": 0,
                        "current_direction_label": None,
                        "direction_start_timestamp": timestamp,
                        "direction_frame_count": 0,
                        "ever_berthed": False,
                        "distinct_berth_slots": set(),
                    },
                )
                last_seen_timestamp = memory.get("last_timestamp")
                last_seen_track = memory.get("last_track")
                gap_sec = None
                if timestamp is not None and last_seen_timestamp is not None:
                    gap_sec = max(0.0, (timestamp - int(last_seen_timestamp)) / 1_000_000.0)

                update_track_run(memory, "region", track.current_region, timestamp)
                update_track_run(
                    memory,
                    "berth",
                    track.current_berth_slot or NONE_BERTH,
                    timestamp,
                )
                update_track_run(memory, "motion", track.motion_state, timestamp)
                update_track_run(memory, "direction", track.direction_label, timestamp)
                memory["seen_frame_count"] = int(memory.get("seen_frame_count") or 0) + 1
                memory["ever_berthed"] = bool(memory.get("ever_berthed")) or (
                    track.current_berth_slot is not None
                )
                if track.current_berth_slot:
                    memory["distinct_berth_slots"].add(track.current_berth_slot)

                age_sec = None
                first_timestamp = memory.get("first_timestamp")
                if timestamp is not None and first_timestamp is not None:
                    age_sec = max(0.0, (timestamp - int(first_timestamp)) / 1_000_000.0)
                dx, dy = track_delta_xy(track)
                prev_dx, prev_dy = (
                    track_delta_xy(last_seen_track)
                    if isinstance(last_seen_track, TrackState)
                    else (0.0, 0.0)
                )
                features = {
                    "has_prev_track": prev is not None,
                    "has_prior_track": last_seen_track is not None,
                    "track_reappeared_after_gap": bool(last_seen_track and prev is None),
                    "prev_region": prev.current_region if prev else None,
                    "prev_berth_slot": prev.current_berth_slot if prev else None,
                    "prev_motion_state": prev.motion_state if prev else None,
                    "region_changed_from_prev": bool(prev and prev.current_region != track.current_region),
                    "berth_changed_from_prev": bool(prev and prev.current_berth_slot != track.current_berth_slot),
                    "motion_changed_from_prev": bool(prev and prev.motion_state != track.motion_state),
                    "dt_prev_sec": None,
                    "track_seen_frame_count": memory["seen_frame_count"],
                    "track_age_sec": age_sec,
                    "track_gap_sec": gap_sec,
                    "region_dwell_frame_count": memory.get("region_frame_count"),
                    "region_dwell_sec": run_duration_sec(memory, "region", timestamp),
                    "berth_dwell_frame_count": memory.get("berth_frame_count"),
                    "berth_dwell_sec": run_duration_sec(memory, "berth", timestamp),
                    "motion_dwell_frame_count": memory.get("motion_frame_count"),
                    "motion_dwell_sec": run_duration_sec(memory, "motion", timestamp),
                    "direction_dwell_frame_count": memory.get("direction_frame_count"),
                    "direction_dwell_sec": run_duration_sec(memory, "direction", timestamp),
                    "track_ever_berthed": memory.get("ever_berthed"),
                    "track_distinct_berth_slot_count": len(memory.get("distinct_berth_slots") or []),
                    "prev_seen_region": last_seen_track.current_region if last_seen_track else None,
                    "prev_seen_berth_slot": last_seen_track.current_berth_slot if last_seen_track else None,
                    "prev_seen_motion_state": last_seen_track.motion_state if last_seen_track else None,
                    "prev_speed_mps": last_seen_track.end_speed_mps if last_seen_track else None,
                    "speed_delta_prev": (
                        track.end_speed_mps - last_seen_track.end_speed_mps
                        if last_seen_track
                        else None
                    ),
                    "abs_speed_delta_prev": (
                        abs(track.end_speed_mps - last_seen_track.end_speed_mps)
                        if last_seen_track
                        else None
                    ),
                    "prev_delta_x": prev_dx if last_seen_track else None,
                    "prev_delta_y": prev_dy if last_seen_track else None,
                    "delta_x_delta_prev": dx - prev_dx if last_seen_track else None,
                    "delta_y_delta_prev": dy - prev_dy if last_seen_track else None,
                }
                if timestamp is not None and previous_timestamp is not None:
                    features["dt_prev_sec"] = max(0.0, (timestamp - previous_timestamp) / 1_000_000.0)
                per_track[token] = features
                memory["last_timestamp"] = timestamp
                memory["last_track"] = track
            output[str(row.get("sample_token"))] = per_track
            previous_tracks = current_tracks
            previous_timestamp = timestamp
    return output


def update_track_run(
    memory: dict[str, Any],
    name: str,
    value: Any,
    timestamp: Optional[int],
) -> None:
    current_key = f"current_{name}"
    start_key = f"{name}_start_timestamp"
    count_key = f"{name}_frame_count"
    if memory.get(current_key) != value:
        memory[current_key] = value
        memory[start_key] = timestamp
        memory[count_key] = 0
    memory[count_key] = int(memory.get(count_key) or 0) + 1


def run_duration_sec(
    memory: dict[str, Any],
    name: str,
    timestamp: Optional[int],
) -> Optional[float]:
    start = memory.get(f"{name}_start_timestamp")
    if timestamp is None or start is None:
        return None
    return max(0.0, (timestamp - int(start)) / 1_000_000.0)


def add_track_history_features(
    feat: dict[str, Any],
    history: Optional[dict[str, Any]],
    *,
    use_temporal_history_features: bool = False,
) -> None:
    history = history or {}
    feat["hist_has_prev_track"] = bool(history.get("has_prev_track"))
    feat["hist_prev_region"] = history.get("prev_region") or "missing"
    feat["hist_prev_berth_slot"] = history.get("prev_berth_slot") or NONE_BERTH
    feat["hist_prev_motion_state"] = history.get("prev_motion_state") or "missing"
    feat["hist_region_changed_from_prev"] = bool(history.get("region_changed_from_prev"))
    feat["hist_berth_changed_from_prev"] = bool(history.get("berth_changed_from_prev"))
    feat["hist_motion_changed_from_prev"] = bool(history.get("motion_changed_from_prev"))
    feat["hist_dt_prev_sec"] = float_or_zero(history.get("dt_prev_sec"))
    feat["hist_dt_prev_sec_missing"] = history.get("dt_prev_sec") is None
    if not use_temporal_history_features:
        return
    feat["hist_has_prior_track"] = bool(history.get("has_prior_track"))
    feat["hist_track_reappeared_after_gap"] = bool(
        history.get("track_reappeared_after_gap")
    )
    feat["hist_prev_seen_region"] = history.get("prev_seen_region") or "missing"
    feat["hist_prev_seen_berth_slot"] = history.get("prev_seen_berth_slot") or NONE_BERTH
    feat["hist_prev_seen_motion_state"] = history.get("prev_seen_motion_state") or "missing"
    feat["hist_track_ever_berthed"] = bool(history.get("track_ever_berthed"))
    numeric_fields = (
        "track_seen_frame_count",
        "track_age_sec",
        "track_gap_sec",
        "region_dwell_frame_count",
        "region_dwell_sec",
        "berth_dwell_frame_count",
        "berth_dwell_sec",
        "motion_dwell_frame_count",
        "motion_dwell_sec",
        "direction_dwell_frame_count",
        "direction_dwell_sec",
        "track_distinct_berth_slot_count",
        "prev_speed_mps",
        "speed_delta_prev",
        "abs_speed_delta_prev",
        "prev_delta_x",
        "prev_delta_y",
        "delta_x_delta_prev",
        "delta_y_delta_prev",
    )
    for field in numeric_fields:
        feat[f"hist_{field}"] = float_or_zero(history.get(field))
        feat[f"hist_{field}_missing"] = history.get(field) is None


def predict_horizon(
    model: dict[str, Any],
    world: dict[str, Any],
    tracks: list[TrackState],
    horizon_key: str,
    *,
    history_for_sample: dict[str, dict[str, Any]],
    assignment_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    track_predictions = build_track_prediction_candidates(
        model,
        world,
        tracks,
        horizon_key,
        history_for_sample=history_for_sample,
    )
    assigned = assign_future_berths(track_predictions, config=assignment_config)
    occupancy = occupancy_from_track_predictions(
        assigned,
        berth_slot_ids=model.get("berth_slot_ids") or [],
        coarse_region_ids=model.get("coarse_region_ids") or COARSE_REGIONS,
    )
    motion_counts = motion_counts_from_track_predictions(assigned)
    return {
        "future_occupancy": occupancy,
        "motion_counts": motion_counts,
        "num_ships": len(assigned),
        "per_track_predictions": assigned,
    }


def build_track_prediction_candidates(
    model: dict[str, Any],
    world: dict[str, Any],
    tracks: list[TrackState],
    horizon_key: str,
    *,
    history_for_sample: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    heads = (model.get("heads") or {}).get(horizon_key) or empty_horizon_heads()
    track_predictions = []
    for track in tracks:
        features = track_features(
            track,
            world,
            history_features=history_for_sample.get(track.token),
            use_temporal_history_features=bool(
                model.get("use_temporal_history_features")
            ),
        )
        x = model["vectorizer"].transform([features])
        future_region, region_score = predict_label_with_score(heads["future_region"], x)
        is_berthed, berthed_score = predict_binary_with_score(heads["future_berthed"], x)
        future_berth_slot, slot_score = predict_label_with_score(heads["future_berth_slot"], x)
        if is_berthed and future_berth_slot == NONE_BERTH:
            future_berth_slot, slot_score = predict_best_non_none_label_with_score(
                heads["future_berth_slot"],
                x,
            )
        if not is_berthed:
            future_berth_slot = NONE_BERTH
        berth_score = berthed_score * slot_score if future_berth_slot != NONE_BERTH else 1.0 - berthed_score
        future_motion_state, motion_score = predict_label_with_score(heads["future_motion_state"], x)
        if future_berth_slot != NONE_BERTH:
            future_region = "between_berths"
        track_predictions.append(
            {
                "track_token": track.token,
                "category": track.category,
                "current_region": track.current_region,
                "current_berth_slot": track.current_berth_slot,
                "future_region": future_region,
                "future_berth_slot": future_berth_slot,
                "future_motion_state": future_motion_state,
                "scores": {
                    "future_region": region_score,
                    "future_berth_slot": berth_score,
                    "future_berthed": berthed_score,
                    "future_slot_label": slot_score,
                    "future_motion_state": motion_score,
                },
            }
        )
    return track_predictions


def assign_future_berths(
    track_predictions: list[dict[str, Any]],
    *,
    config: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    config = normalize_assignment_config(config)
    winners: dict[str, tuple[float, str]] = {}
    for pred in track_predictions:
        slot_id = pred.get("future_berth_slot")
        if not slot_id or slot_id == NONE_BERTH:
            continue
        score = float((pred.get("scores") or {}).get("future_berth_slot") or 0.0)
        if score < assignment_threshold_for_prediction(pred, config):
            continue
        rank_score = score
        if slot_id == pred.get("current_berth_slot"):
            rank_score += float(config.get("current_slot_bonus") or 0.0)
        token = str(pred.get("track_token"))
        if slot_id not in winners or (rank_score, token) > winners[slot_id]:
            winners[slot_id] = (rank_score, token)
    assigned = []
    for pred in track_predictions:
        item = dict(pred)
        slot_id = item.get("future_berth_slot")
        token = str(item.get("track_token"))
        if (
            slot_id
            and slot_id != NONE_BERTH
            and winners.get(slot_id, (None, None))[1] != token
        ):
            item["future_berth_slot"] = NONE_BERTH
        if item.get("future_berth_slot") != NONE_BERTH:
            item["future_region"] = "between_berths"
        assigned.append(item)
    return assigned


def normalize_assignment_config(
    config: Optional[dict[str, Any]]
) -> dict[str, Any]:
    config = dict(config or {})
    config.setdefault("min_berth_score", 0.0)
    config.setdefault("keep_current_min_score", config["min_berth_score"])
    config.setdefault("current_slot_bonus", 0.0)
    return config


def assignment_threshold_for_prediction(
    pred: dict[str, Any],
    config: dict[str, Any],
) -> float:
    slot_id = pred.get("future_berth_slot")
    current_slot = pred.get("current_berth_slot")
    if slot_id and current_slot and slot_id == current_slot:
        return float(config.get("keep_current_min_score") or 0.0)
    return float(config.get("min_berth_score") or 0.0)


def tune_assignment_configs(
    label_rows: list[dict[str, Any]],
    world_by_sample: dict[str, dict[str, Any]],
    model: dict[str, Any],
    *,
    history_by_sample: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    configs = {}
    for horizon in HORIZONS:
        horizon_key = horizon_name(horizon)
        candidate_rows = build_assignment_candidate_rows(
            label_rows,
            world_by_sample,
            model,
            horizon_key,
            history_by_sample=history_by_sample,
        )
        configs[horizon_key] = tune_assignment_config_for_horizon(
            candidate_rows,
            model=model,
            horizon_key=horizon_key,
        )
    return configs


def build_assignment_candidate_rows(
    label_rows: list[dict[str, Any]],
    world_by_sample: dict[str, dict[str, Any]],
    model: dict[str, Any],
    horizon_key: str,
    *,
    history_by_sample: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for label in label_rows:
        target = horizon_target(label, horizon_key)
        if not isinstance(target, dict):
            continue
        sample_token = str(label.get("sample_token"))
        world = world_by_sample.get(sample_token)
        if world is None:
            continue
        tracks = extract_track_states(world)
        history_for_sample = history_by_sample.get(sample_token) or {}
        candidates = build_track_prediction_candidates(
            model,
            world,
            tracks,
            horizon_key,
            history_for_sample=history_for_sample,
        )
        rows.append({"label": label, "target": target, "candidates": candidates})
    return rows


def tune_assignment_config_for_horizon(
    candidate_rows: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    horizon_key: str,
) -> dict[str, Any]:
    if not candidate_rows:
        return {
            "min_berth_score": 0.0,
            "keep_current_min_score": 0.0,
            "current_slot_bonus": 0.0,
            "train_num_targets": 0,
            "train_berth_f1": 0.0,
        }
    best_config: dict[str, Any] = {}
    best_rank = (-1.0, -1.0, -1.0, 0.0)
    for fill_threshold in ASSIGNMENT_FILL_THRESHOLDS:
        for keep_threshold in ASSIGNMENT_KEEP_THRESHOLDS:
            if keep_threshold > fill_threshold:
                continue
            config = {
                "min_berth_score": fill_threshold,
                "keep_current_min_score": keep_threshold,
                "current_slot_bonus": 0.02,
            }
            report = evaluate_assignment_config(
                candidate_rows,
                config,
                model=model,
            )
            rank = (
                report["f1"],
                report["precision"],
                report["recall"],
                -fill_threshold,
            )
            if rank > best_rank:
                best_rank = rank
                best_config = {
                    **config,
                    "train_num_targets": len(candidate_rows),
                    "train_berth_tp": report["tp"],
                    "train_berth_fp": report["fp"],
                    "train_berth_fn": report["fn"],
                    "train_berth_precision": report["precision"],
                    "train_berth_recall": report["recall"],
                    "train_berth_f1": report["f1"],
                    "horizon": horizon_key,
                }
    return best_config


def evaluate_assignment_config(
    candidate_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model: dict[str, Any],
) -> dict[str, Any]:
    tp = fp = fn = 0
    for row in candidate_rows:
        assigned = assign_future_berths(row["candidates"], config=config)
        pred_occ = occupancy_from_track_predictions(
            assigned,
            berth_slot_ids=model.get("berth_slot_ids") or [],
            coarse_region_ids=model.get("coarse_region_ids") or COARSE_REGIONS,
        )
        btp, bfp, bfn = set_counts(
            occupied_slot_set(pred_occ),
            occupied_slot_set((row["target"] or {}).get("future_occupancy") or {}),
        )
        tp += btp
        fp += bfp
        fn += bfn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def assignment_config_for_horizon(
    model: dict[str, Any],
    horizon_key: str,
) -> dict[str, Any]:
    return normalize_assignment_config(
        (model.get("assignment_configs") or {}).get(horizon_key)
    )


def occupancy_from_track_predictions(
    predictions: list[dict[str, Any]],
    *,
    berth_slot_ids: Iterable[str],
    coarse_region_ids: Iterable[str],
) -> dict[str, Any]:
    berth_counts = Counter()
    berth_tokens: dict[str, list[str]] = defaultdict(list)
    coarse_counts = Counter()
    coarse_tokens: dict[str, list[str]] = defaultdict(list)
    for pred in predictions:
        token = str(pred.get("track_token"))
        slot_id = pred.get("future_berth_slot")
        region_id = normalize_label(pred.get("future_region"))
        if slot_id and slot_id != NONE_BERTH:
            berth_counts[str(slot_id)] += 1
            berth_tokens[str(slot_id)].append(token)
            region_id = "between_berths"
        coarse_counts[region_id] += 1
        coarse_tokens[region_id].append(token)
    all_slots = sorted(set(str(value) for value in berth_slot_ids) | set(berth_counts))
    all_regions = sorted(set(str(value) for value in coarse_region_ids) | set(coarse_counts))
    return {
        "berth_slots": [
            {
                "region_id": slot_id,
                "occupied": berth_counts[slot_id] > 0,
                "ship_count": int(berth_counts[slot_id]),
                "ship_tokens": berth_tokens.get(slot_id, []),
            }
            for slot_id in all_slots
        ],
        "coarse_regions": [
            {
                "region_id": region_id,
                "ship_count": int(coarse_counts[region_id]),
                "ship_tokens": coarse_tokens.get(region_id, []),
            }
            for region_id in all_regions
        ],
        "num_occupied_berths": sum(1 for value in berth_counts.values() if value > 0),
        "num_ships": len(predictions),
    }


def motion_counts_from_track_predictions(
    predictions: list[dict[str, Any]],
    *,
    max_tracks: Optional[int] = None,
) -> dict[str, int]:
    items = list(predictions)
    if max_tracks is not None and max_tracks >= 0:
        items = sorted(
            items,
            key=lambda pred: float(
                (pred.get("scores") or {}).get("future_motion_state") or 0.0
            ),
            reverse=True,
        )[:max_tracks]
    return dict(Counter(pred.get("future_motion_state") or UNKNOWN for pred in items))


def rollout_error_breakdown(
    label_rows: list[dict[str, Any]],
    pred_by_sample: dict[str, dict[str, Any]],
    mode: str,
    *,
    world_by_sample: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output = {}
    for horizon in HORIZONS:
        horizon_key = horizon_name(horizon)
        phase_counts: dict[str, Counter[str]] = defaultdict(Counter)
        transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        fp_slots = Counter()
        fn_slots = Counter()
        coarse_confusion = Counter()
        motion_confusion = Counter()
        covered = 0
        for label in label_rows:
            target = horizon_target(label, horizon_key)
            pred_row = pred_by_sample.get(str(label.get("sample_token")))
            if not isinstance(target, dict) or pred_row is None:
                continue
            pred = (
                pred_row.get("rollout_modes", {})
                .get(mode, {})
                .get(horizon_key, {})
            )
            if not isinstance(pred, dict):
                continue
            covered += 1
            world = world_by_sample.get(str(label.get("sample_token"))) or {}
            phase = (world.get("planner_feature_stitch") or {}).get(
                "ship_operation_phase"
            ) or "missing"
            target_occ = target.get("future_occupancy") or {}
            pred_occ = pred.get("future_occupancy") or {}
            pred_slots = occupied_slot_set(pred_occ)
            target_slots = occupied_slot_set(target_occ)
            current_slots = occupied_slot_set(
                (world.get("lock_occupancy") or {}).get("current") or {}
            )
            tp, fp, fn = set_counts(pred_slots, target_slots)
            phase_counts[phase]["tp"] += tp
            phase_counts[phase]["fp"] += fp
            phase_counts[phase]["fn"] += fn
            for slot in sorted(pred_slots - target_slots):
                fp_slots[slot] += 1
            for slot in sorted(target_slots - pred_slots):
                fn_slots[slot] += 1
            for slot in sorted(current_slots | target_slots | pred_slots):
                transition = berth_transition_label(
                    slot in current_slots,
                    slot in target_slots,
                )
                transition_counts[transition]["total"] += 1
                if (slot in pred_slots) == (slot in target_slots):
                    transition_counts[transition]["correct"] += 1
                elif slot in pred_slots:
                    transition_counts[transition]["fp"] += 1
                else:
                    transition_counts[transition]["fn"] += 1
            record_multiset_confusion(
                coarse_confusion,
                coarse_count_counter(pred_occ),
                coarse_count_counter(target_occ),
            )
            record_multiset_confusion(
                motion_confusion,
                Counter(pred.get("motion_counts") or {}),
                target_motion_counter(target),
            )
        output[horizon_key] = {
            "num_targets": covered,
            "berth_by_phase": counter_prf_map(phase_counts),
            "berth_by_transition": transition_report(transition_counts),
            "top_false_positive_slots": fp_slots.most_common(5),
            "top_false_negative_slots": fn_slots.most_common(5),
            "top_coarse_count_confusions": coarse_confusion.most_common(8),
            "top_motion_count_confusions": motion_confusion.most_common(8),
        }
    return output


def berth_transition_label(current: bool, target: bool) -> str:
    if current and target:
        return "persist_occupied"
    if current and not target:
        return "clear"
    if not current and target:
        return "fill"
    return "remain_empty"


def counter_prf_map(counts: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    output = {}
    for key, counter in sorted(counts.items()):
        tp = int(counter.get("tp") or 0)
        fp = int(counter.get("fp") or 0)
        fn = int(counter.get("fn") or 0)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        output[key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return output


def transition_report(
    counts: dict[str, Counter[str]]
) -> dict[str, dict[str, Any]]:
    output = {}
    for key, counter in sorted(counts.items()):
        total = int(counter.get("total") or 0)
        correct = int(counter.get("correct") or 0)
        output[key] = {
            "total": total,
            "correct": correct,
            "accuracy": safe_div(correct, total),
            "fp": int(counter.get("fp") or 0),
            "fn": int(counter.get("fn") or 0),
        }
    return output


def record_multiset_confusion(
    output: Counter[tuple[str, str]],
    pred: Counter[str],
    target: Counter[str],
) -> None:
    labels = sorted(set(pred) | set(target))
    surplus = []
    missing = []
    for label in labels:
        pred_count = int(pred.get(label, 0))
        target_count = int(target.get(label, 0))
        if pred_count > target_count:
            surplus.extend([label] * (pred_count - target_count))
        elif target_count > pred_count:
            missing.extend([label] * (target_count - pred_count))
    while surplus and missing:
        output[(surplus.pop(0), missing.pop(0))] += 1
    for label in surplus:
        output[(label, "__extra__")] += 1
    for label in missing:
        output[("__missing__", label)] += 1


def per_track_berth_hybrid(
    per_track_pred: dict[str, Any], persistence_pred: dict[str, Any]
) -> dict[str, Any]:
    base_occ = persistence_pred.get("future_occupancy") or {}
    learned_occ = per_track_pred.get("future_occupancy") or {}
    num_ships = int(base_occ.get("num_ships", per_track_pred.get("num_ships", 0)) or 0)
    return {
        "future_occupancy": {
            **learned_occ,
            "berth_slots": base_occ.get("berth_slots") or [],
            "num_occupied_berths": base_occ.get("num_occupied_berths"),
            "num_ships": num_ships,
        },
        "motion_counts": motion_counts_from_track_predictions(
            per_track_pred.get("per_track_predictions") or [],
            max_tracks=num_ships,
        ),
        "num_ships": num_ships,
        "per_track_predictions": per_track_pred.get("per_track_predictions") or [],
    }


def deployable_persistence(world: dict[str, Any]) -> dict[str, Any]:
    occupancy = (world.get("lock_occupancy") or {}).get("current") or {}
    motion_items = (world.get("vessel_motion_flow") or {}).get("input_window") or []
    return {
        "future_occupancy": occupancy,
        "motion_counts": motion_counts_from_items(motion_items),
        "num_ships": int(occupancy.get("num_ships") or len(motion_items)),
    }


def dispatch_aware_rollout(world: dict[str, Any]) -> dict[str, Any]:
    pred = deployable_persistence(world)
    phase = (world.get("planner_feature_stitch") or {}).get("ship_operation_phase")
    counts = Counter(pred.get("motion_counts") or {})
    if phase == "ship_entering":
        counts["ship_entering_lock"] = max(counts.get("ship_entering_lock", 0), 1)
        counts.pop("ship_static", None)
        counts.pop("ship_berthed", None)
    elif phase == "ship_leaving":
        counts["ship_leaving_lock"] = max(counts.get("ship_leaving_lock", 0), 1)
        counts.pop("ship_static", None)
        counts.pop("ship_berthed", None)
    pred["motion_counts"] = dict(counts)
    return pred


def fit_classifier(x: Any, values: list[str], *, max_iter: int) -> dict[str, Any]:
    labels = [normalize_label(value) for value in values]
    counts = Counter(labels)
    if len(counts) <= 1:
        return {
            "kind": "constant",
            "value": labels[0] if labels else UNKNOWN,
            "class_counts": dict(counts),
        }
    head = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=0,
        ),
    )
    head.fit(x, labels)
    return {
        "kind": "classifier",
        "head": head,
        "class_counts": {str(key): int(value) for key, value in counts.items()},
    }


def fit_binary_classifier(x: Any, values: list[bool], *, max_iter: int) -> dict[str, Any]:
    labels = [bool(value) for value in values]
    counts = Counter(labels)
    if len(counts) <= 1:
        return {
            "kind": "constant_binary",
            "value": bool(labels[0]) if labels else False,
            "class_counts": {str(key): int(value) for key, value in counts.items()},
            "threshold": 0.5,
        }
    head = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=0,
        ),
    )
    head.fit(x, labels)
    probabilities = positive_probabilities(head, x)
    threshold_report = tune_probability_threshold(probabilities, labels)
    return {
        "kind": "binary_classifier",
        "head": head,
        "threshold": threshold_report["threshold"],
        "train_precision": threshold_report["precision"],
        "train_recall": threshold_report["recall"],
        "train_f1": threshold_report["f1"],
        "class_counts": {str(key): int(value) for key, value in counts.items()},
    }


def predict_label_with_score(head: dict[str, Any], x: Any) -> tuple[str, float]:
    if head.get("kind") == "constant":
        return normalize_label(head.get("value")), 1.0
    classifier = head["head"]
    label = normalize_label(classifier.predict(x)[0])
    score = 1.0
    if hasattr(classifier, "predict_proba"):
        classes = [normalize_label(value) for value in classifier.classes_]
        if label in classes:
            score = float(classifier.predict_proba(x)[0][classes.index(label)])
    return label, score


def predict_best_non_none_label_with_score(
    head: dict[str, Any], x: Any
) -> tuple[str, float]:
    if head.get("kind") == "constant":
        label = normalize_label(head.get("value"))
        return (label, 1.0) if label != NONE_BERTH else (NONE_BERTH, 0.0)
    classifier = head["head"]
    if not hasattr(classifier, "predict_proba"):
        return predict_label_with_score(head, x)
    classes = [normalize_label(value) for value in classifier.classes_]
    probabilities = classifier.predict_proba(x)[0]
    ranked = sorted(
        ((float(probability), label) for label, probability in zip(classes, probabilities)),
        reverse=True,
    )
    for probability, label in ranked:
        if label != NONE_BERTH:
            return label, probability
    return NONE_BERTH, 0.0


def predict_binary_with_score(head: dict[str, Any], x: Any) -> tuple[bool, float]:
    if head.get("kind") == "constant_binary":
        value = bool(head.get("value"))
        return value, 1.0 if value else 0.0
    probability = positive_probability(head["head"], x)
    threshold = float(head.get("threshold") if head.get("threshold") is not None else 0.5)
    return probability >= threshold, probability


def positive_probabilities(head: Any, x: Any) -> list[float]:
    classes = list(getattr(head, "classes_", []))
    if True in classes:
        positive_index = classes.index(True)
    elif 1 in classes:
        positive_index = classes.index(1)
    else:
        positive_index = len(classes) - 1
    probabilities = head.predict_proba(x)
    return [float(row[positive_index]) for row in probabilities]


def positive_probability(head: Any, x: Any) -> float:
    return positive_probabilities(head, x)[0]


def tune_probability_threshold(
    probabilities: list[float], labels: list[bool]
) -> dict[str, float]:
    candidates = sorted(set([0.0, 0.5, 1.0] + [float(value) for value in probabilities]))
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    best_rank = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        tp = fp = fn = 0
        for probability, label in zip(probabilities, labels):
            pred = probability >= threshold
            if pred and label:
                tp += 1
            elif pred and not label:
                fp += 1
            elif not pred and label:
                fn += 1
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        rank = (f1, precision, threshold)
        if rank > best_rank:
            best_rank = rank
            best = {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    return best


def empty_horizon_heads() -> dict[str, Any]:
    return {
        "num_train_rows": 0,
        "future_region": {"kind": "constant", "value": UNKNOWN},
        "future_berthed": {"kind": "constant_binary", "value": False},
        "future_berth_slot": {"kind": "constant", "value": NONE_BERTH},
        "future_motion_state": {"kind": "constant", "value": UNKNOWN},
    }


def motion_counts_from_world(world: dict[str, Any]) -> dict[str, int]:
    return motion_counts_from_items(
        (world.get("vessel_motion_flow") or {}).get("input_window") or []
    )


def motion_counts_from_items(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for item in items:
        counts[normalize_label(item.get("motion_state"))] += 1
    return dict(counts)


def track_delta_xy(track: TrackState) -> tuple[float, float]:
    if isinstance(track.delta_xy, list) and len(track.delta_xy) >= 2:
        return float_or_zero(track.delta_xy[0]), float_or_zero(track.delta_xy[1])
    return 0.0, 0.0


def horizon_target(row: dict[str, Any], horizon_key: str) -> Optional[dict[str, Any]]:
    target = (
        row.get("dense_ship_future_targets", {})
        .get("horizons", {})
        .get(horizon_key)
    )
    return target if isinstance(target, dict) else None


def collect_berth_slot_ids(
    world_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]
) -> list[str]:
    values = set()
    for row in world_rows:
        current = (row.get("lock_occupancy") or {}).get("current") or {}
        for slot in current.get("berth_slots") or []:
            label = normalize_berth_label(slot.get("region_id"))
            if label != NONE_BERTH:
                values.add(label)
    for row in label_rows:
        for target in iter_targets(row):
            for slot in (target.get("future_occupancy") or {}).get("berth_slots") or []:
                label = normalize_berth_label(slot.get("region_id"))
                if label != NONE_BERTH:
                    values.add(label)
    return sorted(values)


def collect_coarse_region_ids(
    world_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]
) -> list[str]:
    values = set(COARSE_REGIONS)
    for row in world_rows:
        current = (row.get("lock_occupancy") or {}).get("current") or {}
        for region in current.get("coarse_regions") or []:
            values.add(normalize_label(region.get("region_id")))
    for row in label_rows:
        for target in iter_targets(row):
            for region in (target.get("future_occupancy") or {}).get("coarse_regions") or []:
                values.add(normalize_label(region.get("region_id")))
    return sorted(values)


def collect_motion_labels(
    world_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]
) -> list[str]:
    values = set(DEFAULT_MOTION_LABELS)
    for row in world_rows:
        for item in (row.get("vessel_motion_flow") or {}).get("input_window") or []:
            values.add(normalize_label(item.get("motion_state")))
    for row in label_rows:
        for target in iter_targets(row):
            for ship in target.get("matched_ships") or []:
                values.add(normalize_label(ship.get("target_motion_state")))
    return sorted(values)


def iter_targets(row: dict[str, Any]):
    for target in (row.get("dense_ship_future_targets", {}) or {}).get(
        "horizons", {}
    ).values():
        if isinstance(target, dict):
            yield target


def normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else UNKNOWN


def normalize_berth_label(value: Any) -> str:
    text = str(value or "").strip()
    return text if text and text.lower() not in {"none", "null"} else NONE_BERTH


def normalize_category(value: Any) -> str:
    text = normalize_label(value).lower()
    if "cargo_fleet" in text:
        return "cargo_fleet"
    if "cargo_ship" in text:
        return "cargo_ship"
    if "container" in text:
        return "container_vessel"
    return text


def direction_from_scene(scene_token: Any) -> str:
    text = str(scene_token or "").lower()
    if "upstream" in text:
        return "upstream"
    if "downstream" in text:
        return "downstream"
    return UNKNOWN


def horizon_name(horizon_sec: int) -> str:
    return f"t_plus_{horizon_sec}s"


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


if __name__ == "__main__":
    main()
