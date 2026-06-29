#!/usr/bin/env python3
"""Train a deployable dense ship rollout head.

The head learns aggregate future ship state from deployable full-frame world
state. It predicts berth-slot counts, coarse-region counts, motion-state counts,
and total ship count at 10/20/30s. Identity matching is intentionally out of
scope because deployable Hydro3DNet/RTMDet tracks and annotation labels do not
share stable token IDs.
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
    build_predictions as build_heuristic_predictions,
    metrics_for_mode,
    motion_counts,
    read_jsonl,
    write_jsonl,
)
from tools.train_action_planner_head import as_float, as_int, float_or_zero


LEARNED_MODE = "learned_deployable_rollout"
HYBRID_MODE = "deployable_berth_learned_motion_rollout"
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
            "outputs/action_conditioned_world_model/deployable_ship_rollout_head.pkl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/"
            "summary_valtest_deployable_ship_rollout_head.json"
        ),
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/"
            "predictions_valtest_deployable_ship_rollout_head.jsonl"
        ),
    )
    parser.add_argument("--no-history-features", action="store_true")
    parser.add_argument("--max-iter", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_world = read_jsonl(args.train_world_state)
    train_labels = read_jsonl(args.train_dense_labels)
    eval_world = read_jsonl(args.eval_world_state)
    eval_labels = read_jsonl(args.eval_dense_labels)

    model = train_rollout_model(
        train_world,
        train_labels,
        use_history_features=not args.no_history_features,
        max_iter=args.max_iter,
    )
    world_by_sample = {str(row.get("sample_token")): row for row in eval_world}
    predictions = build_predictions(eval_labels, world_by_sample, model)
    summary = build_summary(
        train_world,
        train_labels,
        eval_labels,
        predictions,
        model=model,
        train_world_state=args.train_world_state,
        train_dense_labels=args.train_dense_labels,
        eval_world_state=args.eval_world_state,
        eval_dense_labels=args.eval_dense_labels,
        prediction_output=args.prediction_output,
        output_model=args.output_model,
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
    for horizon, metrics in summary["rollout_metrics"][LEARNED_MODE].items():
        print(
            f"{horizon}: berth_f1={metrics['berth_occupied_f1']:.3f} "
            f"coarse_f1={metrics['coarse_region_count_f1']:.3f} "
            f"motion_f1={metrics['motion_count_f1']:.3f}"
        )


def train_rollout_model(
    world_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    *,
    use_history_features: bool = True,
    max_iter: int = 3000,
) -> dict[str, Any]:
    world_by_sample = {str(row.get("sample_token")): row for row in world_rows}
    train_rows = matched_label_rows(label_rows, world_by_sample)
    if not train_rows:
        raise SystemExit("no dense ship labels matched deployable world state")

    berth_slot_ids = collect_berth_slot_ids(world_rows, label_rows)
    coarse_region_ids = collect_coarse_region_ids(world_rows, label_rows)
    motion_labels = collect_motion_labels(world_rows, label_rows)
    history_by_sample = (
        build_world_history_features(world_rows) if use_history_features else {}
    )

    vectorizer = DictVectorizer(sparse=True)
    train_features = [
        rollout_features(
            world_by_sample[str(row.get("sample_token"))],
            history_features=history_by_sample.get(str(row.get("sample_token"))),
        )
        for row in train_rows
    ]
    vectorizer.fit(train_features)

    heads: dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_key = horizon_name(horizon)
        horizon_rows = rows_with_horizon_target(train_rows, horizon_key)
        if not horizon_rows:
            heads[horizon_key] = empty_horizon_heads(
                berth_slot_ids, coarse_region_ids, motion_labels
            )
            continue
        x = vectorizer.transform(
            [
                rollout_features(
                    world_by_sample[str(row.get("sample_token"))],
                    history_features=history_by_sample.get(str(row.get("sample_token"))),
                )
                for row in horizon_rows
            ]
        )
        heads[horizon_key] = {
            "num_train_rows": len(horizon_rows),
            "num_ships": fit_count_head(
                x,
                [target_num_ships(row, horizon_key) for row in horizon_rows],
                max_iter=max_iter,
            ),
            "berth_slot_counts": {
                slot_id: fit_count_head(
                    x,
                    [
                        target_berth_counts(row, horizon_key).get(slot_id, 0)
                        for row in horizon_rows
                    ],
                    max_iter=max_iter,
                )
                for slot_id in berth_slot_ids
            },
            "berth_slot_occupancy": {
                slot_id: fit_binary_head(
                    x,
                    [
                        target_berth_counts(row, horizon_key).get(slot_id, 0) > 0
                        for row in horizon_rows
                    ],
                    max_iter=max_iter,
                )
                for slot_id in berth_slot_ids
            },
            "berth_slot_delta": {
                slot_id: fit_berth_delta_heads(
                    x,
                    horizon_rows,
                    world_by_sample=world_by_sample,
                    slot_id=slot_id,
                    horizon_key=horizon_key,
                    max_iter=max_iter,
                )
                for slot_id in berth_slot_ids
            },
            "coarse_region_counts": {
                region_id: fit_count_head(
                    x,
                    [
                        target_coarse_counts(row, horizon_key).get(region_id, 0)
                        for row in horizon_rows
                    ],
                    max_iter=max_iter,
                )
                for region_id in coarse_region_ids
            },
            "motion_counts": {
                label: fit_count_head(
                    x,
                    [
                        target_motion_counts(row, horizon_key).get(label, 0)
                        for row in horizon_rows
                    ],
                    max_iter=max_iter,
                )
                for label in motion_labels
            },
        }

    return {
        "vectorizer": vectorizer,
        "heads": heads,
        "horizons": list(HORIZONS),
        "berth_slot_ids": list(berth_slot_ids),
        "coarse_region_ids": list(coarse_region_ids),
        "motion_labels": list(motion_labels),
        "use_history_features": use_history_features,
        "num_train_label_frames": len(label_rows),
        "matched_train_frames": len(train_rows),
        "num_features": len(vectorizer.feature_names_),
        "feature_policy": {
            "input": "deployable full-frame world state",
            "target": "dense future aggregate ship state",
            "identity": "not predicted",
        },
    }


def build_predictions(
    label_rows: list[dict[str, Any]],
    world_by_sample: dict[str, dict[str, Any]],
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_rows = build_heuristic_predictions(label_rows, world_by_sample)
    baseline_by_sample = {str(row.get("sample_token")): row for row in baseline_rows}
    history_by_sample = (
        build_world_history_features(list(world_by_sample.values()))
        if model.get("use_history_features")
        else {}
    )
    predictions = []
    for label in label_rows:
        sample_token = str(label.get("sample_token"))
        world = world_by_sample.get(sample_token)
        if world is None:
            continue
        features = rollout_features(
            world,
            history_features=history_by_sample.get(sample_token),
        )
        x = model["vectorizer"].transform([features])
        learned_horizons = {}
        for horizon in model.get("horizons") or HORIZONS:
            horizon_key = horizon_name(int(horizon))
            learned_horizons[horizon_key] = predict_horizon(
                model,
                x,
                horizon_key,
                current_slot_counts=current_berth_counts(
                    world,
                    model.get("berth_slot_ids") or [],
                ),
            )
        row = dict(
            baseline_by_sample.get(
                sample_token,
                {
                    "sample_token": sample_token,
                    "split": label.get("split"),
                    "scene_token": label.get("scene_token"),
                    "timestamp": label.get("timestamp"),
                    "timestamp_str": label.get("timestamp_str"),
                    "rollout_modes": {},
                },
            )
        )
        row.setdefault("rollout_modes", {})
        row["rollout_modes"][LEARNED_MODE] = learned_horizons
        row["rollout_modes"][HYBRID_MODE] = hybrid_horizons(row["rollout_modes"])
        predictions.append(row)
    return predictions


def build_summary(
    train_world_rows: list[dict[str, Any]],
    train_label_rows: list[dict[str, Any]],
    eval_label_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    train_world_state: Path,
    train_dense_labels: Path,
    eval_world_state: Path,
    eval_dense_labels: Path,
    prediction_output: Path,
    output_model: Path,
) -> dict[str, Any]:
    pred_by_sample = {str(row.get("sample_token")): row for row in predictions}
    modes = [
        "deployable_persistence",
        "dispatch_aware_rollout",
        LEARNED_MODE,
        HYBRID_MODE,
    ]
    return {
        "train_world_state": str(train_world_state),
        "train_dense_labels": str(train_dense_labels),
        "eval_world_state": str(eval_world_state),
        "eval_dense_labels": str(eval_dense_labels),
        "output_model": str(output_model),
        "prediction_output": str(prediction_output),
        "num_train_world_state_frames": len(train_world_rows),
        "num_train_label_frames": len(train_label_rows),
        "matched_train_frames": int(model.get("matched_train_frames") or 0),
        "num_eval_label_frames": len(eval_label_rows),
        "matched_eval_frames": len(predictions),
        "num_features": int(model.get("num_features") or 0),
        "use_history_features": bool(model.get("use_history_features")),
        "horizon_train_rows": {
            key: int(head.get("num_train_rows") or 0)
            for key, head in (model.get("heads") or {}).items()
        },
        "rollout_metrics": {
            mode: metrics_for_mode(eval_label_rows, pred_by_sample, mode)
            for mode in modes
        },
        "notes": [
            "Gate/water/lock state is treated as an available precise input; this head only learns dense ship rollout.",
            "The learned deployable rollout predicts aggregate berth, coarse-region, motion, and ship-count state from deployable world state.",
            "The learned berth branch is a model-triggered fill/clear correction over current deployable berth occupancy.",
            "The hybrid rollout uses deployable persistence for berth/total count and learned heads for coarse-region and motion counts because current learned berth corrections are still weaker than persistence.",
            "Identity is not evaluated because Hydro3DNet/RTMDet track IDs and annotation instance tokens are not shared.",
            "deployable_persistence and dispatch_aware_rollout are included as comparable heuristic baselines.",
        ],
    }


def matched_label_rows(
    label_rows: list[dict[str, Any]],
    world_by_sample: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row for row in label_rows
        if str(row.get("sample_token")) in world_by_sample
    ]


def rows_with_horizon_target(
    label_rows: list[dict[str, Any]], horizon_key: str
) -> list[dict[str, Any]]:
    return [
        row for row in label_rows
        if isinstance(horizon_target(row, horizon_key), dict)
    ]


def rollout_features(
    world: dict[str, Any],
    *,
    history_features: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    current = (world.get("lock_occupancy") or {}).get("current") or {}
    motion_items = (world.get("vessel_motion_flow") or {}).get("input_window") or []
    stitch = world.get("planner_feature_stitch") or {}
    track = world.get("track_source") or {}
    recovery = track.get("rtmdet_recovery_summary") or {}
    feat: dict[str, Any] = {
        "direction": direction_from_scene(world.get("scene_token")),
        "ship_operation_phase": stitch.get("ship_operation_phase") or "missing",
        "stitch_source": stitch.get("source") or "missing",
        "num_ships": float_or_zero(current.get("num_ships")),
        "num_occupied_berths": float_or_zero(current.get("num_occupied_berths")),
        "window_size": float_or_zero(track.get("window_size")),
        "rtmdet_recovered_detections": float_or_zero(
            recovery.get("recovered_detections")
        ),
        "rtmdet_recovered_frames": float_or_zero(recovery.get("recovered_frames")),
    }

    for slot in current.get("berth_slots") or []:
        slot_id = str(slot.get("region_id") or "unknown")
        count = int(slot.get("ship_count") or 0)
        feat[f"berth_{slot_id}_count"] = count
        feat[f"berth_{slot_id}_occupied"] = count > 0 or bool(slot.get("occupied"))
    for region in current.get("coarse_regions") or []:
        region_id = str(region.get("region_id") or "unknown")
        feat[f"coarse_{region_id}_count"] = int(region.get("ship_count") or 0)

    add_motion_features(feat, motion_items)
    add_next_ship_features(feat, "next_enter", stitch.get("next_ship_to_enter_weak"))
    add_next_ship_features(feat, "next_leave", stitch.get("next_ship_to_leave_weak"))
    add_history_features(feat, history_features)
    return feat


def add_motion_features(feat: dict[str, Any], motion_items: list[dict[str, Any]]) -> None:
    counts = motion_counts(motion_items)
    speeds = []
    dx_values = []
    dy_values = []
    for item in motion_items:
        motion = str(item.get("motion_state") or "unknown")
        start_region = str(item.get("start_region") or "unknown")
        end_region = str(item.get("end_region") or "unknown")
        direction = str(item.get("direction_label") or "unknown")
        category = str(item.get("category") or "unknown")
        feat[f"motion_{motion}_count"] = counts[motion]
        feat[f"motion_path_{start_region}_to_{end_region}"] = (
            int(feat.get(f"motion_path_{start_region}_to_{end_region}") or 0) + 1
        )
        feat[f"motion_direction_{direction}"] = (
            int(feat.get(f"motion_direction_{direction}") or 0) + 1
        )
        feat[f"motion_category_{category}"] = (
            int(feat.get(f"motion_category_{category}") or 0) + 1
        )
        speed = as_float(item.get("end_speed_mps"))
        if speed is not None:
            speeds.append(speed)
        delta_xy = item.get("delta_xy")
        if isinstance(delta_xy, list) and len(delta_xy) >= 2:
            dx = as_float(delta_xy[0])
            dy = as_float(delta_xy[1])
            if dx is not None:
                dx_values.append(dx)
            if dy is not None:
                dy_values.append(dy)
    feat["motion_item_count"] = len(motion_items)
    feat["motion_speed_max"] = max(speeds) if speeds else 0.0
    feat["motion_speed_mean"] = float(sum(speeds) / len(speeds)) if speeds else 0.0
    feat["motion_abs_dx_sum"] = float(sum(abs(value) for value in dx_values))
    feat["motion_abs_dy_sum"] = float(sum(abs(value) for value in dy_values))
    feat["motion_dy_sum"] = float(sum(dy_values))


def add_next_ship_features(feat: dict[str, Any], prefix: str, value: Any) -> None:
    if not isinstance(value, dict):
        feat[f"{prefix}_exists"] = False
        feat[f"{prefix}_side"] = "missing"
        feat[f"{prefix}_source"] = "missing"
        feat[f"{prefix}_category"] = "missing"
        feat[f"{prefix}_distance_to_gate_m"] = 0.0
        feat[f"{prefix}_speed_mps"] = 0.0
        return
    feat[f"{prefix}_exists"] = True
    feat[f"{prefix}_side"] = value.get("side") or "missing"
    feat[f"{prefix}_source"] = value.get("source") or "missing"
    feat[f"{prefix}_category"] = value.get("category") or "missing"
    feat[f"{prefix}_distance_to_gate_m"] = float_or_zero(value.get("distance_to_gate_m"))
    feat[f"{prefix}_speed_mps"] = float_or_zero(value.get("speed_mps"))


def build_world_history_features(
    world_rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in world_rows:
        by_scene[(str(row.get("split")), str(row.get("scene_token")))].append(row)

    output: dict[str, dict[str, Any]] = {}
    for _, rows in by_scene.items():
        rows.sort(
            key=lambda row: (
                int(row.get("timestamp") or 0),
                str(row.get("sample_token")),
            )
        )
        previous: Optional[dict[str, Any]] = None
        phase_runs: dict[str, Any] = {}
        for row in rows:
            features = history_from_previous_world(row, previous)
            phase = (row.get("planner_feature_stitch") or {}).get(
                "ship_operation_phase"
            ) or "missing"
            timestamp = as_int(row.get("timestamp"))
            if phase_runs.get("value") != phase:
                phase_runs = {
                    "value": phase,
                    "start_timestamp": timestamp,
                    "frame_count": 0,
                }
            phase_runs["frame_count"] = int(phase_runs.get("frame_count") or 0) + 1
            start = phase_runs.get("start_timestamp")
            run_sec = 0.0
            if timestamp is not None and start is not None:
                run_sec = max(0.0, (timestamp - int(start)) / 1_000_000.0)
            features["ship_operation_phase_run_sec"] = run_sec
            features["ship_operation_phase_run_frame_count"] = phase_runs["frame_count"]
            output[str(row.get("sample_token"))] = features
            previous = row
    return output


def history_from_previous_world(
    row: dict[str, Any], previous: Optional[dict[str, Any]]
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "has_prev_world": previous is not None,
        "dt_prev_sec": None,
        "num_ships_delta_prev": None,
        "num_occupied_berths_delta_prev": None,
        "ship_operation_phase_changed_from_prev": False,
        "prev_ship_operation_phase": None,
    }
    if previous is None:
        return features
    timestamp = as_int(row.get("timestamp"))
    prev_timestamp = as_int(previous.get("timestamp"))
    if timestamp is not None and prev_timestamp is not None:
        features["dt_prev_sec"] = max(0.0, (timestamp - prev_timestamp) / 1_000_000.0)
    current_occ = (row.get("lock_occupancy") or {}).get("current") or {}
    prev_occ = (previous.get("lock_occupancy") or {}).get("current") or {}
    features["num_ships_delta_prev"] = (
        float_or_zero(current_occ.get("num_ships"))
        - float_or_zero(prev_occ.get("num_ships"))
    )
    features["num_occupied_berths_delta_prev"] = (
        float_or_zero(current_occ.get("num_occupied_berths"))
        - float_or_zero(prev_occ.get("num_occupied_berths"))
    )
    current_phase = (row.get("planner_feature_stitch") or {}).get("ship_operation_phase")
    prev_phase = (previous.get("planner_feature_stitch") or {}).get(
        "ship_operation_phase"
    )
    features["prev_ship_operation_phase"] = prev_phase
    features["ship_operation_phase_changed_from_prev"] = (
        current_phase is not None and prev_phase is not None and current_phase != prev_phase
    )
    return features


def add_history_features(
    feat: dict[str, Any], history_features: Optional[dict[str, Any]]
) -> None:
    history_features = history_features or {}
    categorical = ("prev_ship_operation_phase",)
    boolean = ("has_prev_world", "ship_operation_phase_changed_from_prev")
    numeric = (
        "dt_prev_sec",
        "num_ships_delta_prev",
        "num_occupied_berths_delta_prev",
        "ship_operation_phase_run_sec",
        "ship_operation_phase_run_frame_count",
    )
    for field in categorical:
        feat[f"hist_{field}"] = history_features.get(field) or "missing"
    for field in boolean:
        feat[f"hist_{field}"] = bool(history_features.get(field))
    for field in numeric:
        feat[f"hist_{field}"] = float_or_zero(history_features.get(field))
        feat[f"hist_{field}_missing"] = history_features.get(field) is None


def fit_count_head(x: Any, values: list[int], *, max_iter: int) -> dict[str, Any]:
    labels = [str(max(0, int(value))) for value in values]
    counts = Counter(labels)
    if len(counts) <= 1:
        return {"kind": "constant", "value": int(labels[0]) if labels else 0}
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


def fit_binary_head(x: Any, values: list[bool], *, max_iter: int) -> dict[str, Any]:
    labels = [bool(value) for value in values]
    counts = Counter(labels)
    if len(counts) <= 1:
        return {
            "kind": "constant_binary",
            "value": bool(labels[0]) if labels else False,
            "class_counts": {str(key): int(value) for key, value in counts.items()},
        }
    head = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver="lbfgs",
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


def fit_berth_delta_heads(
    x: Any,
    rows: list[dict[str, Any]],
    *,
    world_by_sample: dict[str, dict[str, Any]],
    slot_id: str,
    horizon_key: str,
    max_iter: int,
) -> dict[str, Any]:
    fill_labels = []
    clear_labels = []
    for row in rows:
        world = world_by_sample[str(row.get("sample_token"))]
        current = current_berth_counts(world, [slot_id]).get(slot_id, 0) > 0
        target = target_berth_counts(row, horizon_key).get(slot_id, 0) > 0
        fill_labels.append((not current) and target)
        clear_labels.append(current and (not target))
    return {
        "fill": fit_binary_head(x, fill_labels, max_iter=max_iter),
        "clear": fit_binary_head(x, clear_labels, max_iter=max_iter),
    }


def predict_horizon(
    model: dict[str, Any],
    x: Any,
    horizon_key: str,
    *,
    current_slot_counts: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    heads = (model.get("heads") or {}).get(horizon_key)
    if not isinstance(heads, dict):
        heads = empty_horizon_heads(
            model.get("berth_slot_ids") or [],
            model.get("coarse_region_ids") or COARSE_REGIONS,
            model.get("motion_labels") or DEFAULT_MOTION_LABELS,
        )
    coarse_counts = {
        region_id: predict_count(head, x)
        for region_id, head in (heads.get("coarse_region_counts") or {}).items()
    }
    num_ships = predict_count(
        heads.get("num_ships") or {"kind": "constant", "value": 0},
        x,
    )
    slot_counts = predict_berth_slot_counts(
        heads,
        x,
        num_ships,
        current_slot_counts=current_slot_counts,
    )
    motion_candidates = {
        label: predict_count_with_score(head, x)
        for label, head in (heads.get("motion_counts") or {}).items()
    }
    predicted_motion_counts = normalize_motion_counts(motion_candidates, num_ships)
    return {
        "future_occupancy": occupancy_from_counts(slot_counts, coarse_counts, num_ships),
        "motion_counts": predicted_motion_counts,
        "num_ships": num_ships,
    }


def predict_count(head: dict[str, Any], x: Any) -> int:
    return predict_count_with_score(head, x)[0]


def predict_count_with_score(head: dict[str, Any], x: Any) -> tuple[int, float]:
    if head.get("kind") == "constant":
        return max(0, int(head.get("value") or 0)), 1.0
    label = str(head["head"].predict(x)[0])
    score = 1.0
    classifier = head.get("head")
    if hasattr(classifier, "predict_proba"):
        classes = [str(value) for value in classifier.classes_]
        if label in classes:
            proba = classifier.predict_proba(x)[0]
            score = float(proba[classes.index(label)])
    return max(0, int(float(label))), score


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


def predict_berth_slot_counts(
    heads: dict[str, Any],
    x: Any,
    num_ships: int,
    *,
    current_slot_counts: Optional[dict[str, int]] = None,
) -> dict[str, int]:
    delta_heads = heads.get("berth_slot_delta") or {}
    if delta_heads and current_slot_counts is not None:
        return predict_berth_delta_counts(delta_heads, x, current_slot_counts)
    occupancy_heads = heads.get("berth_slot_occupancy") or {}
    if not occupancy_heads:
        return {
            slot_id: predict_count(head, x)
            for slot_id, head in (heads.get("berth_slot_counts") or {}).items()
        }
    candidates = {
        slot_id: predict_binary_with_score(head, x)
        for slot_id, head in occupancy_heads.items()
    }
    return normalize_berth_slot_counts(candidates, num_ships)


def predict_berth_delta_counts(
    delta_heads: dict[str, Any],
    x: Any,
    current_slot_counts: dict[str, int],
) -> dict[str, int]:
    output = {str(slot_id): int(count or 0) for slot_id, count in current_slot_counts.items()}
    for slot_id, heads in delta_heads.items():
        current = output.get(slot_id, 0) > 0
        if current:
            should_clear, _ = predict_binary_with_score(heads["clear"], x)
            if should_clear:
                output[slot_id] = 0
        else:
            should_fill, _ = predict_binary_with_score(heads["fill"], x)
            if should_fill:
                output[slot_id] = 1
    for slot_id in delta_heads:
        output.setdefault(slot_id, 0)
    return output


def normalize_berth_slot_counts(
    candidates: dict[str, tuple[bool, float]], num_ships: int
) -> dict[str, int]:
    occupied = [
        (score, slot_id)
        for slot_id, (is_occupied, score) in candidates.items()
        if is_occupied
    ]
    occupied.sort(reverse=True)
    if num_ships <= 0:
        kept = []
    else:
        kept = occupied[:num_ships]
    kept_slots = {slot_id for _, slot_id in kept}
    return {slot_id: 1 if slot_id in kept_slots else 0 for slot_id in candidates}


def normalize_motion_counts(
    motion_candidates: dict[str, tuple[int, float]],
    num_ships: int,
) -> dict[str, int]:
    counts = {
        label: count
        for label, (count, _) in motion_candidates.items()
        if count > 0
    }
    if num_ships <= 0:
        return {}
    total = sum(counts.values())
    if total <= num_ships:
        return counts

    expanded = []
    for label, count in counts.items():
        score = motion_candidates[label][1]
        for _ in range(count):
            expanded.append((score, label))
    expanded.sort(reverse=True)
    kept = Counter(label for _, label in expanded[:num_ships])
    return dict(kept)


def hybrid_horizons(rollout_modes: dict[str, Any]) -> dict[str, Any]:
    persistence = rollout_modes.get("deployable_persistence") or {}
    learned = rollout_modes.get(LEARNED_MODE) or {}
    output = {}
    for horizon_key, learned_pred in learned.items():
        base = persistence.get(horizon_key) or {}
        base_occ = base.get("future_occupancy") or {}
        learned_occ = learned_pred.get("future_occupancy") or {}
        num_ships = int(base_occ.get("num_ships") or base.get("num_ships") or 0)
        motion_counts = normalize_motion_count_dict(
            learned_pred.get("motion_counts") or {},
            num_ships,
        )
        output[horizon_key] = {
            "future_occupancy": {
                **learned_occ,
                "berth_slots": base_occ.get("berth_slots") or [],
                "num_occupied_berths": base_occ.get("num_occupied_berths"),
                "num_ships": num_ships,
            },
            "motion_counts": motion_counts,
            "num_ships": num_ships,
        }
    return output


def normalize_motion_count_dict(counts: dict[str, Any], num_ships: int) -> dict[str, int]:
    int_counts = {
        str(label): max(0, int(count or 0))
        for label, count in counts.items()
        if int(count or 0) > 0
    }
    if num_ships <= 0:
        return {}
    total = sum(int_counts.values())
    if total <= num_ships:
        return int_counts
    expanded = []
    for label, count in int_counts.items():
        for _ in range(count):
            expanded.append(label)
    kept = Counter(expanded[:num_ships])
    return dict(kept)


def occupancy_from_counts(
    slot_counts: dict[str, int],
    coarse_counts: dict[str, int],
    num_ships: int,
) -> dict[str, Any]:
    return {
        "berth_slots": [
            {
                "region_id": slot_id,
                "occupied": count > 0,
                "ship_count": count,
                "ship_tokens": synthetic_tokens("learned_berth", slot_id, count),
            }
            for slot_id, count in sorted(slot_counts.items())
        ],
        "coarse_regions": [
            {
                "region_id": region_id,
                "ship_count": count,
                "ship_tokens": synthetic_tokens("learned_region", region_id, count),
            }
            for region_id, count in sorted(coarse_counts.items())
        ],
        "num_occupied_berths": sum(1 for count in slot_counts.values() if count > 0),
        "num_ships": num_ships,
    }


def synthetic_tokens(prefix: str, region_id: str, count: int) -> list[str]:
    safe_region = region_id.replace(" ", "_")
    return [f"{prefix}_{safe_region}_{index:02d}" for index in range(1, count + 1)]


def empty_horizon_heads(
    berth_slot_ids: Iterable[str],
    coarse_region_ids: Iterable[str],
    motion_labels: Iterable[str],
) -> dict[str, Any]:
    return {
        "num_train_rows": 0,
        "num_ships": {"kind": "constant", "value": 0},
        "berth_slot_counts": {
            str(slot_id): {"kind": "constant", "value": 0}
            for slot_id in berth_slot_ids
        },
        "berth_slot_occupancy": {
            str(slot_id): {"kind": "constant_binary", "value": False}
            for slot_id in berth_slot_ids
        },
        "berth_slot_delta": {
            str(slot_id): {
                "fill": {"kind": "constant_binary", "value": False},
                "clear": {"kind": "constant_binary", "value": False},
            }
            for slot_id in berth_slot_ids
        },
        "coarse_region_counts": {
            str(region_id): {"kind": "constant", "value": 0}
            for region_id in coarse_region_ids
        },
        "motion_counts": {
            str(label): {"kind": "constant", "value": 0}
            for label in motion_labels
        },
    }


def target_num_ships(row: dict[str, Any], horizon_key: str) -> int:
    target = horizon_target(row, horizon_key) or {}
    occupancy = target.get("future_occupancy") or {}
    value = occupancy.get("num_ships")
    if value is not None:
        return max(0, int(value or 0))
    return len(target.get("matched_ships") or [])


def target_berth_counts(row: dict[str, Any], horizon_key: str) -> Counter[str]:
    target = horizon_target(row, horizon_key) or {}
    occupancy = target.get("future_occupancy") or {}
    counts = Counter()
    for slot in occupancy.get("berth_slots") or []:
        counts[str(slot.get("region_id") or "unknown")] += int(slot.get("ship_count") or 0)
    return counts


def current_berth_counts(world: dict[str, Any], berth_slot_ids: Iterable[str]) -> dict[str, int]:
    current = (world.get("lock_occupancy") or {}).get("current") or {}
    counts = {str(slot_id): 0 for slot_id in berth_slot_ids}
    for slot in current.get("berth_slots") or []:
        slot_id = str(slot.get("region_id") or "unknown")
        count = int(slot.get("ship_count") or 0)
        if count <= 0 and slot.get("occupied"):
            count = 1
        counts[slot_id] = max(0, count)
    return counts


def target_coarse_counts(row: dict[str, Any], horizon_key: str) -> Counter[str]:
    target = horizon_target(row, horizon_key) or {}
    occupancy = target.get("future_occupancy") or {}
    counts = Counter()
    for region in occupancy.get("coarse_regions") or []:
        counts[str(region.get("region_id") or "unknown")] += int(
            region.get("ship_count") or 0
        )
    return counts


def target_motion_counts(row: dict[str, Any], horizon_key: str) -> Counter[str]:
    target = horizon_target(row, horizon_key) or {}
    counts = Counter()
    for ship in target.get("matched_ships") or []:
        counts[str(ship.get("target_motion_state") or "unknown")] += 1
    return counts


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
            values.add(str(slot.get("region_id") or "unknown"))
    for row in label_rows:
        for target in iter_targets(row):
            for slot in (target.get("future_occupancy") or {}).get("berth_slots") or []:
                values.add(str(slot.get("region_id") or "unknown"))
    return sorted(values)


def collect_coarse_region_ids(
    world_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]
) -> list[str]:
    values = set(COARSE_REGIONS)
    for row in world_rows:
        current = (row.get("lock_occupancy") or {}).get("current") or {}
        for region in current.get("coarse_regions") or []:
            values.add(str(region.get("region_id") or "unknown"))
    for row in label_rows:
        for target in iter_targets(row):
            for region in (target.get("future_occupancy") or {}).get("coarse_regions") or []:
                values.add(str(region.get("region_id") or "unknown"))
    return sorted(values)


def collect_motion_labels(
    world_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]
) -> list[str]:
    values = set(DEFAULT_MOTION_LABELS)
    for row in world_rows:
        for item in (row.get("vessel_motion_flow") or {}).get("input_window") or []:
            values.add(str(item.get("motion_state") or "unknown"))
    for row in label_rows:
        for target in iter_targets(row):
            for ship in target.get("matched_ships") or []:
                values.add(str(ship.get("target_motion_state") or "unknown"))
    return sorted(values)


def iter_targets(row: dict[str, Any]):
    for target in (row.get("dense_ship_future_targets", {}) or {}).get(
        "horizons", {}
    ).values():
        if isinstance(target, dict):
            yield target


def direction_from_scene(scene_token: Any) -> str:
    text = str(scene_token or "").lower()
    if "upstream" in text:
        return "upstream"
    if "downstream" in text:
        return "downstream"
    return "unknown"


def horizon_name(horizon_sec: int) -> str:
    return f"t_plus_{horizon_sec}s"


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


if __name__ == "__main__":
    main()
