#!/usr/bin/env python3
"""Train a structured action/planner head on action-conditioned candidates."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
LEAKY_CURRENT_STATE_FIELDS = {
    "valid_actions",
    "invalid_actions",
    "violation_reason",
    "observed_action",
    "action_target",
    "action_source",
    "action_confidence",
    "action_start_time",
    "action_end_time",
    "ship_dispatch_action",
    "ship_dispatch_targets",
    "ship_dispatch_target_count",
    "ship_dispatch_source",
    "ship_dispatch_confidence",
    "ship_dispatch_conflict",
}
CATEGORICAL_FIELDS = {
    "upper_gate_state",
    "lower_gate_state",
    "water_state",
    "operation_phase",
    "ship_operation_phase",
}
BOOLEAN_FIELDS = {
    "no_ship_in_upper_gate_zone",
    "no_ship_in_lower_gate_zone",
    "entry_path_clear",
    "exit_path_clear",
    "chamber_capacity_available",
    "all_in_chamber_ships_berthed_or_static",
    "no_ship_entering_or_leaving_inside_chamber",
}
NUMERIC_FIELDS = {
    "water_level",
    "upstream_water_level",
    "downstream_water_level",
    "num_occupied_berths",
    "num_ships_in_chamber",
    "max_parallel_entries",
    "max_parallel_departures",
}
HISTORY_CATEGORICAL_FIELDS = (
    "prev_upper_gate_state",
    "prev_lower_gate_state",
    "prev_water_state",
    "prev_operation_phase",
    "prev_ship_operation_phase",
)
HISTORY_BOOLEAN_FIELDS = (
    "has_prev_frame",
    "water_state_changed_from_prev",
    "operation_phase_changed_from_prev",
    "upper_gate_state_changed_from_prev",
    "lower_gate_state_changed_from_prev",
)
HISTORY_NUMERIC_FIELDS = (
    "dt_prev_sec",
    "water_level_delta_prev",
    "water_level_abs_delta_prev",
    "water_level_slope_prev",
    "upstream_water_level_delta_prev",
    "downstream_water_level_delta_prev",
    "upper_water_abs_diff_delta_prev",
    "lower_water_abs_diff_delta_prev",
    "water_state_run_sec",
    "water_state_run_frame_count",
    "operation_phase_run_sec",
    "operation_phase_run_frame_count",
    "ship_operation_phase_run_sec",
    "ship_operation_phase_run_frame_count",
    "upper_gate_state_run_sec",
    "upper_gate_state_run_frame_count",
    "lower_gate_state_run_sec",
    "lower_gate_state_run_frame_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-candidates",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/action_conditioned_train_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--eval-candidates",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/action_conditioned_valtest_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("outputs/action_conditioned_world_model/action_planner_head.pkl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/summary_valtest_action_planner_head.json"
        ),
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/predictions_valtest_action_planner_head.jsonl"
        ),
    )
    parser.add_argument(
        "--include-observed-action-features",
        action="store_true",
        help=(
            "Include observed_action/action metadata and ship_dispatch fields in "
            "features. Defaults off to avoid target leakage."
        ),
    )
    parser.add_argument(
        "--no-hard-mask",
        action="store_true",
        help="Do not force final planner prediction to be one of the valid actions.",
    )
    parser.add_argument(
        "--no-history-features",
        action="store_true",
        help=(
            "Disable online-safe temporal features such as previous-frame water "
            "level slope and state run duration."
        ),
    )
    parser.add_argument(
        "--no-dispatch-continuity-override",
        action="store_true",
        help=(
            "Disable the conservative planner postprocess that keeps dispatch_exit "
            "active while a leaving ship is still present and dispatch_exit is valid."
        ),
    )
    parser.add_argument("--max-iter", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_candidates)
    eval_rows = read_jsonl(args.eval_candidates)
    model = train_heads(
        train_rows,
        include_observed_action_features=args.include_observed_action_features,
        use_history_features=not args.no_history_features,
        max_iter=args.max_iter,
    )
    eval_result = evaluate_model(
        model,
        eval_rows,
        hard_mask=not args.no_hard_mask,
        include_observed_action_features=args.include_observed_action_features,
        dispatch_continuity_override=not args.no_dispatch_continuity_override,
    )
    summary = {
        "train_candidates": str(args.train_candidates),
        "eval_candidates": str(args.eval_candidates),
        "output_model": str(args.output_model),
        "prediction_output": str(args.prediction_output),
        "include_observed_action_features": args.include_observed_action_features,
        "use_history_features": not args.no_history_features,
        "dispatch_continuity_override": not args.no_dispatch_continuity_override,
        "hard_mask": not args.no_hard_mask,
        "num_train_candidates": len(train_rows),
        "num_eval_candidates": len(eval_rows),
        "num_features": len(model["vectorizer"].feature_names_),
        "train_label_counts": label_counts(train_rows),
        "eval": eval_result["summary"],
        "notes": [
            "Default features exclude valid_actions/violation_reason and observed action labels to reduce leakage.",
            "History features use only earlier frames from the same scene; weak wave labels derived from water_state are not used as independent evidence.",
            "The dispatch continuity override is a rule-constrained planner postprocess, not a learned feature.",
            "Final planner prediction is selected per frame from candidate planner scores; hard-mask mode restricts selection to valid actions.",
            "validity_head is diagnostic in hard-mask mode because rule labels are already available.",
        ],
    }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with args.output_model.open("wb") as handle:
        pickle.dump(model, handle)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.prediction_output, eval_result["predictions"])

    print(f"wrote_model={args.output_model}")
    print(f"wrote_summary={args.summary_output}")
    print(f"wrote_predictions={args.prediction_output}")
    print(f"num_features={summary['num_features']}")
    print(f"validity_f1={summary['eval']['candidate_validity']['positive_f1']:.3f}")
    print(f"planner_candidate_f1={summary['eval']['candidate_planner']['positive_f1']:.3f}")
    print(
        "frame_action_head "
        f"legal={summary['eval']['frame_action_head']['legal_rate']:.3f} "
        f"target={summary['eval']['frame_action_head']['target_set_accuracy']:.3f} "
        f"primary={summary['eval']['frame_action_head']['primary_target_accuracy']:.3f}"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def train_heads(
    rows: list[dict[str, Any]],
    *,
    include_observed_action_features: bool,
    use_history_features: bool = True,
    max_iter: int,
) -> dict[str, Any]:
    vectorizer = DictVectorizer(sparse=True)
    history_by_sample = build_history_features(rows) if use_history_features else {}
    features = [
        featurize_candidate(
            row,
            include_observed_action_features=include_observed_action_features,
            history_features=history_by_sample.get(str(row.get("sample_token"))),
        )
        for row in rows
    ]
    x = vectorizer.fit_transform(features)
    y_valid = np.array([bool(row.get("is_valid")) for row in rows], dtype=np.int32)
    y_plan = np.array(
        [bool(row.get("is_rule_consistent_planner_action")) for row in rows],
        dtype=np.int32,
    )
    frame_examples = frame_examples_from_candidate_rows(rows)
    x_frame = vectorizer.transform(
        [
            featurize_frame(
                example["representative"],
                include_observed_action_features=include_observed_action_features,
                history_features=history_by_sample.get(
                    str(example["representative"].get("sample_token"))
                ),
            )
            for example in frame_examples
        ]
    )
    y_action = np.array([example["primary_target_action"] for example in frame_examples])
    validity_head = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=0,
        ),
    )
    planner_head = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=0,
        ),
    )
    action_head = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=0,
        ),
    )
    validity_head.fit(x, y_valid)
    planner_head.fit(x, y_plan)
    action_head.fit(x_frame, y_action)
    return {
        "vectorizer": vectorizer,
        "validity_head": validity_head,
        "planner_head": planner_head,
        "action_head": action_head,
        "planner_actions": list(PLANNER_ACTIONS),
        "include_observed_action_features": include_observed_action_features,
        "use_history_features": use_history_features,
        "feature_policy": {
            "excluded_current_state_fields": sorted(LEAKY_CURRENT_STATE_FIELDS),
            "categorical_fields": sorted(CATEGORICAL_FIELDS),
            "boolean_fields": sorted(BOOLEAN_FIELDS),
            "numeric_fields": sorted(NUMERIC_FIELDS),
            "history_categorical_fields": list(HISTORY_CATEGORICAL_FIELDS),
            "history_boolean_fields": list(HISTORY_BOOLEAN_FIELDS),
            "history_numeric_fields": list(HISTORY_NUMERIC_FIELDS),
        },
    }


def featurize_candidate(
    row: dict[str, Any],
    *,
    include_observed_action_features: bool = False,
    history_features: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    state = row.get("current_state") or {}
    feat: dict[str, Any] = {
        "candidate_action": row.get("candidate_action"),
        "direction": row.get("direction") or "unknown",
    }
    for action in PLANNER_ACTIONS:
        feat[f"candidate_is_{action}"] = row.get("candidate_action") == action
    for field in CATEGORICAL_FIELDS:
        feat[field] = state.get(field) or "missing"
    for field in BOOLEAN_FIELDS:
        feat[field] = bool(state.get(field))
    for field in NUMERIC_FIELDS:
        feat[field] = float_or_zero(state.get(field))
        feat[f"{field}_missing"] = state.get(field) is None

    water_level = as_float(state.get("water_level"))
    upstream_level = as_float(state.get("upstream_water_level"))
    downstream_level = as_float(state.get("downstream_water_level"))
    feat["upper_water_abs_diff"] = abs_diff_or_zero(water_level, upstream_level)
    feat["lower_water_abs_diff"] = abs_diff_or_zero(water_level, downstream_level)
    feat["upper_water_diff_missing"] = water_level is None or upstream_level is None
    feat["lower_water_diff_missing"] = water_level is None or downstream_level is None

    occupied = state.get("occupied_berth_slots") or []
    available = state.get("available_berth_slots") or []
    feat["occupied_berth_count_list"] = len(occupied) if isinstance(occupied, list) else 0
    feat["available_berth_count_list"] = len(available) if isinstance(available, list) else 0
    for slot in list_value(occupied):
        feat[f"occupied_{slot}"] = True
    for slot in list_value(available):
        feat[f"available_{slot}"] = True

    add_next_ship_features(feat, "next_enter", state.get("next_ship_to_enter_weak"))
    add_next_ship_features(feat, "next_leave", state.get("next_ship_to_leave_weak"))
    queue = state.get("queue_rank") or []
    feat["queue_rank_count"] = len(queue) if isinstance(queue, list) else 0

    if include_observed_action_features:
        add_optional_observed_features(feat, state, row)
    add_history_feature_values(feat, history_features)
    return feat


def featurize_frame(
    row: dict[str, Any],
    *,
    include_observed_action_features: bool = False,
    history_features: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    feat = featurize_candidate(
        row,
        include_observed_action_features=include_observed_action_features,
        history_features=history_features,
    )
    for key in list(feat):
        if key == "candidate_action" or key.startswith("candidate_is_"):
            del feat[key]
    return feat


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


def add_optional_observed_features(
    feat: dict[str, Any], state: dict[str, Any], row: dict[str, Any]
) -> None:
    feat["observed_action"] = state.get("observed_action") or "missing"
    feat["action_target"] = state.get("action_target") or "missing"
    feat["action_source"] = state.get("action_source") or "missing"
    feat["action_confidence"] = float_or_zero(state.get("action_confidence"))
    feat["ship_dispatch_action"] = state.get("ship_dispatch_action") or "missing"
    feat["ship_dispatch_target_count"] = float_or_zero(
        state.get("ship_dispatch_target_count")
    )
    feat["ship_dispatch_confidence"] = float_or_zero(
        state.get("ship_dispatch_confidence")
    )
    action = row.get("candidate_action")
    feat["candidate_matches_observed_action"] = action == state.get("observed_action")
    feat["candidate_matches_ship_dispatch_action"] = action == state.get(
        "ship_dispatch_action"
    )


def add_history_feature_values(
    feat: dict[str, Any], history_features: Optional[dict[str, Any]]
) -> None:
    history_features = history_features or {}
    for field in HISTORY_CATEGORICAL_FIELDS:
        feat[f"hist_{field}"] = history_features.get(field) or "missing"
    for field in HISTORY_BOOLEAN_FIELDS:
        feat[f"hist_{field}"] = bool(history_features.get(field))
    for field in HISTORY_NUMERIC_FIELDS:
        feat[f"hist_{field}"] = float_or_zero(history_features.get(field))
        feat[f"hist_{field}_missing"] = history_features.get(field) is None


def build_history_features(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    representatives: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_token = str(row.get("sample_token"))
        representatives.setdefault(sample_token, row)

    by_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in representatives.values():
        by_scene[(str(row.get("split")), str(row.get("scene_token")))].append(row)

    output: dict[str, dict[str, Any]] = {}
    run_track_fields = (
        "water_state",
        "operation_phase",
        "ship_operation_phase",
        "upper_gate_state",
        "lower_gate_state",
    )
    for _, scene_rows in by_scene.items():
        scene_rows.sort(
            key=lambda row: (
                int_or_zero(row.get("timestamp")),
                str(row.get("sample_token")),
            )
        )
        previous: Optional[dict[str, Any]] = None
        run_state: dict[str, dict[str, Any]] = {}
        for row in scene_rows:
            state = row.get("current_state") or {}
            timestamp = as_int(row.get("timestamp"))
            features = history_from_previous_frame(row, previous)
            for field in run_track_fields:
                value = state.get(field) or "missing"
                tracker = run_state.get(field)
                if tracker is None or tracker.get("value") != value:
                    tracker = {
                        "value": value,
                        "start_timestamp": timestamp,
                        "frame_count": 0,
                    }
                    run_state[field] = tracker
                tracker["frame_count"] = int(tracker.get("frame_count") or 0) + 1
                start_timestamp = tracker.get("start_timestamp")
                run_sec = 0.0
                if timestamp is not None and start_timestamp is not None:
                    run_sec = max(0.0, (timestamp - int(start_timestamp)) / 1_000_000.0)
                features[f"{field}_run_sec"] = run_sec
                features[f"{field}_run_frame_count"] = tracker["frame_count"]
            output[str(row.get("sample_token"))] = features
            previous = row
    return output


def history_from_previous_frame(
    row: dict[str, Any], previous: Optional[dict[str, Any]]
) -> dict[str, Any]:
    state = row.get("current_state") or {}
    timestamp = as_int(row.get("timestamp"))
    features: dict[str, Any] = {
        "has_prev_frame": previous is not None,
        "prev_upper_gate_state": None,
        "prev_lower_gate_state": None,
        "prev_water_state": None,
        "prev_operation_phase": None,
        "prev_ship_operation_phase": None,
        "dt_prev_sec": None,
        "water_level_delta_prev": None,
        "water_level_abs_delta_prev": None,
        "water_level_slope_prev": None,
        "upstream_water_level_delta_prev": None,
        "downstream_water_level_delta_prev": None,
        "upper_water_abs_diff_delta_prev": None,
        "lower_water_abs_diff_delta_prev": None,
        "water_state_changed_from_prev": False,
        "operation_phase_changed_from_prev": False,
        "upper_gate_state_changed_from_prev": False,
        "lower_gate_state_changed_from_prev": False,
    }
    if previous is None:
        return features

    prev_state = previous.get("current_state") or {}
    prev_timestamp = as_int(previous.get("timestamp"))
    dt_sec = None
    if timestamp is not None and prev_timestamp is not None:
        dt_sec = max(0.0, (timestamp - prev_timestamp) / 1_000_000.0)
    features["dt_prev_sec"] = dt_sec
    for field in (
        "upper_gate_state",
        "lower_gate_state",
        "water_state",
        "operation_phase",
        "ship_operation_phase",
    ):
        prev_value = prev_state.get(field)
        current_value = state.get(field)
        features[f"prev_{field}"] = prev_value
        if field in {"water_state", "operation_phase", "upper_gate_state", "lower_gate_state"}:
            features[f"{field}_changed_from_prev"] = (
                prev_value is not None and current_value is not None and prev_value != current_value
            )

    current_level = as_float(state.get("water_level"))
    prev_level = as_float(prev_state.get("water_level"))
    delta = diff_or_none(current_level, prev_level)
    features["water_level_delta_prev"] = delta
    features["water_level_abs_delta_prev"] = abs(delta) if delta is not None else None
    if delta is not None and dt_sec is not None and dt_sec > 0:
        features["water_level_slope_prev"] = delta / dt_sec
    features["upstream_water_level_delta_prev"] = diff_or_none(
        as_float(state.get("upstream_water_level")),
        as_float(prev_state.get("upstream_water_level")),
    )
    features["downstream_water_level_delta_prev"] = diff_or_none(
        as_float(state.get("downstream_water_level")),
        as_float(prev_state.get("downstream_water_level")),
    )
    features["upper_water_abs_diff_delta_prev"] = diff_or_none(
        level_abs_diff(state, "upstream_water_level"),
        level_abs_diff(prev_state, "upstream_water_level"),
    )
    features["lower_water_abs_diff_delta_prev"] = diff_or_none(
        level_abs_diff(state, "downstream_water_level"),
        level_abs_diff(prev_state, "downstream_water_level"),
    )
    return features


def evaluate_model(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    hard_mask: bool,
    include_observed_action_features: bool,
    dispatch_continuity_override: bool = True,
    history_source_rows: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    use_history_features = bool(model.get("use_history_features"))
    history_rows = history_source_rows if history_source_rows is not None else rows
    history_by_sample = build_history_features(history_rows) if use_history_features else {}
    x = model["vectorizer"].transform(
        [
            featurize_candidate(
                row,
                include_observed_action_features=include_observed_action_features,
                history_features=history_by_sample.get(str(row.get("sample_token"))),
            )
            for row in rows
        ]
    )
    validity_scores = positive_scores(model["validity_head"], x)
    planner_scores = positive_scores(model["planner_head"], x)
    y_valid = np.array([bool(row.get("is_valid")) for row in rows], dtype=np.int32)
    y_plan = np.array(
        [bool(row.get("is_rule_consistent_planner_action")) for row in rows],
        dtype=np.int32,
    )
    candidate_summary = {
        "candidate_validity": binary_summary(y_valid, validity_scores),
        "candidate_planner": binary_summary(y_plan, planner_scores),
    }
    frame_predictions = frame_level_predictions(
        rows,
        validity_scores=validity_scores,
        planner_scores=planner_scores,
        hard_mask=hard_mask,
    )
    action_predictions = frame_level_action_head_predictions(
        model,
        rows,
        hard_mask=hard_mask,
        include_observed_action_features=include_observed_action_features,
        history_by_sample=history_by_sample,
        dispatch_continuity_override=dispatch_continuity_override,
    )
    frame_summary = evaluate_frame_predictions(frame_predictions)
    action_summary = evaluate_frame_predictions(action_predictions)
    return {
        "summary": {
            **candidate_summary,
            "frame_candidate_ranker": frame_summary,
            "frame_action_head": action_summary,
        },
        "predictions": action_predictions,
    }


def positive_scores(model: Any, x: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    decision = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))


def binary_summary(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y_pred = (scores >= 0.5).astype(np.int32)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = None
    if len(set(y_true.tolist())) == 2:
        auc = float(roc_auc_score(y_true, scores))
    return {
        "num_examples": int(y_true.shape[0]),
        "positive_count": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "positive_precision": float(precision),
        "positive_recall": float(recall),
        "positive_f1": float(f1),
        "roc_auc": auc,
        "predicted_positive_count": int(y_pred.sum()),
    }


def frame_level_predictions(
    rows: list[dict[str, Any]],
    *,
    validity_scores: np.ndarray,
    planner_scores: np.ndarray,
    hard_mask: bool,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row.get("sample_token"))].append((index, row))

    predictions = []
    for sample_token, items in grouped.items():
        items.sort(key=lambda item: PLANNER_ACTIONS.index(item[1]["candidate_action"]))
        target_actions = [
            item[1]["candidate_action"]
            for item in items
            if item[1].get("is_rule_consistent_planner_action")
        ]
        primary_target = target_actions[0] if target_actions else "hold"
        candidate_details = []
        selectable: list[tuple[float, int, dict[str, Any]]] = []
        fallback: list[tuple[float, int, dict[str, Any]]] = []
        for action_order, (index, row) in enumerate(items):
            detail = {
                "action": row["candidate_action"],
                "is_valid": bool(row.get("is_valid")),
                "is_target": bool(row.get("is_rule_consistent_planner_action")),
                "validity_score": round(float(validity_scores[index]), 6),
                "planner_score": round(float(planner_scores[index]), 6),
                "violation_reason": row.get("violation_reason") or [],
            }
            candidate_details.append(detail)
            score_tuple = (float(planner_scores[index]), -action_order, row)
            fallback.append(score_tuple)
            if (not hard_mask) or row.get("is_valid"):
                selectable.append(score_tuple)
        choice_pool = selectable or fallback
        choice_pool.sort(key=lambda item: (item[0], item[1]), reverse=True)
        pred_row = choice_pool[0][2]
        pred_action = pred_row["candidate_action"]
        predictions.append(
            {
                "sample_token": sample_token,
                "split": pred_row.get("split"),
                "scene_token": pred_row.get("scene_token"),
                "timestamp": pred_row.get("timestamp"),
                "timestamp_str": pred_row.get("timestamp_str"),
                "predicted_action": pred_action,
                "target_actions": target_actions,
                "primary_target_action": primary_target,
                "is_legal": bool(pred_row.get("is_valid")),
                "target_set_hit": pred_action in set(target_actions),
                "primary_target_hit": pred_action == primary_target,
                "hard_mask": hard_mask,
                "candidate_scores": candidate_details,
            }
        )
    return sorted(
        predictions,
        key=lambda row: (
            str(row.get("split")),
            int(row.get("timestamp") or 0),
            str(row.get("sample_token")),
        ),
    )


def frame_examples_from_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_token"))].append(row)
    examples = []
    for sample_token, items in grouped.items():
        items.sort(key=lambda row: PLANNER_ACTIONS.index(row["candidate_action"]))
        target_actions = [
            row["candidate_action"]
            for row in items
            if row.get("is_rule_consistent_planner_action")
        ]
        examples.append(
            {
                "sample_token": sample_token,
                "representative": items[0],
                "target_actions": target_actions or ["hold"],
                "primary_target_action": (target_actions or ["hold"])[0],
                "candidates": items,
            }
        )
    return sorted(
        examples,
        key=lambda item: (
            str(item["representative"].get("split")),
            int(item["representative"].get("timestamp") or 0),
            str(item.get("sample_token")),
        ),
    )


def frame_level_action_head_predictions(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    hard_mask: bool,
    include_observed_action_features: bool,
    history_by_sample: Optional[dict[str, dict[str, Any]]] = None,
    dispatch_continuity_override: bool = True,
) -> list[dict[str, Any]]:
    if history_by_sample is None and model.get("use_history_features"):
        history_by_sample = build_history_features(rows)
    history_by_sample = history_by_sample or {}
    examples = frame_examples_from_candidate_rows(rows)
    x_frame = model["vectorizer"].transform(
        [
            featurize_frame(
                example["representative"],
                include_observed_action_features=include_observed_action_features,
                history_features=history_by_sample.get(
                    str(example["representative"].get("sample_token"))
                ),
            )
            for example in examples
        ]
    )
    probabilities = model["action_head"].predict_proba(x_frame)
    classes = [str(item) for item in model_classes(model["action_head"])]
    predictions = []
    for example, probs in zip(examples, probabilities):
        score_by_action = {action: 0.0 for action in PLANNER_ACTIONS}
        for action, score in zip(classes, probs):
            score_by_action[action] = float(score)
        valid = {
            row["candidate_action"]: bool(row.get("is_valid"))
            for row in example["candidates"]
        }
        target_actions = example["target_actions"]
        selectable = [
            action
            for action in PLANNER_ACTIONS
            if (not hard_mask) or valid.get(action, False)
        ]
        if not selectable:
            selectable = list(PLANNER_ACTIONS)
        pred_action = max(
            selectable,
            key=lambda action: (
                score_by_action.get(action, 0.0),
                -PLANNER_ACTIONS.index(action),
            ),
        )
        raw_pred_action = pred_action
        postprocess_rule = None
        if dispatch_continuity_override:
            pred_action, postprocess_rule = apply_dispatch_continuity_override(
                example, pred_action
            )
        pred_candidate = next(
            row for row in example["candidates"] if row["candidate_action"] == pred_action
        )
        candidate_scores = []
        for row in example["candidates"]:
            action = row["candidate_action"]
            candidate_scores.append(
                {
                    "action": action,
                    "is_valid": bool(row.get("is_valid")),
                    "is_target": bool(row.get("is_rule_consistent_planner_action")),
                    "action_head_score": round(score_by_action.get(action, 0.0), 6),
                    "violation_reason": row.get("violation_reason") or [],
                }
            )
        predictions.append(
            {
                "sample_token": example["sample_token"],
                "split": pred_candidate.get("split"),
                "scene_token": pred_candidate.get("scene_token"),
                "timestamp": pred_candidate.get("timestamp"),
                "timestamp_str": pred_candidate.get("timestamp_str"),
                "predicted_action": pred_action,
                "raw_predicted_action": raw_pred_action,
                "postprocess_rule": postprocess_rule,
                "target_actions": target_actions,
                "primary_target_action": example["primary_target_action"],
                "is_legal": bool(pred_candidate.get("is_valid")),
                "target_set_hit": pred_action in set(target_actions),
                "primary_target_hit": pred_action == example["primary_target_action"],
                "hard_mask": hard_mask,
                "candidate_scores": candidate_scores,
            }
        )
    return predictions


def apply_dispatch_continuity_override(
    example: dict[str, Any], pred_action: str
) -> tuple[str, Optional[str]]:
    if pred_action != "hold":
        return pred_action, None
    valid = {
        row["candidate_action"]: bool(row.get("is_valid"))
        for row in example["candidates"]
    }
    state = example["representative"].get("current_state") or {}
    next_ship = state.get("next_ship_to_leave_weak")
    exit_gate_open = state.get("upper_gate_state") == "open" or state.get(
        "lower_gate_state"
    ) == "open"
    if (
        valid.get("dispatch_exit")
        and state.get("ship_operation_phase") == "ship_leaving"
        and isinstance(next_ship, dict)
        and exit_gate_open
        and state.get("water_state") == "idle"
    ):
        return "dispatch_exit", "dispatch_exit_continuity"
    if valid.get("dispatch_enter"):
        entry_side = entry_side_for_direction(str(example["representative"].get("direction") or ""))
        entry_gate_open = (
            entry_side is not None
            and state.get(f"{entry_side}_gate_state") == "open"
        )
        next_ship = state.get("next_ship_to_enter_weak")
        if (
            state.get("ship_operation_phase") == "ship_entering"
            and isinstance(next_ship, dict)
            and entry_gate_open
            and state.get("water_state") == "idle"
        ):
            return "dispatch_enter", "dispatch_enter_continuity"
    return pred_action, None


def entry_side_for_direction(direction: str) -> Optional[str]:
    if direction == "upstream":
        return "lower"
    if direction == "downstream":
        return "upper"
    return None


def model_classes(model: Any) -> np.ndarray:
    if hasattr(model, "classes_"):
        return model.classes_
    if hasattr(model, "named_steps"):
        for step in reversed(model.steps):
            estimator = step[1]
            if hasattr(estimator, "classes_"):
                return estimator.classes_
    raise AttributeError("model does not expose classes_")


def evaluate_frame_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    legal = sum(1 for row in rows if row["is_legal"])
    target_hits = sum(1 for row in rows if row["target_set_hit"])
    primary_hits = sum(1 for row in rows if row["primary_target_hit"])
    prediction_counts = Counter(row["predicted_action"] for row in rows)
    primary_counts = Counter(row["primary_target_action"] for row in rows)
    confusion = Counter(
        f"{row['primary_target_action']}->{row['predicted_action']}" for row in rows
    )
    total = len(rows)
    return {
        "num_frames": total,
        "legal_count": legal,
        "legal_rate": safe_div(legal, total),
        "target_set_hit_count": target_hits,
        "target_set_accuracy": safe_div(target_hits, total),
        "primary_target_hit_count": primary_hits,
        "primary_target_accuracy": safe_div(primary_hits, total),
        "prediction_counts": dict(prediction_counts),
        "primary_target_counts": dict(primary_counts),
        "top_confusions": dict(confusion.most_common(40)),
    }


def label_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "valid": dict(Counter(bool(row.get("is_valid")) for row in rows)),
        "planner_target": dict(
            Counter(bool(row.get("is_rule_consistent_planner_action")) for row in rows)
        ),
        "candidate_action": dict(Counter(row.get("candidate_action") for row in rows)),
        "target_action": dict(
            Counter(
                row.get("candidate_action")
                for row in rows
                if row.get("is_rule_consistent_planner_action")
            )
        ),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def list_value(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def float_or_zero(value: Any) -> float:
    out = as_float(value)
    return 0.0 if out is None else out


def abs_diff_or_zero(left: Optional[float], right: Optional[float]) -> float:
    if left is None or right is None:
        return 0.0
    return abs(left - right)


def diff_or_none(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return left - right


def level_abs_diff(state: dict[str, Any], outside_key: str) -> Optional[float]:
    water_level = as_float(state.get("water_level"))
    outside_level = as_float(state.get(outside_key))
    if water_level is None or outside_level is None:
        return None
    return abs(water_level - outside_level)


def as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def int_or_zero(value: Any) -> int:
    out = as_int(value)
    return 0 if out is None else out


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


if __name__ == "__main__":
    main()
