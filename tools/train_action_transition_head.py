#!/usr/bin/env python3
"""Train an action-conditioned lock-state transition head.

The current dataset only has observed-trajectory future targets. This model
therefore learns the transition for the action that actually happened in each
frame, not counterfactual futures for every valid candidate action.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools.train_action_planner_head import (
    build_history_features,
    featurize_candidate,
    read_jsonl,
    write_jsonl,
)


HORIZONS = (10, 20, 30)
STATE_FIELDS = ("upper_gate_state", "lower_gate_state", "water_state")
PHASE_FIELD = "operation_phase"
CATEGORICAL_TARGET_FIELDS = STATE_FIELDS + (PHASE_FIELD,)


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
        "--deployable-eval-candidates",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/action_conditioned_valtest_full_deployable_candidates.jsonl"
        ),
        help=(
            "Optional eval candidates with deployable current_state replacement. "
            "If present, a deployable-current input mode is evaluated."
        ),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/action_transition_head.pkl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/summary_valtest_action_transition_head.json"
        ),
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/predictions_valtest_action_transition_head.jsonl"
        ),
        help="Diagnostic pure learned-transition predictions.",
    )
    parser.add_argument(
        "--hybrid-prediction-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/predictions_valtest_action_transition_hybrid.jsonl"
        ),
        help=(
            "Selected-policy predictions: hold uses persistence, non-hold uses "
            "the learned transition head."
        ),
    )
    parser.add_argument("--no-history-features", action="store_true")
    parser.add_argument("--max-iter", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_candidates)
    eval_rows = read_jsonl(args.eval_candidates)
    model = train_transition_model(
        train_rows,
        use_history_features=not args.no_history_features,
        max_iter=args.max_iter,
    )
    eval_modes = {
        "gt_structured_input": evaluate_transition_model(
            model,
            eval_rows,
            history_source_rows=eval_rows,
        )
    }
    prediction_rows = [
        {
            **row,
            "evaluation_mode": "gt_structured_input",
            "transition_policy": "learned_transition_head",
        }
        for row in eval_modes["gt_structured_input"]["predictions"]
    ]
    hybrid_prediction_rows = [
        {
            **row,
            "evaluation_mode": "gt_structured_input",
            "transition_policy": "hold_persistence_hybrid",
        }
        for row in eval_modes["gt_structured_input"]["hybrid_predictions"]
    ]

    deployable_path = args.deployable_eval_candidates
    if deployable_path is not None and deployable_path.exists():
        deployable_rows = read_jsonl(deployable_path)
        deployable_eval = evaluate_transition_model(
            model,
            deployable_rows,
            history_source_rows=deployable_rows,
        )
        eval_modes["deployable_current_input"] = deployable_eval
        prediction_rows.extend(
            {
                **row,
                "evaluation_mode": "deployable_current_input",
                "transition_policy": "learned_transition_head",
            }
            for row in deployable_eval["predictions"]
        )
        hybrid_prediction_rows.extend(
            {
                **row,
                "evaluation_mode": "deployable_current_input",
                "transition_policy": "hold_persistence_hybrid",
            }
            for row in deployable_eval["hybrid_predictions"]
        )

    summary = {
        "train_candidates": str(args.train_candidates),
        "eval_candidates": str(args.eval_candidates),
        "deployable_eval_candidates": (
            str(deployable_path) if deployable_path is not None and deployable_path.exists() else None
        ),
        "output_model": str(args.output_model),
        "prediction_output": str(args.prediction_output),
        "hybrid_prediction_output": str(args.hybrid_prediction_output),
        "selected_transition_policy": "hold_persistence_hybrid",
        "use_history_features": not args.no_history_features,
        "num_train_candidates": len(train_rows),
        "num_train_supervised_rows": len(future_supervised_rows(train_rows)),
        "num_eval_candidates": len(eval_rows),
        "num_features": len(model["vectorizer"].feature_names_),
        "supervised_action_counts": {
            "train": dict(action_counts(future_supervised_rows(train_rows))),
            "eval": dict(action_counts(future_supervised_rows(eval_rows))),
        },
        "evaluation_modes": {
            name: result["summary"] for name, result in eval_modes.items()
        },
        "notes": [
            "This transition head is action-conditioned, but supervised only on the observed gate/water action branch.",
            "Candidate actions without observed future labels are excluded from transition training/evaluation.",
            "Targets cover upper_gate_state, lower_gate_state, water_state, operation_phase, and water_level at 10/20/30s.",
            "The deployable_current_input mode swaps only the current_state input; future targets remain the observed val+test labels.",
            "This is a first gate/water/operation_phase transition model; dense future ship occupancy/motion rollout is still separate work.",
            "The selected reportable policy is hold_persistence_hybrid: hold uses persistence, non-hold uses the learned transition head.",
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
    write_jsonl(args.prediction_output, prediction_rows)
    write_jsonl(args.hybrid_prediction_output, hybrid_prediction_rows)

    print(f"wrote_model={args.output_model}")
    print(f"wrote_summary={args.summary_output}")
    print(f"wrote_predictions={args.prediction_output}")
    print(f"wrote_hybrid_predictions={args.hybrid_prediction_output}")
    print(f"num_train_supervised_rows={summary['num_train_supervised_rows']}")
    for mode, result in summary["evaluation_modes"].items():
        print(mode)
        selected = result[summary["selected_transition_policy"]]
        for horizon, metrics in selected.items():
            print(
                f"  {horizon}: state_exact={metrics['state_exact_accuracy']:.3f} "
                f"phase={metrics['phase_accuracy']:.3f} "
                f"water_mae={metrics['water_level_mae']}"
            )


def future_supervised_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("future_gate_water_target_available")
        and isinstance(row.get("future_targets"), dict)
    ]


def action_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("candidate_action") or "missing") for row in rows)


def train_transition_model(
    rows: list[dict[str, Any]],
    *,
    use_history_features: bool = True,
    max_iter: int = 3000,
) -> dict[str, Any]:
    supervised = future_supervised_rows(rows)
    if not supervised:
        raise SystemExit("no rows with future_gate_water_target_available")
    history_by_sample = build_history_features(rows) if use_history_features else {}
    vectorizer = DictVectorizer(sparse=True)
    train_features = [
        transition_features(
            row,
            history_features=history_by_sample.get(str(row.get("sample_token"))),
        )
        for row in supervised
    ]
    vectorizer.fit(train_features)
    heads: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        horizon_key = horizon_name(horizon)
        horizon_rows = rows_with_horizon_target(supervised, horizon_key)
        if not horizon_rows:
            heads[horizon_key] = empty_horizon_head()
            continue
        x = vectorizer.transform(
            [
                transition_features(
                    row,
                    history_features=history_by_sample.get(str(row.get("sample_token"))),
                )
                for row in horizon_rows
            ]
        )
        heads[horizon_key] = {
            "num_train_rows": len(horizon_rows),
            "categorical": {
                field: fit_classifier(
                    x,
                    [target_value(row, horizon_key, field) for row in horizon_rows],
                    max_iter=max_iter,
                )
                for field in CATEGORICAL_TARGET_FIELDS
            },
            "water_level_delta": fit_water_delta_regressor(x, horizon_rows, horizon_key),
        }
    return {
        "vectorizer": vectorizer,
        "heads": heads,
        "horizons": list(HORIZONS),
        "use_history_features": use_history_features,
        "target_fields": {
            "categorical": list(CATEGORICAL_TARGET_FIELDS),
            "water_level_delta": "target_water_level - current_water_level",
        },
    }


def transition_features(
    row: dict[str, Any],
    *,
    history_features: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return featurize_candidate(
        row,
        include_observed_action_features=False,
        history_features=history_features,
    )


def rows_with_horizon_target(
    rows: list[dict[str, Any]], horizon_key: str
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        state = horizon_target_state(row, horizon_key)
        if isinstance(state, dict):
            out.append(row)
    return out


def fit_classifier(x: Any, y_values: list[Any], *, max_iter: int) -> dict[str, Any]:
    labels = [str(value) if value is not None else "missing" for value in y_values]
    counts = Counter(labels)
    if len(counts) <= 1:
        return {"kind": "constant", "value": labels[0] if labels else "missing"}
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
        "class_counts": dict(counts),
    }


def fit_water_delta_regressor(
    x: Any, rows: list[dict[str, Any]], horizon_key: str
) -> dict[str, Any]:
    y = []
    for row in rows:
        current = as_float((row.get("current_state") or {}).get("water_level"))
        target = as_float((horizon_target_state(row, horizon_key) or {}).get("water_level"))
        if current is None or target is None:
            y.append(0.0)
        else:
            y.append(target - current)
    if not y:
        return {"kind": "constant_delta", "value": 0.0}
    if max(y) == min(y):
        return {"kind": "constant_delta", "value": float(y[0])}
    head = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0))
    head.fit(x, np.asarray(y, dtype=np.float64))
    return {"kind": "regressor", "head": head}


def evaluate_transition_model(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    history_source_rows: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    supervised = future_supervised_rows(rows)
    history_rows = history_source_rows if history_source_rows is not None else rows
    history_by_sample = (
        build_history_features(history_rows) if model.get("use_history_features") else {}
    )
    predictions = []
    for row in supervised:
        features = transition_features(
            row,
            history_features=history_by_sample.get(str(row.get("sample_token"))),
        )
        x = model["vectorizer"].transform([features])
        pred_horizons = {}
        for horizon in HORIZONS:
            horizon_key = horizon_name(horizon)
            if horizon_target_state(row, horizon_key) is None:
                continue
            pred_horizons[horizon_key] = predict_horizon(model, x, row, horizon_key)
        if pred_horizons:
            predictions.append(
                {
                    "sample_token": row.get("sample_token"),
                    "split": row.get("split"),
                    "scene_token": row.get("scene_token"),
                    "timestamp": row.get("timestamp"),
                    "timestamp_str": row.get("timestamp_str"),
                    "candidate_action": row.get("candidate_action"),
                    "predictions": pred_horizons,
                    "targets": {
                        key: row.get("future_targets", {}).get("horizons", {}).get(key)
                        for key in pred_horizons
                    },
                }
            )
    return {
        "summary": {
            "num_supervised_rows": len(supervised),
            "action_counts": dict(action_counts(supervised)),
            "transition_head": transition_metrics(supervised, predictions),
            "hold_persistence_hybrid": transition_metrics(
                supervised,
                hold_persistence_hybrid_predictions(
                    model,
                    supervised,
                    history_by_sample,
                    include_details=False,
                ),
            ),
            "persistence_baseline": persistence_metrics(supervised),
        },
        "predictions": predictions,
        "hybrid_predictions": hold_persistence_hybrid_predictions(
            model,
            supervised,
            history_by_sample,
            include_details=True,
        ),
    }


def empty_horizon_head() -> dict[str, Any]:
    return {
        "num_train_rows": 0,
        "categorical": {
            field: {"kind": "constant", "value": "missing"}
            for field in CATEGORICAL_TARGET_FIELDS
        },
        "water_level_delta": {"kind": "constant_delta", "value": 0.0},
    }


def predict_horizon(
    model: dict[str, Any], x: Any, row: dict[str, Any], horizon_key: str
) -> dict[str, Any]:
    heads = model["heads"][horizon_key]
    state = {}
    for field in STATE_FIELDS:
        state[field] = predict_categorical(heads["categorical"][field], x)
    current_water = as_float((row.get("current_state") or {}).get("water_level"))
    delta = predict_water_delta(heads["water_level_delta"], x)
    state["water_level"] = (
        round(current_water + delta, 4) if current_water is not None else round(delta, 4)
    )
    phase = predict_categorical(heads["categorical"][PHASE_FIELD], x)
    state["operation_phase"] = phase
    return {"state": state, "phase": phase}


def predict_categorical(head: dict[str, Any], x: Any) -> str:
    if head.get("kind") == "constant":
        return str(head.get("value"))
    return str(head["head"].predict(x)[0])


def predict_water_delta(head: dict[str, Any], x: Any) -> float:
    if head.get("kind") == "constant_delta":
        return float(head.get("value") or 0.0)
    return float(head["head"].predict(x)[0])


def transition_metrics(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    by_sample = {str(row.get("sample_token")): row for row in rows}
    out: dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_key = horizon_name(horizon)
        totals = Counter()
        hits = Counter()
        water_abs_error = 0.0
        water_count = 0
        confusion: Counter[str] = Counter()
        for item in predictions:
            if horizon_key not in item["predictions"]:
                continue
            row = by_sample.get(str(item.get("sample_token")))
            if not row:
                continue
            target_state = horizon_target_state(row, horizon_key)
            target_phase = horizon_target_phase(row, horizon_key)
            if not isinstance(target_state, dict):
                continue
            pred_state = item["predictions"][horizon_key]["state"]
            totals["state"] += 1
            if all(pred_state.get(field) == target_state.get(field) for field in STATE_FIELDS):
                hits["state_exact"] += 1
            for field in STATE_FIELDS:
                totals[field] += 1
                if pred_state.get(field) == target_state.get(field):
                    hits[field] += 1
            totals["phase"] += 1
            pred_phase = item["predictions"][horizon_key]["phase"]
            confusion[f"{target_phase}->{pred_phase}"] += 1
            if pred_phase == target_phase:
                hits["phase"] += 1
            pred_water = as_float(pred_state.get("water_level"))
            target_water = as_float(target_state.get("water_level"))
            if pred_water is not None and target_water is not None:
                water_abs_error += abs(pred_water - target_water)
                water_count += 1
        out[horizon_key] = {
            "num_targets": totals["state"],
            "state_exact_count": hits["state_exact"],
            "state_exact_accuracy": safe_div(hits["state_exact"], totals["state"]),
            "upper_gate_accuracy": safe_div(hits["upper_gate_state"], totals["upper_gate_state"]),
            "lower_gate_accuracy": safe_div(hits["lower_gate_state"], totals["lower_gate_state"]),
            "water_state_accuracy": safe_div(hits["water_state"], totals["water_state"]),
            "phase_count": hits["phase"],
            "phase_accuracy": safe_div(hits["phase"], totals["phase"]),
            "water_level_mae": (
                round(water_abs_error / water_count, 4) if water_count else None
            ),
            "water_level_count": water_count,
            "top_phase_confusions": dict(confusion.most_common(20)),
        }
    return out


def persistence_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = []
    for row in rows:
        current = row.get("current_state") or {}
        pred_horizons = {}
        for horizon in HORIZONS:
            horizon_key = horizon_name(horizon)
            if horizon_target_state(row, horizon_key) is None:
                continue
            state = {field: current.get(field) for field in STATE_FIELDS}
            state["water_level"] = current.get("water_level")
            phase = current.get(PHASE_FIELD)
            state[PHASE_FIELD] = phase
            pred_horizons[horizon_key] = {"state": state, "phase": phase}
        if pred_horizons:
            predictions.append(
                {
                    "sample_token": row.get("sample_token"),
                    "predictions": pred_horizons,
                }
            )
    return transition_metrics(rows, predictions)


def hold_persistence_hybrid_predictions(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    history_by_sample: dict[str, dict[str, Any]],
    *,
    include_details: bool = False,
) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        if row.get("candidate_action") == "hold":
            current = row.get("current_state") or {}
            pred_horizons = {}
            for horizon in HORIZONS:
                horizon_key = horizon_name(horizon)
                if horizon_target_state(row, horizon_key) is None:
                    continue
                state = {field: current.get(field) for field in STATE_FIELDS}
                state["water_level"] = current.get("water_level")
                phase = current.get(PHASE_FIELD)
                state[PHASE_FIELD] = phase
                pred_horizons[horizon_key] = {"state": state, "phase": phase}
        else:
            features = transition_features(
                row,
                history_features=history_by_sample.get(str(row.get("sample_token"))),
            )
            x = model["vectorizer"].transform([features])
            pred_horizons = {}
            for horizon in HORIZONS:
                horizon_key = horizon_name(horizon)
                if horizon_target_state(row, horizon_key) is None:
                    continue
                pred_horizons[horizon_key] = predict_horizon(model, x, row, horizon_key)
        if pred_horizons:
            item = {
                "sample_token": row.get("sample_token"),
                "predictions": pred_horizons,
            }
            if include_details:
                item.update(
                    {
                        "split": row.get("split"),
                        "scene_token": row.get("scene_token"),
                        "timestamp": row.get("timestamp"),
                        "timestamp_str": row.get("timestamp_str"),
                        "candidate_action": row.get("candidate_action"),
                        "targets": {
                            key: row.get("future_targets", {})
                            .get("horizons", {})
                            .get(key)
                            for key in pred_horizons
                        },
                    }
                )
            predictions.append(item)
    return predictions


def horizon_name(horizon_sec: int) -> str:
    return f"t_plus_{horizon_sec}s"


def horizon_target_state(row: dict[str, Any], horizon_key: str) -> Optional[dict[str, Any]]:
    horizon = (
        row.get("future_targets", {})
        .get("horizons", {})
        .get(horizon_key)
    )
    if not isinstance(horizon, dict):
        return None
    return horizon.get("state")


def horizon_target_phase(row: dict[str, Any], horizon_key: str) -> Any:
    horizon = (
        row.get("future_targets", {})
        .get("horizons", {})
        .get(horizon_key)
    )
    if not isinstance(horizon, dict):
        return None
    return horizon.get("phase")


def target_value(row: dict[str, Any], horizon_key: str, field: str) -> Any:
    if field == PHASE_FIELD:
        phase = horizon_target_phase(row, horizon_key)
        if phase is not None:
            return phase
    return (horizon_target_state(row, horizon_key) or {}).get(field)


def as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


if __name__ == "__main__":
    main()
