#!/usr/bin/env python3
"""Evaluate simple rule/persistence baselines on action-conditioned frames."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


NON_HOLD_PRIORITY = (
    "stop_filling_emptying",
    "dispatch_exit",
    "dispatch_enter",
    "start_filling",
    "start_emptying",
    "open_upper_gate",
    "open_lower_gate",
    "close_upper_gate",
    "close_lower_gate",
)
HORIZONS = (10, 20, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/action_conditioned_valtest_frames.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/summary_valtest_rule_baselines.json"
        ),
    )
    parser.add_argument(
        "--deployable-world-state",
        type=Path,
        default=None,
        help=(
            "Optional full-frame deployable world-state JSONL. When provided, "
            "future_persistence_baseline is evaluated from deployable current "
            "state instead of GT structured current_state; the GT structured "
            "result is kept as a diagnostic field."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    deployable_world_state_by_sample = None
    if args.deployable_world_state is not None:
        deployable_world_state_by_sample = load_deployable_world_state_by_sample(
            args.deployable_world_state
        )
    summary = evaluate(
        rows,
        input_path=args.input,
        deployable_world_state_path=args.deployable_world_state,
        deployable_world_state_by_sample=deployable_world_state_by_sample,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"input={args.input}")
    print(f"wrote={args.summary_output}")
    print(f"num_frames={summary['num_frames']}")
    for name, metrics in summary["planner_baselines"].items():
        print(
            f"{name}: legal={metrics['legal_rate']:.3f} "
            f"target_set_acc={metrics['target_set_accuracy']:.3f} "
            f"primary_acc={metrics['primary_target_accuracy']:.3f}"
        )
    print(f"future_persistence_source={summary['future_persistence_baseline_source']}")
    for horizon, metrics in summary["future_persistence_baseline"].items():
        print(
            f"{horizon}: state_exact={metrics['state_exact_accuracy']:.3f} "
            f"phase_acc={metrics['phase_accuracy']:.3f} "
            f"water_mae={metrics['water_level_mae']}"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate(
    rows: list[dict[str, Any]],
    *,
    input_path: Path,
    deployable_world_state_path: Path | None = None,
    deployable_world_state_by_sample: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    baselines: dict[str, Callable[[dict[str, Any]], str]] = {
        "hold": predict_hold,
        "first_valid_non_hold_priority": predict_first_valid_non_hold_priority,
        "observed_valid_or_hold_oracle": predict_observed_valid_or_hold_oracle,
    }
    raw_observed_primary_counts = Counter(
        row["conditioning"]["primary_observed_planner_action"] for row in rows
    )
    target_primary_counts = Counter(primary_target_action(row) for row in rows)
    split_counts = Counter(row.get("split") for row in rows)
    gt_structured_future = evaluate_future_persistence(rows)
    future_persistence_source = "gt_structured_current_state"
    future_persistence = gt_structured_future
    deployable_replacement_report = None
    if deployable_world_state_by_sample is not None:
        deployable_rows, deployable_replacement_report = (
            replace_future_persistence_current_state_with_deployable(
                rows, deployable_world_state_by_sample=deployable_world_state_by_sample
            )
        )
        future_persistence_source = "deployable_world_state_current_state"
        future_persistence = evaluate_future_persistence(deployable_rows)
    return {
        "input": str(input_path),
        "deployable_world_state": (
            str(deployable_world_state_path)
            if deployable_world_state_path is not None
            else None
        ),
        "num_frames": len(rows),
        "split_counts": dict(split_counts),
        "target_action_source": "rule_consistent_planner_actions",
        "raw_observed_primary_planner_action_counts": dict(raw_observed_primary_counts),
        "target_primary_planner_action_counts": dict(target_primary_counts),
        "raw_observed_target_validity": raw_observed_target_validity_metrics(rows),
        "target_validity": target_validity_metrics(rows),
        "planner_baselines": {
            name: evaluate_planner_baseline(rows, predictor)
            for name, predictor in baselines.items()
        },
        "future_persistence_baseline_source": future_persistence_source,
        "future_persistence_baseline": future_persistence,
        "future_deployable_replacement_report": deployable_replacement_report,
        "future_gt_structured_persistence_diagnostic": gt_structured_future,
        "notes": [
            "hold is a conservative rule baseline because hold is always valid in the current mask",
            "first_valid_non_hold_priority is legal by construction but intentionally non-oracle",
            "observed_valid_or_hold_oracle measures the rule mask upper bound for observed planner actions",
            "future_persistence_baseline predicts current gate/water/phase unchanged at each horizon",
            "when deployable_world_state is provided, future_persistence_baseline is the deployable world-model short-horizon result; GT structured persistence is diagnostic only",
        ],
    }


def load_deployable_world_state_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    by_sample: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_token = row.get("sample_token")
        if sample_token is not None:
            by_sample[str(sample_token)] = row
    return by_sample


def replace_future_persistence_current_state_with_deployable(
    rows: list[dict[str, Any]],
    *,
    deployable_world_state_by_sample: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from tools.evaluate_planner_on_full_deployable_world_state import (
        deployable_current_state,
    )

    output = []
    report = {
        "input_frames": len(rows),
        "deployable_world_state_frames": len(deployable_world_state_by_sample),
        "replaced_frames": 0,
        "missing_frames": 0,
        "deployable_operation_phase_counts": Counter(),
    }
    for row in rows:
        new_row = copy.deepcopy(row)
        state = deployable_world_state_by_sample.get(str(row.get("sample_token")))
        if state is None:
            report["missing_frames"] += 1
        else:
            new_row["current_state"] = deployable_current_state(row, state)
            report["replaced_frames"] += 1
            report["deployable_operation_phase_counts"][
                new_row["current_state"].get("operation_phase")
            ] += 1
        output.append(new_row)
    report["deployable_operation_phase_counts"] = dict(
        report["deployable_operation_phase_counts"]
    )
    return output, report


def valid_actions(row: dict[str, Any]) -> set[str]:
    return set((row.get("current_state") or {}).get("valid_actions") or [])


def target_actions(row: dict[str, Any]) -> list[str]:
    actions = (
        row.get("conditioning", {}).get("rule_consistent_planner_actions")
        or row.get("conditioning", {}).get("observed_planner_actions")
        or ["hold"]
    )
    return [str(action) for action in actions]


def raw_observed_actions(row: dict[str, Any]) -> list[str]:
    actions = row.get("conditioning", {}).get("observed_planner_actions") or ["hold"]
    return [str(action) for action in actions]


def primary_target_action(row: dict[str, Any]) -> str:
    return str(
        row.get("conditioning", {}).get("primary_rule_consistent_planner_action")
        or row.get("conditioning", {}).get("primary_observed_planner_action")
        or "hold"
    )


def predict_hold(row: dict[str, Any]) -> str:
    return "hold"


def predict_first_valid_non_hold_priority(row: dict[str, Any]) -> str:
    valid = valid_actions(row)
    for action in NON_HOLD_PRIORITY:
        if action in valid:
            return action
    return "hold" if "hold" in valid else sorted(valid)[0] if valid else "hold"


def predict_observed_valid_or_hold_oracle(row: dict[str, Any]) -> str:
    valid = valid_actions(row)
    for action in target_actions(row):
        if action in valid:
            return action
    return "hold"


def evaluate_planner_baseline(
    rows: list[dict[str, Any]],
    predictor: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    legal = 0
    target_set_hits = 0
    primary_hits = 0
    prediction_counts: Counter[str] = Counter()
    illegal_counts: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    for row in rows:
        pred = predictor(row)
        target = primary_target_action(row)
        prediction_counts[pred] += 1
        confusion[f"{target}->{pred}"] += 1
        if pred in valid_actions(row):
            legal += 1
        else:
            illegal_counts[pred] += 1
        if pred in target_actions(row):
            target_set_hits += 1
        if pred == target:
            primary_hits += 1
    total = len(rows)
    return {
        "legal_count": legal,
        "legal_rate": safe_div(legal, total),
        "target_set_hit_count": target_set_hits,
        "target_set_accuracy": safe_div(target_set_hits, total),
        "primary_target_hit_count": primary_hits,
        "primary_target_accuracy": safe_div(primary_hits, total),
        "prediction_counts": dict(prediction_counts),
        "illegal_prediction_counts": dict(illegal_counts),
        "top_confusions": dict(confusion.most_common(30)),
    }


def target_validity_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_valid = 0
    all_observed_valid = 0
    any_observed_valid = 0
    invalid_primary_counts: Counter[str] = Counter()
    for row in rows:
        valid = valid_actions(row)
        primary = primary_target_action(row)
        observed = target_actions(row)
        if primary in valid:
            primary_valid += 1
        else:
            invalid_primary_counts[primary] += 1
        if all(action in valid for action in observed):
            all_observed_valid += 1
        if any(action in valid for action in observed):
            any_observed_valid += 1
    total = len(rows)
    return {
        "primary_valid_count": primary_valid,
        "primary_valid_rate": safe_div(primary_valid, total),
        "all_observed_actions_valid_count": all_observed_valid,
        "all_observed_actions_valid_rate": safe_div(all_observed_valid, total),
        "any_observed_action_valid_count": any_observed_valid,
        "any_observed_action_valid_rate": safe_div(any_observed_valid, total),
        "invalid_primary_action_counts": dict(invalid_primary_counts),
    }


def raw_observed_target_validity_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_valid = 0
    all_observed_valid = 0
    any_observed_valid = 0
    invalid_primary_counts: Counter[str] = Counter()
    invalid_observed_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    for row in rows:
        valid = valid_actions(row)
        current = row.get("current_state") or {}
        reasons = current.get("violation_reason") or {}
        primary = str(
            row.get("conditioning", {}).get("primary_observed_planner_action") or "hold"
        )
        observed = raw_observed_actions(row)
        if primary in valid:
            primary_valid += 1
        else:
            invalid_primary_counts[primary] += 1
        if all(action in valid for action in observed):
            all_observed_valid += 1
        if any(action in valid for action in observed):
            any_observed_valid += 1
        for action in observed:
            if action not in valid:
                invalid_observed_counts[action] += 1
                for reason in reasons.get(action, []):
                    invalid_reasons[reason] += 1
    total = len(rows)
    return {
        "primary_valid_count": primary_valid,
        "primary_valid_rate": safe_div(primary_valid, total),
        "all_observed_actions_valid_count": all_observed_valid,
        "all_observed_actions_valid_rate": safe_div(all_observed_valid, total),
        "any_observed_action_valid_count": any_observed_valid,
        "any_observed_action_valid_rate": safe_div(any_observed_valid, total),
        "invalid_primary_action_counts": dict(invalid_primary_counts),
        "invalid_observed_action_counts": dict(invalid_observed_counts),
        "top_invalid_observed_reasons": dict(invalid_reasons.most_common(30)),
    }


def evaluate_future_persistence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for horizon in HORIZONS:
        key = f"t_plus_{horizon}s"
        totals = Counter()
        hits = Counter()
        water_abs_error = 0.0
        water_count = 0
        for row in rows:
            target = (
                row.get("future_targets", {})
                .get("horizons", {})
                .get(key, {})
                .get("state")
            )
            phase = (
                row.get("future_targets", {})
                .get("horizons", {})
                .get(key, {})
                .get("phase")
            )
            if not isinstance(target, dict):
                continue
            current = row.get("current_state") or {}
            totals["state"] += 1
            state_fields = ("upper_gate_state", "lower_gate_state", "water_state")
            if all(current.get(field) == target.get(field) for field in state_fields):
                hits["state_exact"] += 1
            for field in state_fields:
                totals[field] += 1
                if current.get(field) == target.get(field):
                    hits[field] += 1
            totals["phase"] += 1
            if current.get("operation_phase") == phase:
                hits["phase"] += 1
            current_water = as_float(current.get("water_level"))
            target_water = as_float(target.get("water_level"))
            if current_water is not None and target_water is not None:
                water_abs_error += abs(current_water - target_water)
                water_count += 1
        out[key] = {
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
        }
    return out


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


if __name__ == "__main__":
    main()
