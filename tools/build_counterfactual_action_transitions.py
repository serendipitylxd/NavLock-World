#!/usr/bin/env python3
"""Build rule-based counterfactual gate/water action-transition targets.

The dataset contains observed future labels, not counterfactual outcomes. This
tool therefore generates rule-simulator targets for valid candidate actions and
evaluates them only on the factual/observed-action branch.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Optional


HORIZONS = (10, 20, 30)
STATE_FIELDS = ("upper_gate_state", "lower_gate_state", "water_state")
WATER_ACTIONS = {"start_filling", "start_emptying", "stop_filling_emptying"}
GATE_ACTIONS = {
    "open_upper_gate",
    "close_upper_gate",
    "open_lower_gate",
    "close_lower_gate",
}
DISPATCH_ACTIONS = {"dispatch_enter", "dispatch_exit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-candidates",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/action_conditioned_train_candidates.jsonl"
        ),
        help="Used only to estimate water-level transition rates.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/action_conditioned_valtest_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/counterfactual_action_transitions_valtest.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/summary_valtest_counterfactual_action_transitions.json"
        ),
    )
    parser.add_argument("--gate-transition-duration-sec", type=float, default=45.0)
    parser.add_argument("--default-filling-rate-mps", type=float, default=0.010)
    parser.add_argument("--default-emptying-rate-mps", type=float, default=0.009)
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Also emit simulator targets for invalid candidate actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_candidates)
    rows = read_jsonl(args.input)
    params = estimate_transition_params(
        train_rows,
        default_filling_rate=args.default_filling_rate_mps,
        default_emptying_rate=args.default_emptying_rate_mps,
        gate_transition_duration_sec=args.gate_transition_duration_sec,
    )
    output_rows = build_counterfactual_rows(
        rows,
        params=params,
        include_invalid=args.include_invalid,
    )
    summary = build_summary(
        rows,
        output_rows,
        input_path=args.input,
        train_path=args.train_candidates,
        params=params,
    )
    write_jsonl(args.output, output_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote={args.output}")
    print(f"wrote_summary={args.summary_output}")
    print(f"num_input_candidates={summary['num_input_candidates']}")
    print(f"num_counterfactual_targets={summary['num_counterfactual_targets']}")
    for horizon, metrics in summary["factual_eval"].items():
        print(
            f"{horizon}: state_exact={metrics['state_exact_accuracy']:.3f} "
            f"phase={metrics['phase_accuracy']:.3f} "
            f"water_mae={metrics['water_level_mae']}"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def estimate_transition_params(
    rows: list[dict[str, Any]],
    *,
    default_filling_rate: float,
    default_emptying_rate: float,
    gate_transition_duration_sec: float,
) -> dict[str, Any]:
    rates = {"start_filling": [], "start_emptying": []}
    for row in rows:
        if not row.get("future_gate_water_target_available"):
            continue
        action = row.get("candidate_action")
        if action not in rates:
            continue
        current = as_float((row.get("current_state") or {}).get("water_level"))
        if current is None:
            continue
        horizons = (row.get("future_targets") or {}).get("horizons") or {}
        for horizon_sec in HORIZONS:
            target = horizons.get(horizon_name(horizon_sec))
            if not isinstance(target, dict) or not isinstance(target.get("state"), dict):
                continue
            water = as_float(target["state"].get("water_level"))
            if water is None:
                continue
            rate = abs(water - current) / horizon_sec
            if rate > 1e-5:
                rates[action].append(rate)
    return {
        "filling_rate_mps": robust_rate(rates["start_filling"], default_filling_rate),
        "emptying_rate_mps": robust_rate(rates["start_emptying"], default_emptying_rate),
        "gate_transition_duration_sec": gate_transition_duration_sec,
        "rate_source": {
            "start_filling_samples": len(rates["start_filling"]),
            "start_emptying_samples": len(rates["start_emptying"]),
            "estimator": "median observed absolute water-level delta per second",
        },
    }


def robust_rate(values: list[float], default: float) -> float:
    if not values:
        return default
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def build_counterfactual_rows(
    rows: list[dict[str, Any]],
    *,
    params: dict[str, Any],
    include_invalid: bool,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        is_valid = bool(row.get("is_valid"))
        targets = None
        source = "invalid_action_not_simulated"
        if is_valid or include_invalid:
            targets = simulate_action_targets(row, params=params)
            source = "rule_counterfactual_simulator"
        out.append(
            {
                "row_id": row.get("row_id"),
                "sample_token": row.get("sample_token"),
                "split": row.get("split"),
                "scene_token": row.get("scene_token"),
                "timestamp": row.get("timestamp"),
                "timestamp_str": row.get("timestamp_str"),
                "direction": row.get("direction"),
                "candidate_action": row.get("candidate_action"),
                "is_valid": is_valid,
                "is_factual_action": bool(row.get("future_gate_water_target_available")),
                "violation_reason": row.get("violation_reason") or [],
                "counterfactual_targets": targets,
                "target_source": source,
            }
        )
    return out


def simulate_action_targets(row: dict[str, Any], *, params: dict[str, Any]) -> dict[str, Any]:
    action = str(row.get("candidate_action") or "hold")
    current = row.get("current_state") or {}
    horizons = {}
    for horizon_sec in HORIZONS:
        state = simulate_state(current, action, horizon_sec, params=params)
        horizons[horizon_name(horizon_sec)] = {
            "state": state,
            "phase": state["operation_phase"],
        }
    return {
        "conditioning_action": action,
        "horizons": horizons,
        "simulator": "rule_gate_water_transition_v1",
    }


def simulate_state(
    current: dict[str, Any],
    action: str,
    horizon_sec: int,
    *,
    params: dict[str, Any],
) -> dict[str, Any]:
    upper = str(current.get("upper_gate_state") or "unknown")
    lower = str(current.get("lower_gate_state") or "unknown")
    water_state = str(current.get("water_state") or "idle")
    water_level = as_float(current.get("water_level"))
    upstream_level = as_float(current.get("upstream_water_level"))
    downstream_level = as_float(current.get("downstream_water_level"))

    if action == "open_upper_gate":
        upper = gate_transition_state("open", horizon_sec, params)
    elif action == "close_upper_gate":
        upper = gate_transition_state("closed", horizon_sec, params)
    elif action == "open_lower_gate":
        lower = gate_transition_state("open", horizon_sec, params)
    elif action == "close_lower_gate":
        lower = gate_transition_state("closed", horizon_sec, params)

    if action == "start_filling":
        water_state = "filling"
        water_level = move_water_level(
            water_level,
            upstream_level,
            params["filling_rate_mps"],
            horizon_sec,
        )
    elif action == "start_emptying":
        water_state = "emptying"
        water_level = move_water_level(
            water_level,
            downstream_level,
            params["emptying_rate_mps"],
            horizon_sec,
        )
    elif action == "stop_filling_emptying":
        water_state = "idle"
    elif action == "hold":
        if water_state == "filling":
            water_level = move_water_level(
                water_level,
                upstream_level,
                params["filling_rate_mps"],
                horizon_sec,
            )
        elif water_state == "emptying":
            water_level = move_water_level(
                water_level,
                downstream_level,
                params["emptying_rate_mps"],
                horizon_sec,
            )

    state = {
        "upper_gate_state": upper,
        "lower_gate_state": lower,
        "water_state": water_state,
        "water_level": round(water_level, 4) if water_level is not None else None,
    }
    state["operation_phase"] = infer_operation_phase(state, action)
    if action == "dispatch_enter":
        state["ship_operation_phase"] = "ship_entering"
    elif action == "dispatch_exit":
        state["ship_operation_phase"] = "ship_leaving"
    return state


def gate_transition_state(target: str, horizon_sec: int, params: dict[str, Any]) -> str:
    duration = float(params["gate_transition_duration_sec"])
    if horizon_sec >= duration:
        return target
    return "opening" if target == "open" else "closing"


def move_water_level(
    current: Optional[float],
    target: Optional[float],
    rate_mps: float,
    horizon_sec: int,
) -> Optional[float]:
    if current is None:
        return None
    if target is None:
        return current
    delta = target - current
    max_step = abs(rate_mps) * horizon_sec
    if abs(delta) <= max_step:
        return target
    return current + math.copysign(max_step, delta)


def infer_operation_phase(state: dict[str, Any], action: str = "hold") -> str:
    upper = state.get("upper_gate_state")
    lower = state.get("lower_gate_state")
    water = state.get("water_state")
    if water == "filling" or action == "start_filling":
        return "filling"
    if water == "emptying" or action == "start_emptying":
        return "emptying"
    if action in {"open_upper_gate", "open_lower_gate"} or "opening" in {upper, lower}:
        return "gate_opening"
    if action in {"close_upper_gate", "close_lower_gate"} or "closing" in {upper, lower}:
        return "gate_closing"
    if water == "idle" and upper == "open" and lower == "closed":
        return "upper_gate_open_idle"
    if water == "idle" and upper == "closed" and lower == "open":
        return "lower_gate_open_idle"
    if water == "idle" and upper == "closed" and lower == "closed":
        return "all_gates_closed_idle"
    return "hold_uncertain"


def build_summary(
    input_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    *,
    input_path: Path,
    train_path: Path,
    params: dict[str, Any],
) -> dict[str, Any]:
    factual_rows = {
        str(row.get("row_id")): row
        for row in input_rows
        if row.get("future_gate_water_target_available")
        and isinstance(row.get("future_targets"), dict)
    }
    emitted_factual = [
        row for row in output_rows if str(row.get("row_id")) in factual_rows
    ]
    valid_counts = Counter(
        row["candidate_action"] for row in output_rows if row["counterfactual_targets"]
    )
    return {
        "input": str(input_path),
        "train_candidates": str(train_path),
        "num_input_candidates": len(input_rows),
        "num_counterfactual_targets": sum(
            1 for row in output_rows if row["counterfactual_targets"]
        ),
        "num_factual_eval_rows": len(emitted_factual),
        "counterfactual_action_counts": dict(valid_counts),
        "params": params,
        "factual_eval": factual_eval(factual_rows, emitted_factual),
        "notes": [
            "Counterfactual targets are rule-simulator outputs, not observed GT.",
            "Only the factual observed-action branch is compared against observed future labels.",
            "Invalid actions are emitted without targets unless --include-invalid is used.",
            "Ship dispatch actions only update ship_operation_phase in this gate/water simulator; dense ship rollout is handled separately.",
        ],
    }


def factual_eval(
    factual_by_row_id: dict[str, dict[str, Any]],
    output_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for horizon_sec in HORIZONS:
        key = horizon_name(horizon_sec)
        totals = Counter()
        hits = Counter()
        water_abs_error = 0.0
        water_count = 0
        phase_confusion = Counter()
        for pred_row in output_rows:
            source = factual_by_row_id.get(str(pred_row.get("row_id")))
            if not source:
                continue
            target = (
                source.get("future_targets", {})
                .get("horizons", {})
                .get(key)
            )
            pred = (
                (pred_row.get("counterfactual_targets") or {})
                .get("horizons", {})
                .get(key)
            )
            if not isinstance(target, dict) or not isinstance(target.get("state"), dict):
                continue
            if not isinstance(pred, dict) or not isinstance(pred.get("state"), dict):
                continue
            target_state = target["state"]
            pred_state = pred["state"]
            totals["state"] += 1
            if all(pred_state.get(field) == target_state.get(field) for field in STATE_FIELDS):
                hits["state_exact"] += 1
            for field in STATE_FIELDS:
                totals[field] += 1
                if pred_state.get(field) == target_state.get(field):
                    hits[field] += 1
            totals["phase"] += 1
            target_phase = target.get("phase")
            pred_phase = pred.get("phase")
            phase_confusion[f"{target_phase}->{pred_phase}"] += 1
            if pred_phase == target_phase:
                hits["phase"] += 1
            pw = as_float(pred_state.get("water_level"))
            tw = as_float(target_state.get("water_level"))
            if pw is not None and tw is not None:
                water_abs_error += abs(pw - tw)
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
            "top_phase_confusions": dict(phase_confusion.most_common(20)),
        }
    return out


def horizon_name(horizon_sec: int) -> str:
    return f"t_plus_{horizon_sec}s"


def as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


if __name__ == "__main__":
    main()
