from tools.build_counterfactual_action_transitions import (
    build_counterfactual_rows,
    build_summary,
    estimate_transition_params,
    simulate_state,
)


def row(action="start_filling", *, valid=True, factual=True):
    return {
        "row_id": f"sample_1::{action}",
        "sample_token": "sample_1",
        "candidate_action": action,
        "is_valid": valid,
        "future_gate_water_target_available": factual,
        "current_state": {
            "upper_gate_state": "closed",
            "lower_gate_state": "closed",
            "water_state": "idle",
            "water_level": -7.0,
            "upstream_water_level": -6.0,
            "downstream_water_level": -8.0,
        },
        "future_targets": {
            "horizons": {
                "t_plus_10s": {
                    "state": {
                        "upper_gate_state": "closed",
                        "lower_gate_state": "closed",
                        "water_state": "filling",
                        "water_level": -6.9,
                    },
                    "phase": "filling",
                },
                "t_plus_20s": None,
                "t_plus_30s": None,
            }
        },
    }


def params():
    return {
        "filling_rate_mps": 0.01,
        "emptying_rate_mps": 0.01,
        "gate_transition_duration_sec": 45.0,
    }


def test_simulate_start_filling_moves_water_toward_upstream_level():
    state = simulate_state(row()["current_state"], "start_filling", 10, params=params())

    assert state["water_state"] == "filling"
    assert state["operation_phase"] == "filling"
    assert round(state["water_level"], 3) == -6.9


def test_counterfactual_rows_skip_invalid_by_default():
    rows = [row("start_filling", valid=True), row("open_upper_gate", valid=False)]

    out = build_counterfactual_rows(rows, params=params(), include_invalid=False)

    assert out[0]["counterfactual_targets"] is not None
    assert out[1]["counterfactual_targets"] is None


def test_summary_factual_eval_uses_observed_action_branch():
    rows = [row("start_filling", valid=True, factual=True)]
    out = build_counterfactual_rows(rows, params=params(), include_invalid=False)

    summary = build_summary(
        rows,
        out,
        input_path="eval.jsonl",
        train_path="train.jsonl",
        params=params(),
    )

    assert summary["num_factual_eval_rows"] == 1
    assert summary["factual_eval"]["t_plus_10s"]["state_exact_accuracy"] == 1.0
    assert summary["factual_eval"]["t_plus_10s"]["phase_accuracy"] == 1.0


def test_estimate_transition_params_uses_observed_rate():
    train = [row("start_filling", valid=True, factual=True)]

    estimated = estimate_transition_params(
        train,
        default_filling_rate=0.5,
        default_emptying_rate=0.5,
        gate_transition_duration_sec=20.0,
    )

    assert round(estimated["filling_rate_mps"], 3) == 0.01
    assert estimated["emptying_rate_mps"] == 0.5
