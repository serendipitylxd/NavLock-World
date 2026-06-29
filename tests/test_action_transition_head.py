from tools.train_action_transition_head import (
    evaluate_transition_model,
    future_supervised_rows,
    persistence_metrics,
    train_transition_model,
)


def candidate_row(sample, action, *, future=True, phase="upper_gate_open_idle"):
    return {
        "sample_token": sample,
        "split": "val",
        "scene_token": "scene_1",
        "timestamp": 1_000_000 if sample == "sample_1" else 2_000_000,
        "timestamp_str": sample,
        "direction": "upstream",
        "candidate_action": action,
        "future_gate_water_target_available": future,
        "current_state": {
            "upper_gate_state": "open",
            "lower_gate_state": "closed",
            "water_state": "idle",
            "water_level": -5.0,
            "operation_phase": "upper_gate_open_idle",
            "ship_operation_phase": "all_ships_berthed",
            "entry_path_clear": True,
            "exit_path_clear": True,
            "chamber_capacity_available": False,
            "num_occupied_berths": 1,
            "num_ships_in_chamber": 1,
            "max_parallel_entries": 0,
            "max_parallel_departures": 1,
        },
        "future_targets": {
            "horizons": {
                "t_plus_10s": {
                    "state": {
                        "upper_gate_state": "open",
                        "lower_gate_state": "closed",
                        "water_state": "idle",
                        "water_level": -5.0,
                    },
                    "phase": phase,
                },
                "t_plus_20s": {
                    "state": {
                        "upper_gate_state": "open",
                        "lower_gate_state": "closed",
                        "water_state": "idle",
                        "water_level": -5.0,
                    },
                    "phase": phase,
                },
                "t_plus_30s": None,
            }
        }
        if future
        else None,
    }


def test_future_supervised_rows_excludes_counterfactual_candidates():
    rows = [
        candidate_row("sample_1", "hold", future=True),
        candidate_row("sample_1", "start_filling", future=False),
    ]

    supervised = future_supervised_rows(rows)

    assert len(supervised) == 1
    assert supervised[0]["candidate_action"] == "hold"


def test_persistence_metrics_uses_current_state_phase():
    row = candidate_row("sample_1", "hold", future=True, phase="filling")

    metrics = persistence_metrics([row])

    assert metrics["t_plus_10s"]["state_exact_accuracy"] == 1.0
    assert metrics["t_plus_10s"]["phase_accuracy"] == 0.0


def test_train_and_evaluate_transition_head_smoke():
    rows = [
        candidate_row("sample_1", "hold", future=True),
        candidate_row("sample_1", "start_filling", future=False),
        candidate_row("sample_2", "hold", future=True, phase="upper_gate_open_idle"),
    ]

    model = train_transition_model(rows, use_history_features=True, max_iter=100)
    result = evaluate_transition_model(model, rows)

    assert result["summary"]["num_supervised_rows"] == 2
    assert result["summary"]["transition_head"]["t_plus_10s"]["num_targets"] == 2
    assert result["summary"]["hold_persistence_hybrid"]["t_plus_10s"]["num_targets"] == 2
    assert result["summary"]["persistence_baseline"]["t_plus_10s"]["num_targets"] == 2
    assert result["predictions"]
    assert result["hybrid_predictions"]
