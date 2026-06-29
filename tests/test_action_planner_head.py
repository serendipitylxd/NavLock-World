import numpy as np

from tools.train_action_planner_head import (
    build_history_features,
    featurize_candidate,
    frame_level_action_head_predictions,
    frame_level_predictions,
)


def candidate(action, *, valid, target=False):
    return {
        "candidate_action": action,
        "is_valid": valid,
        "is_rule_consistent_planner_action": target,
        "sample_token": "sample_1",
        "split": "val",
        "scene_token": "scene_1",
        "timestamp": 1,
        "timestamp_str": "t1",
        "direction": "upstream",
        "violation_reason": [] if valid else ["blocked"],
        "current_state": {
            "upper_gate_state": "closed",
            "lower_gate_state": "open",
            "water_state": "idle",
            "water_level": -7.6,
            "upstream_water_level": -4.8,
            "downstream_water_level": -7.6,
            "observed_action": "hold",
            "ship_dispatch_action": "dispatch_enter",
            "valid_actions": ["hold", "dispatch_enter"],
            "violation_reason": {"open_upper_gate": ["other_gate_not_closed"]},
            "operation_phase": "lower_gate_open_idle",
            "ship_operation_phase": "ship_entering",
            "entry_path_clear": True,
            "exit_path_clear": True,
            "chamber_capacity_available": True,
            "num_ships_in_chamber": 1,
        },
    }


def test_featurizer_excludes_observed_and_mask_fields_by_default():
    features = featurize_candidate(
        candidate("dispatch_enter", valid=True),
        include_observed_action_features=False,
    )

    assert "observed_action" not in features
    assert "ship_dispatch_action" not in features
    assert "valid_actions" not in features
    assert "violation_reason" not in features
    assert features["candidate_action"] == "dispatch_enter"
    assert features["operation_phase"] == "lower_gate_open_idle"


def test_featurizer_can_include_observed_fields_for_ablation():
    features = featurize_candidate(
        candidate("dispatch_enter", valid=True),
        include_observed_action_features=True,
    )

    assert features["observed_action"] == "hold"
    assert features["ship_dispatch_action"] == "dispatch_enter"
    assert features["candidate_matches_ship_dispatch_action"] is True


def test_history_features_use_previous_frame_only():
    first = candidate("hold", valid=True)
    first["sample_token"] = "sample_1"
    first["timestamp"] = 1_000_000
    first["current_state"]["water_state"] = "idle"
    first["current_state"]["operation_phase"] = "lower_gate_open_idle"
    first["current_state"]["water_level"] = -7.0
    second = candidate("hold", valid=True)
    second["sample_token"] = "sample_2"
    second["timestamp"] = 3_000_000
    second["current_state"]["water_state"] = "filling"
    second["current_state"]["operation_phase"] = "filling"
    second["current_state"]["water_level"] = -6.8
    third = candidate("hold", valid=True)
    third["sample_token"] = "sample_3"
    third["timestamp"] = 6_000_000
    third["current_state"]["water_state"] = "filling"
    third["current_state"]["operation_phase"] = "filling"
    third["current_state"]["water_level"] = -6.2

    history = build_history_features([third, second, first])

    assert history["sample_1"]["has_prev_frame"] is False
    assert history["sample_2"]["prev_water_state"] == "idle"
    assert history["sample_2"]["water_state_changed_from_prev"] is True
    assert np.isclose(history["sample_2"]["water_level_delta_prev"], 0.2)
    assert history["sample_2"]["water_state_run_sec"] == 0.0
    assert history["sample_3"]["prev_water_state"] == "filling"
    assert history["sample_3"]["water_state_run_sec"] == 3.0
    assert history["sample_3"]["water_state_run_frame_count"] == 2


def test_hard_mask_blocks_illegal_high_scoring_action():
    rows = [
        candidate("hold", valid=True, target=False),
        candidate("open_upper_gate", valid=False, target=False),
        candidate("dispatch_enter", valid=True, target=True),
    ]
    validity_scores = np.array([0.7, 0.99, 0.8])
    planner_scores = np.array([0.2, 0.99, 0.6])

    hard = frame_level_predictions(
        rows,
        validity_scores=validity_scores,
        planner_scores=planner_scores,
        hard_mask=True,
    )
    unconstrained = frame_level_predictions(
        rows,
        validity_scores=validity_scores,
        planner_scores=planner_scores,
        hard_mask=False,
    )

    assert hard[0]["predicted_action"] == "dispatch_enter"
    assert hard[0]["is_legal"] is True
    assert hard[0]["target_set_hit"] is True
    assert unconstrained[0]["predicted_action"] == "open_upper_gate"
    assert unconstrained[0]["is_legal"] is False


class DummyActionHead:
    classes_ = np.array(["hold", "open_upper_gate", "dispatch_enter"])

    def predict_proba(self, x):
        return np.array([[0.1, 0.8, 0.7]])


class DummyVectorizer:
    def transform(self, rows):
        return rows


class HoldBiasedActionHead:
    classes_ = np.array(["hold", "dispatch_exit"])

    def predict_proba(self, x):
        return np.array([[0.95, 0.05]])


class HoldBiasedEnterActionHead:
    classes_ = np.array(["hold", "dispatch_enter"])

    def predict_proba(self, x):
        return np.array([[0.95, 0.05]])


def test_frame_action_head_applies_hard_mask_to_multiclass_scores():
    rows = [
        candidate("hold", valid=True, target=False),
        candidate("open_upper_gate", valid=False, target=False),
        candidate("dispatch_enter", valid=True, target=True),
    ]
    predictions = frame_level_action_head_predictions(
        {"vectorizer": DummyVectorizer(), "action_head": DummyActionHead()},
        rows,
        hard_mask=True,
        include_observed_action_features=False,
    )

    assert predictions[0]["predicted_action"] == "dispatch_enter"
    assert predictions[0]["is_legal"] is True
    assert predictions[0]["target_set_hit"] is True


def test_dispatch_continuity_override_keeps_leaving_ship_moving():
    rows = [
        candidate("hold", valid=True, target=False),
        candidate("dispatch_exit", valid=True, target=True),
    ]
    for row in rows:
        row["current_state"]["ship_operation_phase"] = "ship_leaving"
        row["current_state"]["upper_gate_state"] = "open"
        row["current_state"]["lower_gate_state"] = "closed"
        row["current_state"]["water_state"] = "idle"
        row["current_state"]["next_ship_to_leave_weak"] = {
            "source": "inside_exit_side_distance",
            "speed_mps": 0.5,
        }
    predictions = frame_level_action_head_predictions(
        {"vectorizer": DummyVectorizer(), "action_head": HoldBiasedActionHead()},
        rows,
        hard_mask=True,
        include_observed_action_features=False,
    )

    assert predictions[0]["raw_predicted_action"] == "hold"
    assert predictions[0]["predicted_action"] == "dispatch_exit"
    assert predictions[0]["postprocess_rule"] == "dispatch_exit_continuity"
    assert predictions[0]["target_set_hit"] is True


def test_dispatch_continuity_override_keeps_entering_ship_moving():
    rows = [
        candidate("hold", valid=True, target=False),
        candidate("dispatch_enter", valid=True, target=True),
    ]
    for row in rows:
        row["direction"] = "upstream"
        row["current_state"]["ship_operation_phase"] = "ship_entering"
        row["current_state"]["upper_gate_state"] = "closed"
        row["current_state"]["lower_gate_state"] = "open"
        row["current_state"]["water_state"] = "idle"
        row["current_state"]["next_ship_to_enter_weak"] = {
            "source": "fused_vessel_motion_flow",
            "speed_mps": 0.5,
        }
    predictions = frame_level_action_head_predictions(
        {"vectorizer": DummyVectorizer(), "action_head": HoldBiasedEnterActionHead()},
        rows,
        hard_mask=True,
        include_observed_action_features=False,
    )

    assert predictions[0]["raw_predicted_action"] == "hold"
    assert predictions[0]["predicted_action"] == "dispatch_enter"
    assert predictions[0]["postprocess_rule"] == "dispatch_enter_continuity"
    assert predictions[0]["target_set_hit"] is True
