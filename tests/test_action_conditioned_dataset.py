from tools.build_action_conditioned_dataset import (
    build_candidate_rows,
    build_frame_rows,
)
from tools.evaluate_action_conditioned_baseline import evaluate


def test_build_frame_and_candidate_rows_mark_observed_dispatch_and_future_target():
    sequences = [
        {
            "split": "val",
            "scene_token": "scene_1",
            "scene_name": "scene_name",
            "direction": "upstream",
            "operation_date": "2025-10-30",
            "operation_index": 1,
            "line_index": 2,
            "segment_index": 3,
            "frames": [
                {
                    "sample_token": "sample_1",
                    "timestamp": 100,
                    "timestamp_str": "t1",
                    "frame_index": 0,
                    "relative_time_sec": 0.0,
                    "lidar": {"file_name": "lidar.bin"},
                    "images": {"CAM_1": {"file_name": "cam.jpg"}},
                    "lock_state": {
                        "upper_gate_state": "closed",
                        "lower_gate_state": "open",
                        "water_state": "idle",
                        "water_level": -7.6,
                        "observed_action": "hold",
                        "operation_phase": "lower_gate_open_idle",
                        "ship_dispatch_action": "dispatch_enter",
                        "ship_dispatch_target_count": 1,
                        "valid_actions": ["hold", "dispatch_enter"],
                        "invalid_actions": ["open_upper_gate"],
                        "violation_reason": {"open_upper_gate": ["other_gate_not_closed"]},
                        "state_t_plus_10s": {
                            "upper_gate_state": "closed",
                            "lower_gate_state": "open",
                            "water_state": "idle",
                            "water_level": -7.6,
                        },
                        "phase_t_plus_10s": "lower_gate_open_idle",
                    },
                    "instances_3d": [
                        {
                            "instance_token": "instance_ship_001",
                            "annotation_token": "ann_1",
                            "category": "Cargo_ship",
                            "ship_intentions": ["ship_entering_lock"],
                            "assigned_berth_slot": "berth_slot_01",
                            "occlusion_state": "no_or_minor_occlusion",
                            "visibility_level": "v80-100",
                        },
                        {"instance_token": "static_1", "category": "Barrier"},
                    ],
                }
            ],
        }
    ]

    frames = build_frame_rows(sequences)
    candidates = build_candidate_rows(frames)

    assert len(frames) == 1
    frame = frames[0]
    assert frame["conditioning"]["observed_planner_actions"] == ["dispatch_enter"]
    assert frame["conditioning"]["primary_observed_planner_action"] == "dispatch_enter"
    assert frame["sensor"]["lidar_file"] == "lidar.bin"
    assert len(frame["ship_context"]) == 1

    dispatch_candidate = next(
        item for item in frame["candidate_actions"] if item["action"] == "dispatch_enter"
    )
    assert dispatch_candidate["is_valid"] is True
    assert dispatch_candidate["is_observed_planner_action"] is True
    assert dispatch_candidate["future_gate_water_target_available"] is False

    hold_candidate = next(item for item in candidates if item["candidate_action"] == "hold")
    assert hold_candidate["future_gate_water_target_available"] is True
    assert hold_candidate["future_targets"]["horizons"]["t_plus_10s"]["phase"] == (
        "lower_gate_open_idle"
    )


def test_evaluate_action_conditioned_baselines():
    rows = [
        {
            "split": "val",
            "conditioning": {
                "observed_planner_actions": ["dispatch_enter"],
                "primary_observed_planner_action": "dispatch_enter",
            },
            "current_state": {
                "valid_actions": ["hold", "dispatch_enter"],
                "upper_gate_state": "closed",
                "lower_gate_state": "open",
                "water_state": "idle",
                "water_level": -7.6,
                "operation_phase": "lower_gate_open_idle",
            },
            "future_targets": {
                "horizons": {
                    "t_plus_10s": {
                        "state": {
                            "upper_gate_state": "closed",
                            "lower_gate_state": "open",
                            "water_state": "idle",
                            "water_level": -7.5,
                        },
                        "phase": "lower_gate_open_idle",
                    },
                    "t_plus_20s": {"state": None, "phase": None},
                    "t_plus_30s": {"state": None, "phase": None},
                }
            },
        }
    ]

    summary = evaluate(rows, input_path="dummy.jsonl")

    assert summary["target_validity"]["primary_valid_rate"] == 1.0
    assert summary["planner_baselines"]["hold"]["legal_rate"] == 1.0
    assert summary["planner_baselines"]["hold"]["target_set_accuracy"] == 0.0
    assert (
        summary["planner_baselines"]["observed_valid_or_hold_oracle"][
            "target_set_accuracy"
        ]
        == 1.0
    )
    assert (
        summary["future_persistence_baseline"]["t_plus_10s"]["state_exact_accuracy"]
        == 1.0
    )
    assert summary["future_persistence_baseline"]["t_plus_10s"]["water_level_mae"] == 0.1


def test_future_persistence_can_use_deployable_current_state():
    rows = [
        {
            "sample_token": "sample_1",
            "split": "val",
            "direction": "upstream",
            "conditioning": {
                "observed_planner_actions": ["hold"],
                "primary_observed_planner_action": "hold",
            },
            "current_state": {
                "valid_actions": ["hold"],
                "upper_gate_state": "closed",
                "lower_gate_state": "closed",
                "water_state": "idle",
                "water_level": -6.0,
                "operation_phase": "gt_only_wrong_phase",
            },
            "future_targets": {
                "horizons": {
                    "t_plus_10s": {
                        "state": {
                            "upper_gate_state": "closed",
                            "lower_gate_state": "closed",
                            "water_state": "idle",
                            "water_level": -6.0,
                        },
                        "phase": "chamber_closed_idle",
                    },
                    "t_plus_20s": {"state": None, "phase": None},
                    "t_plus_30s": {"state": None, "phase": None},
                }
            },
        }
    ]
    deployable_world_state_by_sample = {
        "sample_1": {
            "sample_token": "sample_1",
            "lock_occupancy": {
                "current": {
                    "berth_slots": [],
                    "coarse_regions": [
                        {"region_id": "upper_gate_zone", "ship_count": 0},
                        {"region_id": "lower_gate_zone", "ship_count": 0},
                        {"region_id": "between_berths", "ship_count": 0},
                    ],
                    "num_ships": 0,
                }
            },
            "vessel_motion_flow": {"input_window": []},
        }
    }

    summary = evaluate(
        rows,
        input_path="dummy.jsonl",
        deployable_world_state_by_sample=deployable_world_state_by_sample,
    )

    assert (
        summary["future_persistence_baseline_source"]
        == "deployable_world_state_current_state"
    )
    assert (
        summary["future_persistence_baseline"]["t_plus_10s"]["phase_accuracy"]
        == 1.0
    )
    assert (
        summary["future_gt_structured_persistence_diagnostic"]["t_plus_10s"][
            "phase_accuracy"
        ]
        == 0.0
    )
    assert summary["future_deployable_replacement_report"]["replaced_frames"] == 1
