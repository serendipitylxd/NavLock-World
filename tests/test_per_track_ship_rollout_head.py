from tools.train_per_track_ship_rollout_head import (
    PER_TRACK_CALIBRATED_MODE,
    PER_TRACK_MODE,
    assign_future_berths,
    build_track_history_features,
    build_predictions,
    build_summary,
    normalize_assignment_config,
    train_per_track_model,
)


def world_row(sample_token="sample_1"):
    return {
        "sample_token": sample_token,
        "split": "val",
        "scene_token": "scene_upstream_1",
        "timestamp": 1_000_000,
        "lock_occupancy": {
            "current": {
                "berth_slots": [
                    {
                        "region_id": "berth_slot_01",
                        "occupied": False,
                        "ship_count": 0,
                        "ship_tokens": [],
                    }
                ],
                "coarse_regions": [
                    {
                        "region_id": "upper_gate_zone",
                        "ship_count": 0,
                        "ship_tokens": [],
                    },
                    {
                        "region_id": "lower_gate_zone",
                        "ship_count": 0,
                        "ship_tokens": [],
                    },
                    {
                        "region_id": "outside_lock_width",
                        "ship_count": 0,
                        "ship_tokens": [],
                    },
                    {
                        "region_id": "between_berths",
                        "ship_count": 1,
                        "ship_tokens": ["hydro_track_1"],
                    },
                ],
                "num_occupied_berths": 0,
                "num_ships": 1,
            }
        },
        "track_source": {"window_size": 2},
        "vessel_motion_flow": {
            "input_window": [
                {
                    "instance_token": "hydro_track_1",
                    "category": "Fully_loaded_cargo_ship",
                    "motion_state": "ship_entering_lock",
                    "start_region": "upper_gate_zone",
                    "end_region": "between_berths",
                    "direction_label": "toward_lower_gate",
                    "end_speed_mps": 0.4,
                    "delta_xy": [0.0, -4.0],
                }
            ]
        },
    }


def label_row(sample_token="sample_1"):
    return {
        "sample_token": sample_token,
        "split": "val",
        "scene_token": "scene_upstream_1",
        "timestamp": 1_000_000,
        "dense_ship_future_targets": {
            "horizons": {
                "t_plus_10s": {
                    "future_occupancy": {
                        "berth_slots": [
                            {
                                "region_id": "berth_slot_01",
                                "occupied": True,
                                "ship_count": 1,
                                "ship_tokens": ["annotation_ship_1"],
                            }
                        ],
                        "coarse_regions": [
                            {
                                "region_id": "between_berths",
                                "ship_count": 1,
                                "ship_tokens": ["annotation_ship_1"],
                            },
                            {"region_id": "upper_gate_zone", "ship_count": 0},
                            {"region_id": "lower_gate_zone", "ship_count": 0},
                            {"region_id": "outside_lock_width", "ship_count": 0},
                        ],
                        "num_occupied_berths": 1,
                        "num_ships": 1,
                    },
                    "matched_ships": [
                        {
                            "instance_token": "annotation_ship_1",
                            "category": "Unladen_cargo_ship",
                            "current_region": "between_berths",
                            "current_berth_slot": None,
                            "current_motion_state": "ship_entering_lock",
                            "future_region": "between_berths",
                            "future_berth_slot": "berth_slot_01",
                            "target_motion_state": "ship_berthed",
                        }
                    ],
                },
                "t_plus_20s": None,
                "t_plus_30s": None,
            }
        },
    }


def test_per_track_rollout_learns_future_berth_assignment():
    world = world_row()
    label = label_row()
    model = train_per_track_model([world], [label], use_history_features=False)

    predictions = build_predictions([label], {"sample_1": world}, model)
    pred = predictions[0]["rollout_modes"][PER_TRACK_MODE]["t_plus_10s"]

    assert pred["num_ships"] == 1
    assert pred["future_occupancy"]["berth_slots"][0]["occupied"] is True
    assert pred["motion_counts"]["ship_berthed"] == 1
    assert pred["per_track_predictions"][0]["future_berth_slot"] == "berth_slot_01"


def test_berth_assignment_keeps_highest_score_per_slot():
    assigned = assign_future_berths(
        [
            {
                "track_token": "track_low",
                "future_berth_slot": "berth_slot_01",
                "future_region": "between_berths",
                "scores": {"future_berth_slot": 0.20},
            },
            {
                "track_token": "track_high",
                "future_berth_slot": "berth_slot_01",
                "future_region": "between_berths",
                "scores": {"future_berth_slot": 0.90},
            },
        ]
    )

    by_token = {item["track_token"]: item for item in assigned}
    assert by_token["track_high"]["future_berth_slot"] == "berth_slot_01"
    assert by_token["track_low"]["future_berth_slot"] == "__none__"


def test_berth_assignment_applies_calibrated_threshold():
    assigned = assign_future_berths(
        [
            {
                "track_token": "track_low",
                "current_berth_slot": None,
                "future_berth_slot": "berth_slot_01",
                "future_region": "between_berths",
                "scores": {"future_berth_slot": 0.20},
            }
        ],
        config={"min_berth_score": 0.50, "keep_current_min_score": 0.10},
    )

    assert assigned[0]["future_berth_slot"] == "__none__"


def test_berth_assignment_uses_lower_keep_threshold_for_current_slot():
    assigned = assign_future_berths(
        [
            {
                "track_token": "track_current",
                "current_berth_slot": "berth_slot_01",
                "future_berth_slot": "berth_slot_01",
                "future_region": "between_berths",
                "scores": {"future_berth_slot": 0.20},
            }
        ],
        config={"min_berth_score": 0.50, "keep_current_min_score": 0.10},
    )

    assert assigned[0]["future_berth_slot"] == "berth_slot_01"


def test_track_history_features_include_temporal_dwell():
    first = world_row("sample_1")
    second = world_row("sample_2")
    second["timestamp"] = 11_000_000
    second["vessel_motion_flow"]["input_window"][0]["end_speed_mps"] = 0.6

    history = build_track_history_features([first, second])
    second_hist = history["sample_2"]["hydro_track_1"]

    assert second_hist["has_prev_track"] is True
    assert second_hist["track_seen_frame_count"] == 2
    assert second_hist["track_age_sec"] == 10.0
    assert second_hist["region_dwell_frame_count"] == 2
    assert second_hist["speed_delta_prev"] == 0.19999999999999996


def test_per_track_summary_uses_existing_rollout_metrics():
    world = world_row()
    label = label_row()
    model = train_per_track_model([world], [label], use_history_features=False)
    predictions = build_predictions([label], {"sample_1": world}, model)

    summary = build_summary(
        [world],
        [label],
        [world],
        [label],
        predictions,
        model=model,
        train_world_state="train_world.jsonl",
        train_dense_labels="train_dense.jsonl",
        eval_world_state="eval_world.jsonl",
        eval_dense_labels="eval_dense.jsonl",
        output_model="model.pkl",
        prediction_output="pred.jsonl",
    )

    metrics = summary["rollout_metrics"][PER_TRACK_MODE]["t_plus_10s"]
    assert summary["track_training_rows"] == 1
    assert metrics["berth_occupied_f1"] == 1.0
    assert metrics["coarse_region_count_f1"] == 1.0
    assert metrics["motion_count_f1"] == 1.0
    assert PER_TRACK_CALIBRATED_MODE in summary["rollout_metrics"]
    assert "assignment_calibration" in summary
    assert "rollout_error_breakdown" in summary
    assert normalize_assignment_config({})["min_berth_score"] == 0.0
