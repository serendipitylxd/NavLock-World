from tools.train_deployable_ship_rollout_head import (
    LEARNED_MODE,
    build_predictions,
    build_summary,
    normalize_berth_slot_counts,
    train_rollout_model,
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
                    {"region_id": "berth_slot_01", "occupied": True, "ship_count": 1}
                ],
                "coarse_regions": [
                    {"region_id": "between_berths", "ship_count": 1},
                    {"region_id": "upper_gate_zone", "ship_count": 0},
                    {"region_id": "lower_gate_zone", "ship_count": 0},
                    {"region_id": "outside_lock_width", "ship_count": 0},
                ],
                "num_occupied_berths": 1,
                "num_ships": 1,
            }
        },
        "planner_feature_stitch": {
            "ship_operation_phase": "all_ships_berthed",
            "source": "test",
        },
        "vessel_motion_flow": {
            "input_window": [
                {
                    "instance_token": "hydro_track_1",
                    "motion_state": "ship_berthed",
                    "start_region": "between_berths",
                    "end_region": "between_berths",
                    "end_speed_mps": 0.0,
                    "delta_xy": [0.0, 0.0],
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
                            }
                        ],
                        "coarse_regions": [
                            {"region_id": "between_berths", "ship_count": 1},
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
                            "target_motion_state": "ship_berthed",
                        }
                    ],
                },
                "t_plus_20s": None,
                "t_plus_30s": None,
            }
        },
    }


def test_learned_deployable_ship_rollout_predicts_aggregate_state():
    world = world_row()
    label = label_row()
    model = train_rollout_model([world], [label], use_history_features=False)

    predictions = build_predictions([label], {"sample_1": world}, model)
    pred = predictions[0]["rollout_modes"][LEARNED_MODE]["t_plus_10s"]

    assert pred["num_ships"] == 1
    assert pred["future_occupancy"]["berth_slots"][0]["occupied"] is True
    assert pred["future_occupancy"]["coarse_regions"][0]["ship_count"] == 1
    assert pred["motion_counts"]["ship_berthed"] == 1


def test_learned_deployable_ship_rollout_summary_uses_existing_metrics():
    world = world_row()
    label = label_row()
    model = train_rollout_model([world], [label], use_history_features=False)
    predictions = build_predictions([label], {"sample_1": world}, model)

    summary = build_summary(
        [world],
        [label],
        [label],
        predictions,
        model=model,
        train_world_state="train_world.jsonl",
        train_dense_labels="train_dense.jsonl",
        eval_world_state="eval_world.jsonl",
        eval_dense_labels="eval_dense.jsonl",
        prediction_output="pred.jsonl",
        output_model="model.pkl",
    )

    learned = summary["rollout_metrics"][LEARNED_MODE]["t_plus_10s"]
    assert summary["matched_eval_frames"] == 1
    assert learned["berth_occupied_f1"] == 1.0
    assert learned["coarse_region_count_f1"] == 1.0
    assert learned["motion_count_f1"] == 1.0


def test_berth_slot_predictions_are_capped_by_ship_count():
    counts = normalize_berth_slot_counts(
        {
            "berth_slot_01": (True, 0.55),
            "berth_slot_02": (True, 0.90),
            "berth_slot_03": (False, 0.20),
        },
        num_ships=1,
    )

    assert counts == {
        "berth_slot_01": 0,
        "berth_slot_02": 1,
        "berth_slot_03": 0,
    }


def test_learned_berth_delta_can_fill_empty_current_slot():
    world = world_row()
    world["lock_occupancy"]["current"]["berth_slots"][0]["occupied"] = False
    world["lock_occupancy"]["current"]["berth_slots"][0]["ship_count"] = 0
    world["lock_occupancy"]["current"]["num_occupied_berths"] = 0
    label = label_row()
    model = train_rollout_model([world], [label], use_history_features=False)

    predictions = build_predictions([label], {"sample_1": world}, model)
    pred = predictions[0]["rollout_modes"][LEARNED_MODE]["t_plus_10s"]

    assert pred["future_occupancy"]["berth_slots"][0]["occupied"] is True
