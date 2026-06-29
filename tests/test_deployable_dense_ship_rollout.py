from tools.evaluate_deployable_dense_ship_rollout import (
    build_predictions,
    build_summary,
    metrics_for_mode,
)


def world_row():
    return {
        "sample_token": "sample_1",
        "lock_occupancy": {
            "current": {
                "berth_slots": [
                    {
                        "region_id": "berth_slot_01",
                        "occupied": True,
                        "ship_count": 1,
                        "ship_tokens": ["hydro_track_001"],
                    }
                ],
                "coarse_regions": [
                    {
                        "region_id": "between_berths",
                        "ship_count": 1,
                        "ship_tokens": ["hydro_track_001"],
                    }
                ],
                "num_ships": 1,
            }
        },
        "vessel_motion_flow": {
            "input_window": [
                {
                    "instance_token": "hydro_track_001",
                    "motion_state": "ship_berthed",
                }
            ]
        },
    }


def label_row():
    return {
        "sample_token": "sample_1",
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
                            }
                        ],
                        "num_ships": 1,
                    },
                    "matched_ships": [
                        {
                            "instance_token": "annotation_ship_1",
                            "target_motion_state": "ship_berthed",
                        }
                    ],
                }
            }
        },
    }


def test_deployable_rollout_scores_spatial_state_without_token_identity():
    predictions = build_predictions([label_row()], {"sample_1": world_row()})

    metrics = metrics_for_mode(
        [label_row()],
        {row["sample_token"]: row for row in predictions},
        "deployable_persistence",
    )

    assert metrics["t_plus_10s"]["berth_occupied_f1"] == 1.0
    assert metrics["t_plus_10s"]["coarse_region_count_f1"] == 1.0
    assert metrics["t_plus_10s"]["motion_count_f1"] == 1.0


def test_deployable_rollout_summary_records_matching_frames():
    predictions = build_predictions([label_row()], {"sample_1": world_row()})

    summary = build_summary(
        [label_row()],
        predictions,
        deployable_world_state="world.jsonl",
        dense_labels="dense.jsonl",
        prediction_output="pred.jsonl",
    )

    assert summary["matched_deployable_frames"] == 1
    assert "deployable_persistence" in summary["rollout_metrics"]
