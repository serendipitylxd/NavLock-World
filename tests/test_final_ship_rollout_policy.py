from tools.build_final_ship_rollout_policy import (
    FINAL_MODE,
    build_final_predictions,
)


def aggregate_row():
    return {
        "sample_token": "sample_1",
        "rollout_modes": {
            "deployable_berth_learned_motion_rollout": {
                "t_plus_10s": horizon("agg_10", "agg_motion"),
                "t_plus_20s": horizon("agg_20", "agg_motion"),
                "t_plus_30s": horizon("agg_30", "agg_motion"),
            },
            "deployable_persistence": {
                "t_plus_10s": horizon("persist", "persist_motion"),
                "t_plus_20s": horizon("persist", "persist_motion"),
                "t_plus_30s": horizon("persist", "persist_motion"),
            },
            "dispatch_aware_rollout": {
                "t_plus_10s": horizon("dispatch", "dispatch_motion"),
                "t_plus_20s": horizon("dispatch", "dispatch_motion"),
                "t_plus_30s": horizon("dispatch", "dispatch_motion"),
            },
        },
    }


def per_track_row():
    return {
        "sample_token": "sample_1",
        "rollout_modes": {
            "per_track_berth_hybrid_rollout": {
                "t_plus_10s": horizon("track_10", "track_motion"),
                "t_plus_20s": horizon("track_20", "track_motion"),
                "t_plus_30s": horizon("track_30", "track_motion"),
            }
        },
    }


def horizon(region_token, motion_token):
    return {
        "future_occupancy": {
            "berth_slots": [{"region_id": "berth_slot_01", "occupied": True}],
            "coarse_regions": [{"region_id": region_token, "ship_count": 1}],
            "num_occupied_berths": 1,
            "num_ships": 1,
        },
        "motion_counts": {motion_token: 1},
        "num_ships": 1,
    }


def test_final_policy_uses_aggregate_short_and_track_aware_long_coarse():
    predictions = build_final_predictions([aggregate_row()], [per_track_row()])
    final = predictions[0]["rollout_modes"][FINAL_MODE]

    assert final["t_plus_10s"]["future_occupancy"]["coarse_regions"][0]["region_id"] == "agg_10"
    assert final["t_plus_20s"]["future_occupancy"]["coarse_regions"][0]["region_id"] == "track_20"
    assert final["t_plus_30s"]["future_occupancy"]["coarse_regions"][0]["region_id"] == "track_30"
    assert final["t_plus_20s"]["motion_counts"] == {"agg_motion": 1}
    assert final["t_plus_20s"]["policy_sources"]["coarse_regions"] == "per_track_berth_hybrid_rollout"
