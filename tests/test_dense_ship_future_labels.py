from tools.build_dense_ship_future_labels import (
    baseline_metrics,
    build_dense_ship_future_labels,
    horizon_coverage,
)


BERTHS = [
    {
        "slot_id": "berth_slot_01",
        "x_min": 40.0,
        "x_max": 60.0,
        "y_min": 40.0,
        "y_max": 70.0,
        "cx": 50.0,
        "cy": 55.0,
    }
]
CHAMBER = {"x_min": 40.0, "x_max": 60.0, "y_min": 0.0, "y_max": 100.0}


def frame(token, ts_sec, y, intention):
    return {
        "sample_token": token,
        "timestamp": int(ts_sec * 1_000_000),
        "timestamp_str": token,
        "instances_3d": [
            {
                "instance_token": "ship_1",
                "category": "Fully_loaded_cargo_ship",
                "translation": [50.0, y, 0.0],
                "ship_intentions": [intention],
                "attribute_names": [],
            }
        ],
    }


def test_dense_ship_future_labels_align_future_frame_and_ship_target():
    sequences = [
        {
            "split": "val",
            "scene_token": "scene_1",
            "direction": "upstream",
            "frames": [
                frame("sample_0", 0, 20.0, "ship_entering_lock"),
                frame("sample_10", 10, 50.0, "ship_berthed"),
            ],
        }
    ]

    rows = build_dense_ship_future_labels(
        sequences,
        berths_by_scene={"scene_1": BERTHS},
        chamber=CHAMBER,
        horizons_sec=[10],
        max_time_delta_sec=0.5,
    )
    label = rows[0]["dense_ship_future_targets"]["horizons"]["t_plus_10s"]

    assert label["sample_token"] == "sample_10"
    assert label["future_occupancy"]["berth_slots"][0]["occupied"] is True
    assert label["matched_ships"][0]["target_motion_state"] == "ship_berthed"
    assert label["matched_ships"][0]["future_berth_slot"] == "berth_slot_01"


def test_dense_ship_summary_helpers_count_coverage_and_baseline_metrics():
    sequences = [
        {
            "split": "val",
            "scene_token": "scene_1",
            "direction": "upstream",
            "frames": [
                frame("sample_0", 0, 50.0, "ship_berthed"),
                frame("sample_10", 10, 50.0, "ship_berthed"),
            ],
        }
    ]
    rows = build_dense_ship_future_labels(
        sequences,
        berths_by_scene={"scene_1": BERTHS},
        chamber=CHAMBER,
        horizons_sec=[10],
        max_time_delta_sec=0.5,
    )

    assert horizon_coverage(rows, [10])["t_plus_10s"] == 1
    metrics = baseline_metrics(rows, [10], "persistence_prediction")

    assert metrics["t_plus_10s"]["berth_occupied_f1"] == 1.0
    assert metrics["t_plus_10s"]["motion_accuracy"] == 1.0
