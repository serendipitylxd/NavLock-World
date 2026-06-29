from tools.build_action_future_labels import (
    build_action_future_labels,
    build_ship_context_labels,
)


def row(token, scene, ts_sec, phase, action="hold"):
    return {
        "token": token,
        "scene_token": scene,
        "timestamp": int(ts_sec * 1_000_000),
        "upper_gate_state": "closed",
        "lower_gate_state": "open",
        "lock_water_state": "idle",
        "water_level": -7.5,
        "operation_phase": phase,
        "observed_action": action,
    }


def test_action_future_labels_stay_within_scene_and_horizon_tolerance():
    rows = [
        row("sample_a0", "scene_a", 0.0, "lower_gate_open_idle"),
        row("sample_a10", "scene_a", 10.2, "gate_closing", "close_lower_gate"),
        row("sample_a20", "scene_a", 20.0, "all_gates_closed_idle"),
        row("sample_b10", "scene_b", 10.0, "upper_gate_open_idle"),
    ]

    labels = build_action_future_labels(
        rows, horizons_sec=[10, 20], max_time_delta_sec=0.5
    )

    assert labels["sample_a0"]["state_t_plus_10s"]["sample_token"] == "sample_a10"
    assert labels["sample_a0"]["phase_t_plus_10s"] == "gate_closing"
    assert labels["sample_a0"]["state_t_plus_20s"]["sample_token"] == "sample_a20"
    assert labels["sample_a10"]["state_t_plus_20s"] is None
    assert labels["sample_b10"]["state_t_plus_10s"] is None
    assert labels["sample_a0"]["future_phase_after_observed_action"][
        "t_plus_10s"
    ] == "gate_closing"


def test_ship_context_labels_use_visibility_and_inside_berth_only():
    annotations = [
        {
            "token": "ann_ship_inside",
            "sample_token": "sample_a",
            "instance_token": "instance_ship",
            "visibility_token": "2",
            "translation": [45.0, 55.0, 0.0],
        },
        {
            "token": "ann_ship_outside",
            "sample_token": "sample_a",
            "instance_token": "instance_ship2",
            "visibility_token": "4",
            "translation": [45.0, 120.0, 0.0],
        },
        {
            "token": "ann_non_ship",
            "sample_token": "sample_a",
            "instance_token": "instance_gate",
            "visibility_token": "1",
            "translation": [45.0, 55.0, 0.0],
        },
    ]
    berths_by_scene = {
        "scene_a": [
            {
                "slot_id": "berth_slot_01",
                "x_min": 40.0,
                "x_max": 50.0,
                "y_min": 50.0,
                "y_max": 60.0,
            }
        ]
    }

    labels = build_ship_context_labels(
        annotations,
        scene_by_sample={"sample_a": "scene_a"},
        category_by_instance={
            "instance_ship": "Fully_loaded_cargo_ship",
            "instance_ship2": "Unladen_cargo_ship",
            "instance_gate": "Building",
        },
        berths_by_scene=berths_by_scene,
        visibility_levels={"2": "v40-60", "4": "v80-100", "1": "v0-40"},
    )

    assert labels["ann_ship_inside"]["assigned_berth_slot"] == "berth_slot_01"
    assert labels["ann_ship_inside"]["occlusion_state"] == "moderate_occlusion"
    assert labels["ann_ship_inside"]["visibility_level"] == "v40-60"
    assert labels["ann_ship_outside"]["assigned_berth_slot"] is None
    assert labels["ann_ship_outside"]["occlusion_state"] == "no_or_minor_occlusion"
    assert "ann_non_ship" not in labels
