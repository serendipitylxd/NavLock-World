from tools.build_ship_operation_phase_labels import (
    build_ship_phase_labels,
    classify_ship_operation_phase,
    entry_exit_sides,
    ship_intentions_from_attributes,
)


def row(token="sample_a", scene="scene_up", **overrides):
    item = {
        "token": token,
        "scene_token": scene,
        "timestamp": 0,
        "timestamp_str": "2025_10_30_00_00_00_000000",
        "upper_gate_state": "closed",
        "lower_gate_state": "closed",
        "lock_water_state": "idle",
        "chamber_capacity_available": True,
        "entry_path_clear": True,
    }
    item.update(overrides)
    return item


def ship(token, *intentions):
    return {
        "instance_token": token,
        "category": "Cargo_ship",
        "x": 50.0,
        "y": 100.0,
        "ship_intentions": list(intentions),
    }


def test_entry_exit_sides_match_lockage_direction():
    assert entry_exit_sides("upstream") == ("lower", "upper")
    assert entry_exit_sides("downstream") == ("upper", "lower")
    assert entry_exit_sides("unknown") == (None, None)


def test_ship_intentions_are_read_from_annotation_attributes():
    assert ship_intentions_from_attributes(
        ["ship.entering_lock", "ship.leaving_lock", "ship.berthed", "object.static"]
    ) == [
        "ship_entering_lock",
        "ship_leaving_lock",
        "ship_berthed",
        "object_static",
    ]


def test_ship_phase_uses_intention_annotations():
    assert classify_ship_operation_phase(
        row(), ships=[ship("s1", "ship_entering_lock")], direction="upstream"
    )["ship_operation_phase"] == "ship_entering"

    assert classify_ship_operation_phase(
        row(), ships=[ship("s1", "ship_leaving_lock")], direction="upstream"
    )["ship_operation_phase"] == "ship_leaving"

    assert classify_ship_operation_phase(
        row(), ships=[ship("s1", "ship_berthed"), ship("s2", "object_static")], direction="upstream"
    )["ship_operation_phase"] == "all_ships_berthed"


def test_mixed_entering_and_leaving_is_split_by_open_gate_side():
    entering_phase = classify_ship_operation_phase(
        row(lower_gate_state="open"),
        ships=[ship("entering", "ship_entering_lock"), ship("leaving", "ship_leaving_lock")],
        direction="upstream",
    )
    assert entering_phase["ship_operation_phase"] == "ship_entering"
    assert entering_phase["diagnostics"]["mixed_entering_leaving_resolution"] == (
        "entry_gate_open"
    )

    leaving_phase = classify_ship_operation_phase(
        row(upper_gate_state="open"),
        ships=[ship("entering", "ship_entering_lock"), ship("leaving", "ship_leaving_lock")],
        direction="upstream",
    )
    assert leaving_phase["ship_operation_phase"] == "ship_leaving"
    assert leaving_phase["diagnostics"]["mixed_entering_leaving_resolution"] == (
        "exit_gate_open"
    )


def test_no_ship_phase_uses_whole_lockage_context_not_current_gate_state():
    rows = [
        row("empty_before", timestamp=0, lower_gate_state="closed"),
        row("entering", timestamp=1_000_000),
        row("empty_after", timestamp=2_000_000, lower_gate_state="open"),
    ]
    labels, diagnostics = build_ship_phase_labels(
        rows,
        ships_by_sample={"entering": [ship("s1", "ship_entering_lock")]},
        direction_by_scene={"scene_up": "upstream"},
        lockage_key_by_scene={"scene_up": "lockage_1"},
        max_gap_sec=120.0,
    )

    assert labels["empty_before"]["ship_operation_phase"] == "waiting_for_entry"
    assert diagnostics["empty_before"]["future_entering_in_lockage"] is True
    assert labels["empty_after"]["ship_operation_phase"] == "lock_clear"
    assert diagnostics["empty_after"]["future_entering_in_lockage"] is False


def test_build_ship_phase_labels_adds_episode_times():
    rows = [
        row("a", timestamp=0),
        row("b", timestamp=1_000_000),
        row("c", timestamp=2_000_000),
    ]
    labels, _ = build_ship_phase_labels(
        rows,
        ships_by_sample={
            "a": [ship("s1", "ship_entering_lock")],
            "b": [ship("s1", "ship_entering_lock")],
            "c": [ship("s1", "ship_berthed")],
        },
        direction_by_scene={"scene_up": "upstream"},
        max_gap_sec=120.0,
    )

    assert labels["a"]["ship_operation_phase"] == "ship_entering"
    assert labels["b"]["ship_phase_start_time"] == 0
    assert labels["b"]["ship_phase_end_time"] == 1_000_000
    assert labels["c"]["ship_operation_phase"] == "all_ships_berthed"
