from tools.build_valid_action_labels import (
    ACTION_SET,
    action_violation_reasons,
    build_valid_action_labels,
)


def base_row(**overrides):
    row = {
        "token": "sample_1",
        "scene_token": "scene_1",
        "upper_gate_state": "closed",
        "lower_gate_state": "closed",
        "lock_water_state": "idle",
        "water_level": 1.0,
        "upstream_water_level": 1.05,
        "downstream_water_level": 1.0,
        "no_ship_in_upper_gate_zone": True,
        "no_ship_in_lower_gate_zone": True,
        "entry_path_clear": True,
        "exit_path_clear": True,
        "chamber_capacity_available": True,
        "all_in_chamber_ships_berthed_or_static": True,
        "no_ship_entering_or_leaving_inside_chamber": True,
        "observed_action": "hold",
    }
    row.update(overrides)
    return row


def reasons(action, row, direction="upstream"):
    return action_violation_reasons(
        action,
        row,
        direction=direction,
        water_tolerance_m=0.20,
    )


def test_open_gate_requires_closed_gate_equal_side_water_and_clear_zone():
    row = base_row()
    assert reasons("open_upper_gate", row) == []

    mismatch = base_row(upstream_water_level=1.35)
    assert "upper_water_level_not_equal" in reasons("open_upper_gate", mismatch)

    already_open = base_row(upper_gate_state="open")
    assert "upper_gate_not_closed" in reasons("open_upper_gate", already_open)

    occupied = base_row(no_ship_in_upper_gate_zone=False)
    assert "ship_in_upper_gate_zone" in reasons("open_upper_gate", occupied)


def test_water_actions_require_idle_start_and_filling_or_emptying_stop():
    filling = base_row(lock_water_state="filling", observed_action="start_filling")

    assert reasons("stop_filling_emptying", filling) == []
    assert reasons("start_filling", filling) == []
    emptying_reasons = reasons("start_emptying", filling)
    assert "start_emptying_not_annotated" in emptying_reasons
    assert "water_state_conflicts_with_action" in emptying_reasons

    idle = base_row(lock_water_state="idle")
    assert reasons("stop_filling_emptying", idle) == [
        "water_state_not_filling_or_emptying"
    ]
    assert "start_filling_not_annotated" in reasons("start_filling", idle)

    annotated_fill = base_row(observed_action="start_filling")
    assert reasons("start_filling", annotated_fill) == []
    assert "start_emptying_not_annotated" in reasons("start_emptying", annotated_fill)

    annotated_empty = base_row(observed_action="start_emptying")
    assert reasons("start_emptying", annotated_empty) == []
    assert "start_filling_not_annotated" in reasons("start_filling", annotated_empty)

    moving_ship = base_row(
        observed_action="start_filling",
        all_in_chamber_ships_berthed_or_static=False,
        no_ship_entering_or_leaving_inside_chamber=False,
    )
    start_reasons = reasons("start_filling", moving_ship)
    assert "not_all_in_chamber_ships_berthed_or_static" in start_reasons
    assert "ship_entering_or_leaving_inside_chamber" in start_reasons


def test_dispatch_enter_exit_follow_lockage_direction_and_path_clear():
    upstream_enter = base_row(lower_gate_state="open")
    assert reasons("dispatch_enter", upstream_enter, direction="upstream") == []
    assert "exit_gate_not_open" in reasons(
        "dispatch_exit", upstream_enter, direction="upstream"
    )

    downstream_exit = base_row(lower_gate_state="open")
    assert reasons("dispatch_exit", downstream_exit, direction="downstream") == []

    blocked = base_row(lower_gate_state="open", entry_path_clear=False)
    assert "entry_path_clear_false" in reasons(
        "dispatch_enter", blocked, direction="upstream"
    )

    full_chamber = base_row(
        lower_gate_state="open",
        chamber_capacity_available=False,
    )
    assert "chamber_capacity_unavailable" in reasons(
        "dispatch_enter", full_chamber, direction="upstream"
    )


def test_build_labels_cover_complete_action_set():
    labels = build_valid_action_labels(
        [base_row()],
        direction_by_scene={"scene_1": "upstream"},
        water_tolerance_m=0.20,
    )
    label = labels["sample_1"]

    assert "hold" in label["valid_actions"]
    assert set(label["valid_actions"]) | set(label["invalid_actions"]) == set(
        ACTION_SET
    )
    assert set(label["valid_actions"]) & set(label["invalid_actions"]) == set()
