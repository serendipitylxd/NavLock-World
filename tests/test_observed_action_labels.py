from tools.build_observed_action_labels import build_action_labels


def row(token, ts, upper="closed", lower="closed", water="idle"):
    return {
        "token": token,
        "timestamp": ts,
        "timestamp_str": f"2025_10_30_00_00_{ts:06d}",
        "upper_gate_state": upper,
        "lower_gate_state": lower,
        "lock_water_state": water,
    }


def test_gate_open_episode_labels_start_and_in_progress_frames():
    rows = [
        row("a", 0, upper="closed"),
        row("b", 1_000_000, upper="opening"),
        row("c", 2_000_000, upper="open"),
        row("d", 3_000_000, upper="open"),
    ]

    labels = build_action_labels(rows)

    assert labels["a"]["observed_action"] == "open_upper_gate"
    assert labels["b"]["observed_action"] == "open_upper_gate"
    assert labels["a"]["action_start_time"] == 0
    assert labels["b"]["action_end_time"] == 1_000_000
    assert labels["c"]["observed_action"] == "hold"


def test_water_stop_takes_priority_before_gate_open_boundary():
    rows = [
        row("a", 0, lower="closed", water="emptying"),
        row("b", 1_000_000, lower="opening", water="idle"),
    ]

    labels = build_action_labels(rows)

    assert labels["a"]["observed_action"] == "stop_filling_emptying"
    assert labels["a"]["action_target"] == "water_system"
    assert labels["b"]["observed_action"] == "open_lower_gate"
