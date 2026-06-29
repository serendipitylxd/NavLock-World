from tools.build_operation_phase_labels import build_phase_labels


def row(
    token,
    ts,
    upper="closed",
    lower="closed",
    water="idle",
    action="hold",
):
    return {
        "token": token,
        "timestamp": ts,
        "timestamp_str": f"2025_10_30_00_00_{ts:06d}",
        "upper_gate_state": upper,
        "lower_gate_state": lower,
        "lock_water_state": water,
        "observed_action": action,
    }


def test_operation_phase_prefers_water_phase_over_gate_idle_state():
    rows = [
        row("a", 0, water="filling", action="start_filling"),
        row("b", 1_000_000, water="filling", action="start_filling"),
        row("c", 2_000_000, water="idle"),
    ]

    labels = build_phase_labels(rows)

    assert labels["a"]["operation_phase"] == "filling"
    assert labels["b"]["operation_phase"] == "filling"
    assert labels["a"]["phase_start_time"] == 0
    assert labels["b"]["phase_end_time"] == 1_000_000
    assert labels["c"]["operation_phase"] == "all_gates_closed_idle"


def test_operation_phase_labels_gate_opening_and_open_idle():
    rows = [
        row("a", 0, upper="closed", action="open_upper_gate"),
        row("b", 1_000_000, upper="opening", action="open_upper_gate"),
        row("c", 2_000_000, upper="open"),
    ]

    labels = build_phase_labels(rows)

    assert labels["a"]["operation_phase"] == "gate_opening"
    assert labels["b"]["operation_phase"] == "gate_opening"
    assert labels["c"]["operation_phase"] == "upper_gate_open_idle"
