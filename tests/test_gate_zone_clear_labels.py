from tools.build_gate_zone_clear_labels import build_gate_zone_labels, point_in_gate_zone


CHAMBER = {"x_min": 39.7, "x_max": 62.7, "y_min": 17.2, "y_max": 307.2}


def test_point_in_10m_gate_zone_uses_physical_chamber_bounds():
    assert point_in_gate_zone(50.0, 302.0, CHAMBER, "upper", 10.0)
    assert point_in_gate_zone(50.0, 22.0, CHAMBER, "lower", 10.0)
    assert not point_in_gate_zone(50.0, 296.9, CHAMBER, "upper", 10.0)
    assert not point_in_gate_zone(50.0, 27.5, CHAMBER, "lower", 10.0)
    assert not point_in_gate_zone(39.0, 302.0, CHAMBER, "upper", 10.0)


def test_build_gate_zone_labels_outputs_only_clear_booleans():
    rows = [{"token": "sample_a"}, {"token": "sample_b"}]
    ships_by_sample = {
        "sample_a": [
            {"instance_token": "ship_upper", "x": 50.0, "y": 302.0},
            {"instance_token": "ship_mid", "x": 50.0, "y": 100.0},
        ],
        "sample_b": [{"instance_token": "ship_lower", "x": 50.0, "y": 22.0}],
    }

    labels, diagnostics = build_gate_zone_labels(
        rows,
        ships_by_sample=ships_by_sample,
        chamber=CHAMBER,
        gate_zone_length_m=10.0,
    )

    assert labels["sample_a"] == {
        "no_ship_in_upper_gate_zone": False,
        "no_ship_in_lower_gate_zone": True,
    }
    assert labels["sample_b"] == {
        "no_ship_in_upper_gate_zone": True,
        "no_ship_in_lower_gate_zone": False,
    }
    assert diagnostics["sample_a"]["upper_gate_zone_ship_tokens"] == ["ship_upper"]
