from tools.build_chamber_queue_labels import (
    build_chamber_queue_labels,
    entry_exit_sides,
    ship_intentions_from_attributes,
)


CHAMBER = {"x_min": 39.7, "x_max": 62.7, "y_min": 17.2, "y_max": 307.2}
BERTHS = [
    {
        "slot_id": "berth_slot_01",
        "berth_id": "berth_001",
        "x_min": 40.0,
        "x_max": 52.0,
        "y_min": 30.0,
        "y_max": 90.0,
        "cx": 46.0,
        "cy": 60.0,
    },
    {
        "slot_id": "berth_slot_02",
        "berth_id": "berth_002",
        "x_min": 50.0,
        "x_max": 62.0,
        "y_min": 110.0,
        "y_max": 170.0,
        "cx": 56.0,
        "cy": 140.0,
    },
]


def test_entry_exit_sides_match_lockage_direction():
    assert entry_exit_sides("upstream") == ("lower", "upper")
    assert entry_exit_sides("downstream") == ("upper", "lower")
    assert entry_exit_sides("unknown") == (None, None)


def test_ship_intentions_are_read_from_annotation_attributes():
    assert ship_intentions_from_attributes(
        ["ship.entering_lock", "ship.berthed", "object.static"]
    ) == ["ship_entering_lock", "ship_berthed", "object_static"]


def test_capacity_queue_and_berthed_state_use_labels_not_geometry():
    rows = [{"token": "sample_a", "scene_token": "scene_up"}]
    ships_by_sample = {
        "sample_a": [
            {
                "instance_token": "ship_entering_inside_berth_box",
                "category": "Cargo_ship",
                "x": 46.0,
                "y": 60.0,
                "speed_mps": 0.0,
                "ship_intentions": ["ship_entering_lock"],
            },
            {
                "instance_token": "ship_waiting_lower",
                "category": "Cargo_ship",
                "x": 50.0,
                "y": 10.0,
                "speed_mps": 0.2,
                "ship_intentions": ["ship_entering_lock"],
            },
        ]
    }

    labels, diagnostics = build_chamber_queue_labels(
        rows,
        ships_by_sample=ships_by_sample,
        berths_by_scene={"scene_up": BERTHS},
        direction_by_scene={"scene_up": "upstream"},
        chamber=CHAMBER,
        approach_margin_m=10.0,
        max_parallel_actions=2,
    )

    label = labels["sample_a"]
    assert label["occupied_berth_slots"] == ["berth_slot_01"]
    assert label["available_berth_slots"] == ["berth_slot_02"]
    assert label["chamber_capacity_available"] is True
    assert label["all_in_chamber_ships_berthed_or_static"] is False
    assert label["no_ship_entering_or_leaving_inside_chamber"] is False
    assert label["next_ship_to_enter_weak"]["instance_token"] == "ship_waiting_lower"
    assert label["max_parallel_entries"] == 1
    assert diagnostics["sample_a"]["moving_inside_ship_tokens"] == [
        "ship_entering_inside_berth_box"
    ]


def test_downstream_leave_queue_prefers_lower_exit_nearest_ship():
    rows = [{"token": "sample_b", "scene_token": "scene_down"}]
    ships_by_sample = {
        "sample_b": [
            {
                "instance_token": "ship_far_from_lower",
                "category": "Cargo_ship",
                "x": 50.0,
                "y": 220.0,
                "speed_mps": 0.0,
                "ship_intentions": ["ship_berthed"],
            },
            {
                "instance_token": "ship_near_lower",
                "category": "Cargo_ship",
                "x": 50.0,
                "y": 40.0,
                "speed_mps": 0.0,
                "ship_intentions": ["ship_berthed"],
            },
        ]
    }

    labels, _ = build_chamber_queue_labels(
        rows,
        ships_by_sample=ships_by_sample,
        berths_by_scene={"scene_down": BERTHS},
        direction_by_scene={"scene_down": "downstream"},
        chamber=CHAMBER,
        approach_margin_m=10.0,
        max_parallel_actions=2,
    )

    assert labels["sample_b"]["next_ship_to_leave_weak"]["instance_token"] == (
        "ship_near_lower"
    )
    assert labels["sample_b"]["max_parallel_departures"] == 2
