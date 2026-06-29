from tools.build_path_clear_labels import (
    build_path_clear_labels,
    entry_exit_sides,
    point_in_path_region,
)


CHAMBER = {"x_min": 39.7, "x_max": 62.7, "y_min": 17.2, "y_max": 307.2}


def test_entry_exit_sides_follow_lockage_direction():
    assert entry_exit_sides("upstream") == ("lower", "upper")
    assert entry_exit_sides("downstream") == ("upper", "lower")
    assert entry_exit_sides("unknown") == (None, None)


def test_point_in_30m_path_region_uses_physical_chamber_bounds():
    assert point_in_path_region(50.0, 35.0, CHAMBER, "lower", 30.0)
    assert point_in_path_region(50.0, 290.0, CHAMBER, "upper", 30.0)
    assert not point_in_path_region(50.0, 48.0, CHAMBER, "lower", 30.0)
    assert not point_in_path_region(50.0, 276.0, CHAMBER, "upper", 30.0)
    assert not point_in_path_region(39.0, 35.0, CHAMBER, "lower", 30.0)


def test_path_clear_excludes_ships_inside_ideal_berth_boxes():
    rows = [
        {"token": "sample_up", "scene_token": "scene_up"},
        {"token": "sample_down", "scene_token": "scene_down"},
    ]
    ships_by_sample = {
        "sample_up": [
            {"instance_token": "ship_lower_blocker", "x": 50.0, "y": 35.0},
            {"instance_token": "ship_berth", "x": 50.0, "y": 42.0},
        ],
        "sample_down": [
            {"instance_token": "ship_upper_blocker", "x": 50.0, "y": 290.0},
        ],
    }
    berths_by_scene = {
        "scene_up": [
            {"x_min": 45.0, "x_max": 55.0, "y_min": 38.0, "y_max": 46.0}
        ],
        "scene_down": [],
    }
    direction_by_scene = {"scene_up": "upstream", "scene_down": "downstream"}

    labels, diagnostics = build_path_clear_labels(
        rows,
        ships_by_sample=ships_by_sample,
        berths_by_scene=berths_by_scene,
        direction_by_scene=direction_by_scene,
        chamber=CHAMBER,
        path_length_m=30.0,
    )

    assert labels["sample_up"] == {
        "entry_path_clear": False,
        "exit_path_clear": True,
    }
    assert diagnostics["sample_up"]["entry_path_blocker_tokens"] == [
        "ship_lower_blocker"
    ]
    assert labels["sample_down"] == {
        "entry_path_clear": False,
        "exit_path_clear": True,
    }
