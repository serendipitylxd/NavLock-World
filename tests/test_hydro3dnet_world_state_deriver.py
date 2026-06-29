import importlib.util
import unittest
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "derive_world_state_from_hydro3dnet_tracks",
    Path(__file__).resolve().parent.parent / "tools" / "derive_world_state_from_hydro3dnet_tracks.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def berth(x_min, y_min, x_max, y_max):
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "cx": (x_min + x_max) / 2,
        "cy": (y_min + y_max) / 2,
    }


class TestHydro3DNetWorldStateDeriver(unittest.TestCase):
    def test_track_detections_keeps_nearby_detection_on_same_track(self):
        tracked = _MODULE.track_detections(
            [
                [{"x": 10.0, "y": 20.0, "z": 0.0, "category": "Unladen_cargo_ship", "score": 0.9}],
                [{"x": 11.0, "y": 21.0, "z": 0.0, "category": "Unladen_cargo_ship", "score": 0.8}],
            ],
            track_distance_m=5.0,
        )

        self.assertEqual(tracked[0][0]["track_token"], tracked[1][0]["track_token"])

    def test_track_detections_starts_new_track_for_far_detection(self):
        tracked = _MODULE.track_detections(
            [
                [{"x": 10.0, "y": 20.0, "z": 0.0, "category": "Unladen_cargo_ship", "score": 0.9}],
                [{"x": 50.0, "y": 80.0, "z": 0.0, "category": "Unladen_cargo_ship", "score": 0.8}],
            ],
            track_distance_m=5.0,
        )

        self.assertNotEqual(tracked[0][0]["track_token"], tracked[1][0]["track_token"])

    def test_track_detections_does_not_merge_recovery_across_berths(self):
        berths = [
            berth(38.0, 245.0, 48.0, 295.0),
            berth(55.0, 247.0, 63.0, 291.0),
        ]

        tracked = _MODULE.track_detections(
            [
                [
                    {
                        "x": 43.0,
                        "y": 272.0,
                        "z": 0.0,
                        "category": "Unladen_cargo_ship",
                        "score": 0.9,
                    }
                ],
                [
                    {
                        "x": 59.0,
                        "y": 282.0,
                        "z": 0.0,
                        "category": "Unladen_cargo_ship",
                        "score": 0.8,
                        "detection_source": "rtmdet_multicamera_recovery",
                    }
                ],
            ],
            track_distance_m=40.0,
            berths=berths,
        )

        self.assertNotEqual(tracked[0][0]["track_token"], tracked[1][0]["track_token"])

    def test_eval_token_map_matches_nearest_gt_once(self):
        mapping = _MODULE.eval_token_map_from_current_frame(
            [
                {"track_token": "hydro_track_001", "x": 10.0, "y": 20.0},
                {"track_token": "hydro_track_002", "x": 100.0, "y": 100.0},
            ],
            [
                {"instance_token": "gt_ship_1", "x": 11.0, "y": 20.0},
                {"instance_token": "gt_ship_2", "x": 102.0, "y": 101.0},
            ],
            max_distance_m=5.0,
        )

        self.assertEqual(
            mapping,
            {
                "hydro_track_001": "gt_ship_1",
                "hydro_track_002": "gt_ship_2",
            },
        )

    def test_eval_token_map_uses_input_window_nearest_gt(self):
        mapping = _MODULE.eval_token_map_from_input_window(
            [
                [{"track_token": "hydro_track_001", "x": 10.0, "y": 20.0}],
                [{"track_token": "hydro_track_001", "x": 80.0, "y": 80.0}],
            ],
            [
                {
                    "instances_3d": [
                        {
                            "instance_token": "gt_ship_1",
                            "category": "Unladen_cargo_ship",
                            "translation": [11.0, 20.0, 0.0],
                        }
                    ]
                },
                {
                    "instances_3d": [
                        {
                            "instance_token": "gt_ship_1",
                            "category": "Unladen_cargo_ship",
                            "translation": [40.0, 40.0, 0.0],
                        }
                    ]
                },
            ],
            max_distance_m=5.0,
        )

        self.assertEqual(mapping, {"hydro_track_001": "gt_ship_1"})

    def test_eval_token_map_rejects_recovery_outside_gt_berth(self):
        berths = [
            berth(55.0, 247.0, 63.0, 291.0),
        ]

        mapping = _MODULE.eval_token_map_from_input_window(
            [
                [
                    {
                        "track_token": "hydro_track_009",
                        "x": 49.0,
                        "y": 254.0,
                        "detection_source": "rtmdet_multicamera_recovery",
                    }
                ]
            ],
            [
                {
                    "instances_3d": [
                        {
                            "instance_token": "gt_ship_1",
                            "category": "Unladen_cargo_ship",
                            "translation": [59.5, 285.8, 0.0],
                        }
                    ]
                }
            ],
            max_distance_m=40.0,
            berths=berths,
        )

        self.assertEqual(mapping, {})

    def test_eval_token_map_allows_recovery_same_berth_beyond_center_distance(self):
        berths = [
            berth(39.0, 60.0, 63.0, 290.0),
        ]

        mapping = _MODULE.eval_token_map_from_input_window(
            [
                [
                    {
                        "track_token": "hydro_track_001",
                        "x": 52.9,
                        "y": 269.2,
                        "detection_source": "rtmdet_multicamera_recovery",
                    }
                ]
            ],
            [
                {
                    "instances_3d": [
                        {
                            "instance_token": "gt_ship_1",
                            "category": "Fully_loaded_cargo_fleet",
                            "translation": [51.3, 167.9, 0.0],
                        }
                    ]
                }
            ],
            max_distance_m=40.0,
            berths=berths,
        )

        self.assertEqual(mapping, {"hydro_track_001": "gt_ship_1"})

    def test_eval_open_gate_new_ship_tokens_assigns_next_scene_ship_id(self):
        token_map = {"hydro_track_001": "instance_scene_a_ship_001"}

        _MODULE.add_eval_open_gate_new_ship_tokens(
            token_map,
            [
                [
                    {
                        "track_token": "hydro_track_001",
                        "detection_source": "hydro3dnet",
                        "score": 0.9,
                    },
                    {
                        "track_token": "hydro_track_002",
                        "detection_source": "rtmdet_open_gate_recovery",
                        "score": 0.7,
                    },
                ]
            ],
            [
                {
                    "instances_3d": [
                        {
                            "instance_token": "instance_scene_a_ship_001",
                            "category": "Unladen_cargo_ship",
                            "translation": [0.0, 0.0, 0.0],
                        }
                    ]
                }
            ],
            "scene_a",
        )

        self.assertEqual(
            token_map,
            {
                "hydro_track_001": "instance_scene_a_ship_001",
                "hydro_track_002": "instance_scene_a_ship_002",
            },
        )

    def test_detections_for_frame_filters_non_ship_and_low_score(self):
        predictions = {
            "by_token": {
                "sample_a": {
                    "boxes": [[1, 2, 3, 4, 5, 6, 0], [7, 8, 9, 1, 1, 1, 0]],
                    "label_names": ["Unladen_cargo_ship", "Lock_footbridge"],
                    "scores": [0.2, 0.9],
                }
            },
            "by_index": {},
        }

        detections = _MODULE.detections_for_frame(
            {"sample_token": "sample_a"},
            predictions,
            score_threshold=0.15,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["category"], "Unladen_cargo_ship")


if __name__ == "__main__":
    unittest.main()
