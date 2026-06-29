import json
import tempfile
import unittest
from pathlib import Path

from tools.build_vlm_semantic_instruction_data import (
    _frame_input_summary,
    _gate_transition_context,
    _load_wave_labels,
    _mooring_confidence_evidence,
    _water_surface_task,
)


class TestVLMRouteBInstructionData(unittest.TestCase):
    def test_mooring_evidence_excludes_rtmdet_only_vessel_categories_from_ship_count(self):
        evidence = _mooring_confidence_evidence(
            [
                {
                    "images": {
                        "CAM_1": {
                            "perception_2d_summary": {
                                "counts_by_class": {
                                    "Crew_member": 1,
                                    "Mooring_line": 1,
                                    "Fully_loaded_cargo_ship": 2,
                                    "Tugboat": 3,
                                    "Unknown_vessel": 4,
                                }
                            }
                        }
                    },
                    "lidar": {
                        "perception_3d_summary": {
                            "counts_by_class": {"Fully_loaded_cargo_ship": 1}
                        }
                    },
                }
            ]
        )

        self.assertEqual(evidence["ship_count_2d"], 2)
        self.assertEqual(evidence["ship_count_3d"], 1)

    def test_water_surface_task_uses_external_verified_wave_label(self):
        last_input_frame = {
            "sample_token": "sample_prev",
            "lock_state": {"water_state": "idle", "water_level": -7.2},
        }
        target_frame = {
            "sample_token": "sample_target",
            "lock_state": {"water_state": "filling", "water_level": -7.0},
        }
        wave_labels = {
            ("sample_target", "CAM_3"): {
                "camera": "CAM_3",
                "region_id": "upper_gate_left_in_chamber",
                "region_description": "left side of the upper gate",
                "wave_expected": False,
                "label_source": "manual_image_review",
                "image_verified": True,
                "image_level_waterline_annotation_required": False,
            }
        }

        task = _water_surface_task(
            sequence={},
            last_input_frame=last_input_frame,
            target_frame=target_frame,
            wave_labels_by_sample_camera=wave_labels,
        )

        self.assertTrue(task["visual_check_required"])
        self.assertFalse(task["water_surface_wave_expected"])
        self.assertEqual(task["wave_annotation_source"], "manual_image_review")
        self.assertEqual(task["target_wave_camera"], "CAM_3")
        self.assertTrue(task["wave_label_image_verified"])
        self.assertTrue(task["scene_has_manual_wave_label"])
        self.assertAlmostEqual(task["water_level_delta_from_last_input_to_target"], 0.2)
        self.assertEqual(task["target_water_state"], "filling")

    def test_load_wave_labels_prefers_verified_duplicate(self):
        rows = [
            {
                "sample_token": "sample_1",
                "camera": "CAM_8",
                "wave_expected": True,
                "image_verified": False,
                "label_source": "derived",
            },
            {
                "sample_token": "sample_1",
                "camera": "CAM_8",
                "wave_expected": False,
                "image_verified": True,
                "label_source": "manual_image_review",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            labels = _load_wave_labels(path)

        self.assertFalse(labels[("sample_1", "CAM_8")]["wave_expected"])
        self.assertTrue(labels[("sample_1", "CAM_8")]["image_verified"])
        self.assertEqual(
            labels[("sample_1", "CAM_8")]["label_source"],
            "manual_image_review",
        )

    def test_frame_input_summary_includes_compact_ship_instances(self):
        frame = {
            "frame_index": 1,
            "sample_token": "sample_ship",
            "timestamp": 123456,
            "relative_time_sec": 5.0,
            "images": {},
            "lidar": {
                "file_name": "samples/LIDAR_TOP/sample_ship.bin",
                "channel": "LIDAR_TOP",
                "num_point_features": 5,
            },
            "lock_state": {
                "upper_gate_state": "closed",
                "lower_gate_state": "open",
                "water_state": "idle",
                "water_level": -7.5,
            },
            "instances_3d": [
                {
                    "instance_token": "ship_a",
                    "category": "Fully_loaded_cargo_fleet",
                    "translation": [1.23456, 2.34567, -1.0],
                    "velocity": [0.12, 0.0],
                    "num_lidar_points": 100,
                    "ship_intentions": ["ship_berthed"],
                },
                {
                    "instance_token": "bridge_a",
                    "category": "Lock_footbridge",
                    "translation": [3.0, 4.0, 0.0],
                    "velocity": [0.0, 0.0],
                    "num_lidar_points": 50,
                    "ship_intentions": [],
                },
            ],
        }
        perception = {
            "sample_ship": {
                "image_features": {},
                "lidar_3d_features": {},
                "flat_features": {},
            }
        }

        summary = _frame_input_summary(frame, Path("data"), perception)

        self.assertEqual(
            summary["ship_instances"],
            [
                {
                    "instance_token": "ship_a",
                    "category": "Fully_loaded_cargo_fleet",
                    "ship_intentions": ["ship_berthed"],
                    "translation_xy": [1.235, 2.346],
                    "velocity_xy": [0.12, 0.0],
                    "num_lidar_points": 100,
                }
            ],
        )

    def test_gate_transition_context_marks_confusing_future_pairs(self):
        frame_summaries = [
            {
                "frame_index": 0,
                "relative_time_sec": 0.0,
                "lock_state": {
                    "upper_gate_state": "open",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.6,
                },
                "ship_instances": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_ship",
                        "ship_intentions": ["ship_berthed"],
                    }
                ],
            },
            {
                "frame_index": 1,
                "relative_time_sec": 40.0,
                "lock_state": {
                    "upper_gate_state": "open",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.6,
                },
                "ship_instances": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_ship",
                        "ship_intentions": ["ship_berthed"],
                    }
                ],
            },
        ]
        current_state = {
            "upper_gate_state": "open",
            "lower_gate_state": "closed",
            "water_state": "idle",
            "water_level": -4.6,
        }

        context = _gate_transition_context(frame_summaries, current_state)

        self.assertEqual(
            context["state_camera_mapping"],
            {"upper_gate_state": "CAM_3", "lower_gate_state": "CAM_8"},
        )
        self.assertTrue(
            context["ship_berthing_status"]["all_labeled_ship_instances_berthed"]
        )
        self.assertEqual(context["observed_input_gate_transitions"], [])
        checks = {
            item["gate"]: item
            for item in context["candidate_future_gate_checks"]
        }
        self.assertEqual(
            checks["upper_gate_state"]["confusing_pair"],
            ["open", "closing"],
        )
        self.assertEqual(
            checks["lower_gate_state"]["confusing_pair"],
            ["closed", "opening"],
        )
        self.assertIn(["open", "closing"], context["critical_label_pairs"])
        self.assertIn(["closed", "opening"], context["critical_label_pairs"])

    def test_gate_transition_context_forces_closing_after_open_to_closing_when_berthed(self):
        input_frames = [
            {
                "frame_index": 0,
                "relative_time_sec": 0.0,
                "lock_state": {
                    "upper_gate_state": "open",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.6,
                },
                "instances_3d": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_ship",
                        "ship_intentions": ["ship_berthed"],
                    }
                ],
            },
            {
                "frame_index": 1,
                "relative_time_sec": 48.0,
                "lock_state": {
                    "upper_gate_state": "closing",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.6,
                },
                "instances_3d": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_ship",
                        "ship_intentions": ["ship_berthed"],
                    }
                ],
            },
        ]

        context = _gate_transition_context(
            input_frames,
            {
                "upper_gate_state": "closing",
                "lower_gate_state": "closed",
                "water_state": "idle",
                "water_level": -4.6,
            },
        )

        self.assertEqual(
            context["future_gate_domain_rules"],
            [
                {
                    "gate": "upper_gate_state",
                    "forced_future_label": "closing",
                    "condition": (
                        "input already shows open_to_closing and all labeled ships "
                        "are ship_berthed"
                    ),
                }
            ],
        )

    def test_gate_transition_context_holds_open_after_opening_completed_until_berthed(self):
        input_frames = [
            {
                "frame_index": 0,
                "relative_time_sec": 0.0,
                "lock_state": {
                    "upper_gate_state": "opening",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.6,
                },
                "instances_3d": [],
            },
            {
                "frame_index": 1,
                "relative_time_sec": 1.0,
                "lock_state": {
                    "upper_gate_state": "open",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.6,
                },
                "instances_3d": [],
            },
            {
                "frame_index": 2,
                "relative_time_sec": 48.0,
                "lock_state": {
                    "upper_gate_state": "open",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.6,
                },
                "instances_3d": [],
            },
        ]

        context = _gate_transition_context(
            input_frames,
            {
                "upper_gate_state": "open",
                "lower_gate_state": "closed",
                "water_state": "idle",
                "water_level": -4.6,
            },
        )

        self.assertEqual(
            context["observed_input_gate_transitions"],
            [
                {
                    "gate": "upper_gate_state",
                    "from": "opening",
                    "to": "open",
                    "from_frame_index": 0,
                    "to_frame_index": 1,
                    "to_relative_time_sec": 1.0,
                }
            ],
        )
        self.assertFalse(
            context["ship_berthing_status"]["all_labeled_ship_instances_berthed"]
        )
        self.assertEqual(
            context["opening_completed_hold_rules"],
            [
                {
                    "gate": "upper_gate_state",
                    "forced_future_label": "open",
                    "condition": (
                        "input shows opening_to_open completed; short horizon remains "
                        "open unless all labeled ships are ship_berthed"
                    ),
                    "exception": (
                        "If all labeled ships are ship_berthed, the open gate may start "
                        "closing."
                    ),
                }
            ],
        )
        self.assertEqual(
            context["future_gate_domain_rules"],
            context["opening_completed_hold_rules"],
        )

    def test_gate_transition_context_does_not_hold_open_after_opening_completed_when_berthed(self):
        input_frames = [
            {
                "frame_index": 0,
                "relative_time_sec": 0.0,
                "lock_state": {
                    "upper_gate_state": "opening",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.6,
                },
                "instances_3d": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_ship",
                        "ship_intentions": ["ship_berthed"],
                    }
                ],
            },
            {
                "frame_index": 1,
                "relative_time_sec": 1.0,
                "lock_state": {
                    "upper_gate_state": "open",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.6,
                },
                "instances_3d": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_ship",
                        "ship_intentions": ["ship_berthed"],
                    }
                ],
            },
        ]

        context = _gate_transition_context(
            input_frames,
            {
                "upper_gate_state": "open",
                "lower_gate_state": "closed",
                "water_state": "idle",
                "water_level": -4.6,
            },
        )

        self.assertTrue(
            context["ship_berthing_status"]["all_labeled_ship_instances_berthed"]
        )
        self.assertEqual(context["opening_completed_hold_rules"], [])


if __name__ == "__main__":
    unittest.main()
