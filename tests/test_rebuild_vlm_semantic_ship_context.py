import json
import unittest

from tools.rebuild_vlm_semantic_ship_context import (
    extract_ship_intentions,
    rebuild_item_ship_context,
    source_items_for_row,
)


def _qwen_row() -> dict:
    payload = {
        "ship_behavior_context": {
            "latest_ship_instances": [
                {
                    "instance_token": "annotation_ship",
                    "category": "Fully_loaded_cargo_ship",
                    "ship_intentions": ["ship_entering_lock"],
                }
            ],
            "input_ship_intention_observation_counts": {"ship_entering_lock": 2},
            "input_mooring_evidence_counts": {"ship_count_3d": 1},
        },
        "gate_transition_context": {
            "observed_input_gate_transitions": [],
            "ship_berthing_status": {
                "num_labeled_ship_instances": 1,
                "num_labeled_berthed_ship_instances": 0,
                "ship_berthing_labels_available": True,
                "all_labeled_ship_instances_berthed": False,
            },
            "candidate_future_gate_checks": [
                {
                    "gate": "upper_gate_state",
                    "current_label": "open",
                    "all_labeled_ship_instances_berthed": False,
                }
            ],
        },
        "compact_input_summary": {
            "gate_transition_context": {
                "observed_input_gate_transitions": [],
                "ship_berthing_status": {},
                "candidate_future_gate_checks": [
                    {
                        "gate": "upper_gate_state",
                        "current_label": "open",
                        "all_labeled_ship_instances_berthed": False,
                    }
                ],
            }
        },
        "input": {
            "frames": [
                {
                    "frame_index": 0,
                    "ship_instances": [
                        {
                            "instance_token": "annotation_ship",
                            "category": "Fully_loaded_cargo_ship",
                            "ship_intentions": ["ship_entering_lock"],
                        }
                    ],
                },
                {
                    "frame_index": 1,
                    "ship_instances": [
                        {
                            "instance_token": "annotation_ship",
                            "category": "Fully_loaded_cargo_ship",
                            "ship_intentions": ["ship_entering_lock"],
                        }
                    ],
                },
            ]
        },
    }
    return {
        "id": "test:prediction:scene_a",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload, separators=(",", ":"))}
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "ship_behavior": {
                            "ship_intentions": [
                                {
                                    "instance_token": "reference_ship",
                                    "category": "Fully_loaded_cargo_ship",
                                    "ship_intentions": ["ship_leaving_lock"],
                                }
                            ]
                        }
                    }
                ),
            },
        ],
        "metadata": {"scene_token": "scene_a"},
    }


class TestRebuildRouteBShipContextFromBaseline(unittest.TestCase):
    def test_extract_ship_intentions_ignores_reference_only_source(self):
        row = {
            "reference": {
                "ship_behavior": {
                    "ship_intentions": [
                        {
                            "instance_token": "annotation_ship",
                            "category": "Fully_loaded_cargo_ship",
                            "ship_intentions": ["ship_entering_lock"],
                        }
                    ]
                }
            }
        }

        self.assertEqual(extract_ship_intentions(row), [])

    def test_rebuilds_prompt_context_without_touching_reference(self):
        source_items = [
            {
                "instance_token": "baseline_ship",
                "category": "Fully_loaded_cargo_fleet",
                "ship_intentions": ["ship_berthed"],
            }
        ]

        rebuilt = rebuild_item_ship_context(
            _qwen_row(),
            source_items,
            source_name="baseline",
            source_found=True,
        )

        payload = json.loads(rebuilt["messages"][0]["content"][0]["text"])
        expected = [
            {
                "instance_token": "baseline_ship",
                "category": "Fully_loaded_cargo_fleet",
                "ship_intentions": ["ship_berthed"],
            }
        ]
        self.assertEqual(
            payload["ship_behavior_context"]["latest_ship_instances"], expected
        )
        self.assertEqual(
            payload["ship_behavior_context"]["input_ship_intention_observation_counts"],
            {"ship_berthed": 2},
        )
        self.assertEqual(
            payload["ship_behavior_context"]["input_mooring_evidence_counts"],
            {"ship_count_3d": 1},
        )
        for frame in payload["input"]["frames"]:
            self.assertEqual(frame["ship_instances"], expected)

        for context in (
            payload["gate_transition_context"],
            payload["compact_input_summary"]["gate_transition_context"],
        ):
            self.assertEqual(
                context["ship_berthing_status"]["num_labeled_ship_instances"], 1
            )
            self.assertEqual(
                context["ship_berthing_status"][
                    "num_labeled_berthed_ship_instances"
                ],
                1,
            )
            self.assertTrue(
                context["ship_berthing_status"][
                    "all_labeled_ship_instances_berthed"
                ]
            )
            self.assertTrue(
                context["candidate_future_gate_checks"][0][
                    "all_labeled_ship_instances_berthed"
                ]
            )

        reference = json.loads(rebuilt["messages"][1]["content"])
        self.assertEqual(
            reference["ship_behavior"]["ship_intentions"][0]["instance_token"],
            "reference_ship",
        )
        self.assertEqual(
            rebuilt["metadata"]["ship_intention_context_source"], "baseline"
        )
        self.assertTrue(
            rebuilt["metadata"]["ship_intention_context_source_found"]
        )

    def test_source_items_prefer_frame_sample_before_scene_fallback(self):
        scene_items = [
            {
                "instance_token": "scene_ship",
                "category": "Fully_loaded_cargo_ship",
                "ship_intentions": ["ship_entering_lock"],
            }
        ]
        frame_items = [
            {
                "instance_token": "frame_ship",
                "category": "Fully_loaded_cargo_fleet",
                "ship_intentions": ["ship_berthed"],
            }
        ]
        source_index = {
            "by_scene": {"scene_a": scene_items},
            "by_scene_sample": {("scene_a", "sample_1"): frame_items},
        }

        items, found = source_items_for_row(
            {
                "id": "val:recognition_frame:scene_a:sample_1",
                "metadata": {"scene_token": "scene_a", "sample_token": "sample_1"},
            },
            source_index,
        )
        self.assertTrue(found)
        self.assertEqual(items, frame_items)

        items, found = source_items_for_row(
            {
                "id": "val:recognition_frame:scene_a:sample_2",
                "metadata": {"scene_token": "scene_a", "sample_token": "sample_2"},
            },
            source_index,
        )
        self.assertTrue(found)
        self.assertEqual(items, scene_items)


if __name__ == "__main__":
    unittest.main()
