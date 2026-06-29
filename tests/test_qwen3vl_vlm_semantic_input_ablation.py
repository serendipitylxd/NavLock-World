import json
import unittest

from tools.ablate_qwen3vl_vlm_semantic_input import ablate_item, normalize_ablations


def make_item():
    payload = {
        "response_contract": {
            "schema_critical_rules": [
                "Return only JSON.",
                "water_surface_dynamics.wave_annotation_source is a string.",
                "Weak wave rule: filling uses CAM_3.",
                "Use gate_transition_context.candidate_future_gate_checks.",
                "Copy current_state.water_level from water_level_context.",
                "For ship_behavior.mooring_or_berthing_confidence_evidence, copy ship_behavior_context.input_mooring_evidence_counts exactly.",
            ]
        },
        "compact_response_template": {
            "current_state": {
                "upper_gate_state": "string",
                "lower_gate_state": "string",
                "water_state": "string",
                "water_level": "number",
            },
        },
        "water_level_context": {"current_water_level": 1.0},
        "gate_state_context": {"current_state": {"upper_gate_state": "open"}},
        "gate_transition_context": {"candidate_future_gate_checks": [{"gate": "upper_gate_state"}]},
        "ship_behavior_context": {
            "latest_ship_instances": [{"instance_token": "ship_001"}],
            "input_mooring_evidence_counts": {
                "crew_count_2d": 3,
                "mooring_line_count_2d": 1,
                "ship_count_2d": 2,
            },
        },
        "compact_input_summary": {
            "current_state_from_last_input_frame": {"water_level": 1.0},
            "input_lock_state_sequence": [{"water_level": 1.0}],
            "input_water_level_delta": 0.0,
            "gate_transition_context": {"future_gate_domain_rules": []},
        },
        "input": {
            "current_state_from_last_input_frame": {"water_level": 1.0},
            "frames": [
                {
                    "lock_state": {"upper_gate_state": "open", "water_level": 1.0},
                    "images": {
                        "CAM_3": {
                            "perception_2d_summary": {
                                "counts_by_class": {"Crew_member": 3, "Mooring_line": 1},
                                "score_sums_by_class": {"Crew_member": 2.5, "Mooring_line": 0.7},
                            }
                        }
                    },
                    "ship_instances": [{"instance_token": "ship_001"}],
                    "flat_perception_features": {"camera_num_detections": 4.0},
                }
            ],
            "gate_transition_context": {"candidate_future_gate_checks": []},
        },
        "selected_visual_inputs": [
            {"image_index": 0, "kind": "camera", "channel": "CAM_3"},
            {"image_index": 1, "kind": "camera", "channel": "CAM_8"},
            {"image_index": 2, "kind": "lidar_bev", "channel": "LIDAR_TOP", "view_type": "bev"},
            {"image_index": 3, "kind": "lidar_range_view", "channel": "LIDAR_TOP", "view_type": "range_view"},
        ],
        "instruction": "Check whether filling or emptying may cause waves or surface disturbance in the target water-surface region; use all evidence.",
    }
    return {
        "id": "test:prediction:scene_a",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "file:///data/samples/CAM_3/a.png"},
                    {"type": "image", "image": "file:///data/samples/CAM_8/a.png"},
                    {"type": "image", "image": "file:///outputs/vlm_semantic/lidar_views/a_bev.png"},
                    {"type": "image", "image": "file:///outputs/vlm_semantic/lidar_views/a_range.png"},
                    {"type": "text", "text": json.dumps(payload)},
                ],
            },
            {"role": "assistant", "content": "{}"},
        ],
        "metadata": {"selected_num_images": 4, "selected_num_lidar_images": 2},
    }


def payload_of(item):
    content = item["messages"][0]["content"]
    text = next(part["text"] for part in content if part["type"] == "text")
    return json.loads(text)


class Qwen3VLRouteBInputAblationTest(unittest.TestCase):
    def test_state_cameras_only_filters_images_and_visual_inputs(self):
        item = ablate_item(make_item(), ["state_cameras_only"])
        images = [part for part in item["messages"][0]["content"] if part["type"] == "image"]
        payload = payload_of(item)

        self.assertEqual(len(images), 2)
        self.assertEqual([v["channel"] for v in payload["selected_visual_inputs"]], ["CAM_3", "CAM_8"])
        self.assertEqual(item["metadata"]["selected_num_lidar_images"], 0)

    def test_text_only_removes_all_images(self):
        item = ablate_item(make_item(), ["text_only"])
        images = [part for part in item["messages"][0]["content"] if part["type"] == "image"]
        payload = payload_of(item)

        self.assertEqual(images, [])
        self.assertEqual(payload["selected_visual_inputs"], [])
        self.assertIn("No image parts", payload["image_usage_note"])

    def test_lock_operation_transition_prior_is_removed_without_touching_schema(self):
        payload = payload_of(ablate_item(make_item(), ["no_lock_operation_transition_prior"]))
        dumped = json.dumps(payload)

        self.assertNotIn("gate_transition_context", dumped)
        self.assertIn("water_surface_dynamics.wave_annotation_source", dumped)

    def test_operational_telemetry_hides_inputs_but_keeps_output_contract(self):
        payload = payload_of(ablate_item(make_item(), ["no_operational_telemetry"]))

        self.assertNotIn("water_level_context", payload)
        self.assertNotIn("gate_state_context", payload)
        self.assertNotIn("lock_state", payload["input"]["frames"][0])
        self.assertEqual(
            payload["compact_response_template"]["current_state"]["water_level"],
            "number",
        )

    def test_mooring_evidence_is_scrubbed_but_ship_context_remains(self):
        payload = payload_of(ablate_item(make_item(), ["no_mooring_evidence"]))
        frame = payload["input"]["frames"][0]

        self.assertIn("latest_ship_instances", payload["ship_behavior_context"])
        self.assertNotIn("input_mooring_evidence_counts", payload["ship_behavior_context"])
        self.assertEqual(
            frame["images"]["CAM_3"]["perception_2d_summary"]["counts_by_class"]["Crew_member"],
            0,
        )
        rules = payload["response_contract"]["schema_critical_rules"]
        self.assertFalse(any("mooring" in rule.lower() for rule in rules))

    def test_wave_evidence_rules_are_removed_but_type_rule_stays(self):
        payload = payload_of(ablate_item(make_item(), ["no_wave_evidence"]))
        rules = payload["response_contract"]["schema_critical_rules"]

        self.assertFalse(any("Weak wave rule" in rule for rule in rules))
        self.assertTrue(
            any("water_surface_dynamics.wave_annotation_source" in rule for rule in rules)
        )

    def test_deployable_perception_only_expands_to_non_oracle_inputs(self):
        ablations = normalize_ablations(["deployable_perception_only"])
        self.assertIn("no_ship_behavior_context", ablations)
        self.assertIn("no_lock_operation_transition_prior", ablations)

        payload = payload_of(ablate_item(make_item(), ablations))
        dumped = json.dumps(payload)

        self.assertNotIn("ship_behavior_context", dumped)
        self.assertNotIn("ship_instances", dumped)
        self.assertNotIn("gate_transition_context", dumped)
        self.assertIn("gate_state_context", payload)
        self.assertIn("water_level_context", payload)
        self.assertIn("perception_2d_summary", dumped)
        self.assertIn("flat_perception_features", dumped)
        self.assertIn("Deployable perception-only input", payload["instruction"])


if __name__ == "__main__":
    unittest.main()
