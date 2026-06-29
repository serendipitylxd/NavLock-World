import json
import unittest

from tools.convert_vlm_semantic_to_qwen3vl import convert_item


class TestQwen3VLRouteBConverter(unittest.TestCase):
    def test_convert_item_uses_qwen_messages_and_state_first_images(self):
        item = {
            "id": "train:prediction:scene_a",
            "split": "train",
            "scene_token": "scene_a",
            "scene_name": "scene-a",
            "task": "navlock_vlm_semantic_multimodal_temporal_reasoning",
            "instruction": "Return structured JSON.",
            "images": [
                "data/samples/CAM_1/a.jpg",
                "data/samples/CAM_3/a.jpg",
                "data/samples/CAM_8/a.jpg",
            ],
            "input": {
                "temporal_setup": {"input_duration_sec": 50},
                "frames": [
                    {
                        "frame_index": 0,
                        "relative_time_sec": 0.0,
                        "images": {
                            "CAM_1": {
                                "path": "data/samples/CAM_1/a.jpg",
                                "perception_2d_summary": {
                                    "counts_by_class": {
                                        "Crew_member": 1,
                                        "Mooring_line": 1,
                                        "Fully_loaded_cargo_ship": 2,
                                        "Tugboat": 3,
                                        "Unknown_vessel": 4,
                                    }
                                },
                            },
                            "CAM_3": {"path": "data/samples/CAM_3/a.jpg"},
                            "CAM_8": {"path": "data/samples/CAM_8/a.jpg"},
                        },
                        "lidar": {
                            "perception_3d_summary": {
                                "counts_by_class": {"Fully_loaded_cargo_ship": 1}
                            }
                        },
                        "lock_state": {
                            "upper_gate_state": "closed",
                            "lower_gate_state": "closed",
                            "water_state": "idle",
                            "water_level": -3.3,
                        },
                        "ship_instances": [
                            {
                                "instance_token": "ship_1",
                                "category": "Fully_loaded_cargo_ship",
                                "ship_intentions": ["ship_entering_lock"],
                                "translation_xy": [10.0, 20.0],
                                "velocity_xy": [0.1, 0.0],
                                "num_lidar_points": 42,
                            }
                        ],
                    }
                ],
                "current_state_from_last_input_frame": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -3.3,
                },
            },
            "answer": {
                "current_state": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -3.3,
                    "numeric_water_level_available": True,
                },
                "future_state_10s": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "closed",
                    "water_state": "filling",
                    "water_level": -3.2,
                },
                "future_water_level_delta": 0.1,
                "water_surface_dynamics": {
                    "visual_check_required": True,
                    "water_surface_wave_expected": True,
                    "wave_annotation_source": "derived_from_water_state_target_region_rule",
                    "target_wave_camera": "CAM_3",
                    "target_wave_region_id": "upper_gate_left_in_chamber",
                    "target_wave_region_description": "left side of the upper gate",
                    "image_level_waterline_annotation_required": False,
                    "numeric_water_level_available": True,
                    "wave_label_image_verified": False,
                    "reason": "derived weak label",
                    "water_level_delta_from_last_input_to_target": 0.1,
                    "target_water_state": "filling",
                    "scene_has_manual_wave_label": False,
                },
                "ship_behavior": {
                    "ship_intentions": [
                        {
                            "instance_token": "ship_1",
                            "category": "Fully_loaded_cargo_ship",
                            "ship_intentions": ["ship_entering_lock"],
                        }
                    ],
                    "mooring_or_berthing_confidence_evidence": {
                        "crew_count_2d": 1,
                        "mooring_line_count_2d": 0,
                        "ship_count_2d": 2,
                        "ship_count_3d": 1,
                        "mooring_confidence_boost_present": True,
                        "weak_rule": "Crew and mooring lines increase confidence.",
                    },
                },
                "fusion_reasoning": {
                    "use_calibrated_2d_3d_fusion": True,
                }
            },
        }

        converted = convert_item(
            item,
            model="Qwen/Qwen3-VL-4B-Instruct",
            image_policy="state_first",
            max_images=2,
            image_max_pixels=65536,
        )

        self.assertEqual(converted["model"], "Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(converted["messages"][0]["role"], "user")
        self.assertEqual(converted["messages"][1]["role"], "assistant")
        user_content = converted["messages"][0]["content"]
        self.assertEqual(user_content[0]["type"], "image")
        self.assertTrue(user_content[0]["image"].startswith("file://"))
        self.assertIn("/CAM_3/", user_content[0]["image"])
        self.assertIn("/CAM_8/", user_content[1]["image"])
        self.assertEqual(user_content[0]["max_pixels"], 65536)
        self.assertEqual(user_content[-1]["type"], "text")
        prompt_payload = json.loads(user_content[-1]["text"])
        self.assertEqual(
            list(prompt_payload.keys())[:3],
            [
                "response_contract",
                "compact_response_template",
                "water_level_context",
            ],
        )
        self.assertIn("gate_transition_context", prompt_payload)
        self.assertEqual(
            prompt_payload["water_level_context"],
            {
                "current_water_level": -3.3,
                "current_water_state": "idle",
                "input_water_level_range": [-3.3, -3.3],
                "input_water_level_delta": None,
                "stable_input_water_level": True,
            },
        )
        self.assertEqual(
            prompt_payload["gate_state_context"],
            {
                "current_state": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                },
                "input_gate_state_sequence": [
                    {
                        "frame_index": 0,
                        "relative_time_sec": 0.0,
                        "upper_gate_state": "closed",
                        "lower_gate_state": "closed",
                        "water_state": "idle",
                    }
                ],
                "stable_input_gate_state": True,
                "stable_input_water_state": True,
                "stable_input_lock_state": True,
            },
        )
        transition_context = prompt_payload["gate_transition_context"]
        transition_checks = {
            item["gate"]: item
            for item in transition_context["candidate_future_gate_checks"]
        }
        self.assertEqual(
            transition_checks["upper_gate_state"]["confusing_pair"],
            ["closed", "opening"],
        )
        self.assertEqual(
            transition_checks["lower_gate_state"]["confusing_pair"],
            ["closed", "opening"],
        )
        self.assertEqual(
            transition_context["state_camera_mapping"]["upper_gate_state"],
            "CAM_3",
        )
        self.assertFalse(
            transition_context["ship_berthing_status"][
                "all_labeled_ship_instances_berthed"
            ]
        )
        self.assertEqual(transition_context["future_gate_domain_rules"], [])
        self.assertIn(["open", "closing"], transition_context["critical_label_pairs"])
        self.assertIn(["closed", "opening"], transition_context["critical_label_pairs"])
        self.assertEqual(
            prompt_payload["fusion_reasoning_context"],
            {
                "use_calibrated_2d_3d_fusion": True,
                "calibrated_cameras": [
                    "CAM_1",
                    "CAM_2",
                    "CAM_4",
                    "CAM_5",
                    "CAM_6",
                    "CAM_7",
                ],
                "state_cameras_without_geometry": ["CAM_3", "CAM_8"],
            },
        )
        self.assertEqual(
            prompt_payload["compact_input_summary"][
                "current_state_from_last_input_frame"
            ]["water_state"],
            "idle",
        )
        ship_context = prompt_payload["ship_behavior_context"]
        self.assertEqual(
            ship_context["latest_ship_instances"],
            [
                {
                    "instance_token": "ship_1",
                    "category": "Fully_loaded_cargo_ship",
                    "ship_intentions": [],
                    "translation_xy": [10.0, 20.0],
                    "velocity_xy": [0.1, 0.0],
                    "num_lidar_points": 42,
                }
            ],
        )
        self.assertEqual(
            ship_context["input_ship_intention_observation_counts"],
            {},
        )
        self.assertEqual(
            ship_context["input_mooring_evidence_counts"],
            {
                "crew_count_2d": 1,
                "mooring_line_count_2d": 1,
                "ship_count_2d": 2,
                "ship_count_3d": 1,
                "mooring_confidence_boost_present": True,
                "weak_rule": (
                    "Crew_member + Mooring_line + ship detection should increase "
                    "confidence in berthed/moored behavior, but missing mooring "
                    "lines must not rule it out because occlusion is common."
                ),
            },
        )
        self.assertEqual(
            prompt_payload["response_contract"]["required_top_level_keys_in_order"],
            [
                "current_state",
                "future_state_10s",
                "future_water_level_delta",
                "water_surface_dynamics",
                "ship_behavior",
                "fusion_reasoning",
            ],
        )
        self.assertEqual(
            prompt_payload["response_contract"]["forbidden_top_level_keys"],
            ["navlock_task", "output"],
        )
        self.assertEqual(
            prompt_payload["response_schema_details"]["required_nested_schema"][
                "current_state"
            ],
            {
                "upper_gate_state": "string",
                "lower_gate_state": "string",
                "water_state": "string",
                "water_level": "number",
                "numeric_water_level_available": "boolean",
            },
        )
        self.assertIn(
            "ship_behavior.mooring_or_berthing_confidence_evidence.crew_count_2d",
            prompt_payload["response_schema_details"]["required_json_paths"],
        )
        self.assertIn(
            "fusion_reasoning.use_calibrated_2d_3d_fusion",
            prompt_payload["response_schema_details"]["required_json_paths"],
        )
        self.assertIn(
            "ship_behavior.ship_intentions",
            prompt_payload["response_schema_details"]["required_json_paths"],
        )
        self.assertNotIn(
            "ship_behavior.ship_intentions[].instance_token",
            prompt_payload["response_schema_details"]["required_json_paths"],
        )
        self.assertEqual(
            prompt_payload["compact_response_template"]["current_state"]["water_level"],
            "number",
        )
        self.assertEqual(
            prompt_payload["compact_response_template"]["water_surface_dynamics"][
                "wave_label_image_verified"
            ],
            "boolean",
        )
        self.assertEqual(
            prompt_payload["compact_response_template"]["water_surface_dynamics"][
                "wave_annotation_source"
            ],
            "string; use none when no source",
        )
        self.assertEqual(
            prompt_payload["compact_response_template"]["ship_behavior"][
                "mooring_or_berthing_confidence_evidence"
            ]["weak_rule"],
            "string",
        )
        schema_rules = prompt_payload["response_contract"]["schema_critical_rules"]
        self.assertTrue(any("Use JSON null" in rule for rule in schema_rules))
        self.assertTrue(any("Use []" in rule for rule in schema_rules))
        self.assertTrue(
            any("wave_annotation_source is a string" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("current_state.water_level" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("gate_state_context.current_state" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("future_state_10s gate states are predictions" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("open vs closing and closed vs opening" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("all_labeled_ship_instances_berthed" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("opening_to_open completed" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("forced_future_label" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("future_state_10s.upper_gate_state" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("future_state_10s.lower_gate_state" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("usually retain gate_state_context.current_state" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("stable_input_water_level" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("fusion_reasoning_context" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("Weak wave rule: filling" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("Weak wave rule: emptying" in rule for rule in schema_rules)
        )
        self.assertTrue(any("Weak wave rule: idle" in rule for rule in schema_rules))
        self.assertTrue(
            any("ship_behavior.ship_intentions" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any(
                "ship_behavior_context.input_mooring_evidence_counts" in rule
                for rule in schema_rules
            )
        )
        self.assertTrue(
            any("Do not output ship_behavior_context" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("Do not create path-like keys" in rule for rule in schema_rules)
        )
        self.assertTrue(
            any("Keep every required top-level key" in rule for rule in schema_rules)
        )
        self.assertTrue(
            prompt_payload["response_contract"]["must_output_raw_json_object"]
        )
        self.assertTrue(prompt_payload["output_format"]["do_not_wrap_answer"])
        self.assertEqual(
            json.loads(converted["messages"][1]["content"]),
            item["answer"],
        )
        self.assertEqual(converted["metadata"]["selected_num_images"], 2)

    def test_compact_context_first_prompt_omits_full_input_payload(self):
        frames = []
        for index in range(30):
            frames.append(
                {
                    "frame_index": index,
                    "relative_time_sec": float(index),
                    "images": {
                        "CAM_3": {"path": "data/samples/CAM_3/a.jpg"},
                        "CAM_8": {"path": "data/samples/CAM_8/a.jpg"},
                    },
                    "lock_state": {
                        "upper_gate_state": "closed",
                        "lower_gate_state": "closed",
                        "water_state": "idle" if index < 20 else "filling",
                        "water_level": -3.3 + index * 0.01,
                    },
                    "ship_instances": [
                        {
                            "instance_token": "ship_1",
                            "category": "Fully_loaded_cargo_ship",
                            "ship_intentions": ["ship_berthed"],
                        }
                    ],
                }
            )
        item = {
            "id": "val:recognition_frame:scene_a:sample_029",
            "split": "val",
            "scene_token": "scene_a",
            "scene_name": "scene-a",
            "sample_token": "sample_029",
            "current_frame_index": 29,
            "task": "navlock_vlm_semantic_current_multimodal_recognition",
            "instruction": "Return current structured JSON.",
            "images": ["data/samples/CAM_3/a.jpg", "data/samples/CAM_8/a.jpg"],
            "input": {
                "temporal_setup": {"recognition_duration_sec": 30},
                "frames": frames,
            },
            "answer": {
                "current_state": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "closed",
                    "water_state": "filling",
                    "water_level": -3.01,
                },
                "current_water_level_delta_from_first_selected_frame": 0.29,
                "water_surface_dynamics": {"current_water_state": "filling"},
                "ship_behavior": {
                    "ship_intentions": [
                        {
                            "instance_token": "ship_1",
                            "category": "Fully_loaded_cargo_ship",
                            "ship_intentions": ["ship_berthed"],
                        }
                    ],
                    "mooring_or_berthing_confidence_evidence": {},
                },
                "fusion_reasoning": {"use_calibrated_2d_3d_fusion": False},
            },
        }

        standard = convert_item(
            item,
            model="Qwen/Qwen3-VL-4B-Instruct",
            image_policy="state_first",
            max_images=2,
            image_max_pixels=65536,
        )
        compact = convert_item(
            item,
            model="Qwen/Qwen3-VL-4B-Instruct",
            image_policy="state_first",
            max_images=2,
            image_max_pixels=65536,
            prompt_profile="compact_context_first",
        )
        standard_text = standard["messages"][0]["content"][-1]["text"]
        compact_text = compact["messages"][0]["content"][-1]["text"]
        prompt_payload = json.loads(compact_text)

        self.assertLess(len(compact_text), len(standard_text))
        self.assertEqual(
            list(prompt_payload.keys())[:3],
            ["navlock_task", "instruction", "response_contract"],
        )
        self.assertLess(
            list(prompt_payload.keys()).index("ship_behavior_context"),
            list(prompt_payload.keys()).index("water_level_context"),
        )
        self.assertNotIn("input", prompt_payload)
        self.assertLessEqual(
            len(prompt_payload["compact_input_summary"]["input_lock_state_sequence"]),
            12,
        )
        self.assertEqual(compact["metadata"]["prompt_profile"], "compact_context_first")

    def test_convert_item_keeps_recognition_scalar_top_level_template(self):
        item = {
            "id": "val:recognition:scene_b",
            "split": "val",
            "scene_token": "scene_b",
            "scene_name": "scene-b",
            "task": "navlock_vlm_semantic_current_multimodal_recognition",
            "instruction": "Return current structured JSON.",
            "images": ["data/samples/CAM_3/b.jpg"],
            "input": {
                "temporal_setup": {"recognition_duration_sec": 20},
                "frames": [
                    {
                        "frame_index": 0,
                        "relative_time_sec": 0.0,
                        "images": {
                            "CAM_3": {"path": "data/samples/CAM_3/b.jpg"},
                        },
                        "lock_state": {
                            "upper_gate_state": "open",
                            "lower_gate_state": "closed",
                            "water_state": "filling",
                            "water_level": -3.0,
                        },
                    }
                ],
            },
            "answer": {
                "current_state": {
                    "upper_gate_state": "open",
                    "lower_gate_state": "closed",
                    "water_state": "filling",
                    "water_level": -3.0,
                },
                "current_water_level_delta_from_first_selected_frame": 0.0,
                "water_surface_dynamics": {
                    "wave_annotation_source": "none",
                    "current_water_state": "filling",
                },
                "ship_behavior": {
                    "ship_intentions": [],
                    "mooring_or_berthing_confidence_evidence": {},
                },
                "fusion_reasoning": {"use_calibrated_2d_3d_fusion": False},
            },
        }

        converted = convert_item(
            item,
            model="Qwen/Qwen3-VL-4B-Instruct",
            image_policy="state_first",
            max_images=1,
            image_max_pixels=65536,
        )
        prompt_payload = json.loads(converted["messages"][0]["content"][-1]["text"])

        self.assertEqual(
            prompt_payload["response_contract"]["required_top_level_keys_in_order"],
            [
                "current_state",
                "current_water_level_delta_from_first_selected_frame",
                "water_surface_dynamics",
                "ship_behavior",
                "fusion_reasoning",
            ],
        )
        self.assertEqual(
            prompt_payload["compact_response_template"][
                "current_water_level_delta_from_first_selected_frame"
            ],
            "number",
        )
        self.assertEqual(
            prompt_payload["compact_input_summary"][
                "current_state_from_last_input_frame"
            ],
            {
                "upper_gate_state": "open",
                "lower_gate_state": "closed",
                "water_state": "filling",
                "water_level": -3.0,
            },
        )

    def test_convert_item_includes_rendered_lidar_views_after_state_cameras(self):
        item = {
            "id": "val:prediction:scene_lidar",
            "split": "val",
            "scene_token": "scene_lidar",
            "scene_name": "scene-lidar",
            "task": "navlock_vlm_semantic_multimodal_temporal_reasoning",
            "instruction": "Return structured JSON.",
            "images": [
                "data/samples/CAM_3/c.png",
                "data/samples/CAM_8/c.png",
            ],
            "lidar_images": [
                "outputs/vlm_semantic/lidar_views/val/sample_lidar_bev.png",
                "outputs/vlm_semantic/lidar_views/val/sample_lidar_range.png",
            ],
            "input": {
                "temporal_setup": {"input_duration_sec": 50},
                "frames": [
                    {
                        "frame_index": 5,
                        "relative_time_sec": 50.0,
                        "images": {
                            "CAM_3": {"path": "data/samples/CAM_3/c.png"},
                            "CAM_8": {"path": "data/samples/CAM_8/c.png"},
                        },
                        "lidar": {
                            "channel": "LIDAR_TOP",
                            "rendered_views": {
                                "bev": "outputs/vlm_semantic/lidar_views/val/sample_lidar_bev.png",
                                "range_view": "outputs/vlm_semantic/lidar_views/val/sample_lidar_range.png",
                            },
                        },
                        "lock_state": {
                            "upper_gate_state": "closed",
                            "lower_gate_state": "open",
                            "water_state": "emptying",
                            "water_level": -4.0,
                        },
                    }
                ],
                "current_state_from_last_input_frame": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "open",
                    "water_state": "emptying",
                    "water_level": -4.0,
                },
            },
            "answer": {
                "current_state": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "open",
                    "water_state": "emptying",
                    "water_level": -4.0,
                },
                "future_state_10s": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "open",
                    "water_state": "idle",
                    "water_level": -4.1,
                },
                "future_water_level_delta": -0.1,
                "water_surface_dynamics": {"wave_annotation_source": "none"},
                "ship_behavior": {
                    "ship_intentions": [],
                    "mooring_or_berthing_confidence_evidence": {},
                },
                "fusion_reasoning": {"use_calibrated_2d_3d_fusion": True},
            },
        }

        converted = convert_item(
            item,
            model="Qwen/Qwen3-VL-4B-Instruct",
            image_policy="state_first",
            max_images=4,
            max_lidar_images=2,
            image_max_pixels=65536,
        )

        user_content = converted["messages"][0]["content"]
        image_paths = [part["image"] for part in user_content if part["type"] == "image"]
        self.assertIn("/CAM_3/", image_paths[0])
        self.assertIn("/CAM_8/", image_paths[1])
        self.assertTrue(image_paths[2].endswith("_bev.png"))
        self.assertTrue(image_paths[3].endswith("_range.png"))
        prompt_payload = json.loads(user_content[-1]["text"])
        self.assertEqual(
            [item["kind"] for item in prompt_payload["selected_visual_inputs"]],
            ["camera", "camera", "lidar_bev", "lidar_range_view"],
        )
        self.assertEqual(converted["metadata"]["selected_num_lidar_images"], 2)
        self.assertTrue(converted["metadata"]["source_has_lidar_rendered_views"])

    def test_gate_transition_context_holds_open_after_opening_completed(self):
        item = {
            "id": "test:prediction:scene_opening_done",
            "split": "test",
            "scene_token": "scene_opening_done",
            "scene_name": "scene-opening-done",
            "task": "navlock_vlm_semantic_multimodal_temporal_reasoning",
            "instruction": "Return structured JSON.",
            "images": [],
            "input": {
                "temporal_setup": {"input_duration_sec": 50},
                "frames": [
                    {
                        "frame_index": 0,
                        "relative_time_sec": 0.0,
                        "images": {},
                        "lidar": {},
                        "lock_state": {
                            "upper_gate_state": "opening",
                            "lower_gate_state": "closed",
                            "water_state": "idle",
                            "water_level": -4.68,
                        },
                    },
                    {
                        "frame_index": 1,
                        "relative_time_sec": 1.0,
                        "images": {},
                        "lidar": {},
                        "lock_state": {
                            "upper_gate_state": "open",
                            "lower_gate_state": "closed",
                            "water_state": "idle",
                            "water_level": -4.68,
                        },
                    },
                    {
                        "frame_index": 18,
                        "relative_time_sec": 44.16,
                        "images": {},
                        "lidar": {},
                        "lock_state": {
                            "upper_gate_state": "open",
                            "lower_gate_state": "closed",
                            "water_state": "idle",
                            "water_level": -4.68,
                        },
                    },
                ],
                "current_state_from_last_input_frame": {
                    "upper_gate_state": "open",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.68,
                },
            },
            "answer": {
                "current_state": {
                    "upper_gate_state": "open",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.68,
                },
                "future_state_10s": {
                    "upper_gate_state": "open",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                    "water_level": -4.68,
                },
                "future_water_level_delta": 0.0,
                "water_surface_dynamics": {"wave_annotation_source": "none"},
                "ship_behavior": {
                    "ship_intentions": [],
                    "mooring_or_berthing_confidence_evidence": {},
                },
                "fusion_reasoning": {"use_calibrated_2d_3d_fusion": True},
            },
        }

        converted = convert_item(
            item,
            model="Qwen/Qwen3-VL-4B-Instruct",
            image_policy="state_first",
            max_images=0,
            image_max_pixels=65536,
        )

        prompt_payload = json.loads(converted["messages"][0]["content"][-1]["text"])
        context = prompt_payload["gate_transition_context"]

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


if __name__ == "__main__":
    unittest.main()
