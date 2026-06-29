import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_qwen3vl_lora_adapter import (
    apply_gate_state_guard,
    apply_fusion_reasoning_guard,
    apply_ship_behavior_guard,
    apply_water_level_guard,
    extract_json_text,
    fusion_reasoning_context_from_prompt_messages,
    gate_state_context_from_prompt_messages,
    invalid_json_ids,
    load_prediction_results,
    load_eval_items,
    require_valid_json_results,
    schema_check,
    semantic_check,
    split_prompt_and_reference,
    summarize_results,
    water_level_context_from_prompt_messages,
)


class TestQwen3VLLoRAAdapterEval(unittest.TestCase):
    def test_extract_json_text_from_fenced_output(self):
        self.assertEqual(
            extract_json_text('```json\n{"a": 1}\n```'),
            '{"a": 1}',
        )

    def test_split_prompt_and_reference(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Return JSON."}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"current_state": {}}'}],
            },
        ]

        prompt, reference = split_prompt_and_reference(messages)

        self.assertEqual(len(prompt), 1)
        self.assertEqual(reference, {"current_state": {}})

    def test_load_eval_items_limits_samples_and_prepares_prompt(self):
        item = {
            "id": "sample-1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file:///tmp/a.png"},
                        {"type": "image", "image": "file:///tmp/b.png"},
                        {"type": "text", "text": "abcdef"},
                    ],
                },
                {"role": "assistant", "content": '{"current_state": {}}'},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            items = load_eval_items(
                path,
                max_samples=1,
                max_images_per_sample=1,
                max_text_chars=3,
            )

        user_content = items[0]["prompt_messages"][0]["content"]
        self.assertEqual(items[0]["id"], "sample-1")
        self.assertEqual(items[0]["reference"], {"current_state": {}})
        self.assertEqual(
            [part["image"] for part in user_content if part["type"] == "image"],
            ["/tmp/a.png"],
        )
        self.assertIn("TRUNCATED", user_content[-1]["text"])

    def test_extracts_water_level_context_from_truncated_prompt(self):
        prompt_text = (
            '{"response_contract":{},'
            '"water_level_context":{'
            '"current_water_level":-7.65,'
            '"current_water_state":"idle",'
            '"current_water_level_available":true,'
            '"input_water_level_sequence":[{"frame_index":20,"water_level":-7.65}],'
            '"input_water_level_range":[-7.65,-7.65],'
            '"input_water_level_delta":0.0,'
            '"stable_input_water_level":true'
            '},'
            '"ship_behavior_context":{"latest_ship_instances":['
            "[TRUNCATED FOR QWEN3-VL SMOKE TRAINING]"
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]

        context = water_level_context_from_prompt_messages(messages)

        self.assertEqual(context["current_water_level"], -7.65)
        self.assertTrue(context["stable_input_water_level"])

    def test_water_level_guard_repairs_stable_idle_level_outlier(self):
        prediction = {
            "current_state": {"water_state": "idle", "water_level": 12.5},
            "future_state_10s": {"water_state": "idle", "water_level": 12.7},
            "future_water_level_delta": 0.2,
            "water_surface_dynamics": {
                "target_water_state": "idle",
                "water_level_delta_from_last_input_to_target": 0.2,
            },
        }
        context = {
            "current_water_level": -7.65,
            "current_water_state": "idle",
            "stable_input_water_level": True,
        }

        repaired, report = apply_water_level_guard(prediction, context)

        self.assertTrue(report["applied"])
        self.assertEqual(repaired["current_state"]["water_level"], -7.65)
        self.assertEqual(repaired["future_state_10s"]["water_level"], -7.65)
        self.assertEqual(repaired["future_water_level_delta"], 0.0)
        self.assertEqual(
            repaired["water_surface_dynamics"][
                "water_level_delta_from_last_input_to_target"
            ],
            0.0,
        )
        self.assertEqual(prediction["current_state"]["water_level"], 12.5)

    def test_water_level_guard_repairs_current_water_surface_state(self):
        prediction = {
            "current_state": {"water_state": "idle", "water_level": -5.0},
            "water_surface_dynamics": {
                "current_water_state": "filling",
            },
        }
        context = {
            "current_water_level": -5.0,
            "current_water_state": "idle",
        }

        repaired, report = apply_water_level_guard(prediction, context)

        self.assertTrue(report["applied"])
        self.assertEqual(
            repaired["water_surface_dynamics"]["current_water_state"],
            "idle",
        )
        self.assertEqual(
            prediction["water_surface_dynamics"]["current_water_state"],
            "filling",
        )

    def test_water_level_guard_adds_recognition_current_water_surface_state(self):
        prediction = {
            "current_state": {"water_state": "idle", "water_level": -5.0},
            "water_surface_dynamics": {},
        }
        context = {
            "current_water_level": -5.0,
            "current_water_state": "idle",
        }

        repaired, report = apply_water_level_guard(prediction, context)

        self.assertTrue(report["applied"])
        self.assertEqual(
            repaired["water_surface_dynamics"]["current_water_state"],
            "idle",
        )

    def test_extracts_gate_state_context_from_truncated_prompt(self):
        prompt_text = (
            '{"response_contract":{},'
            '"gate_state_context":{'
            '"current_state":{"upper_gate_state":"closed","lower_gate_state":"open","water_state":"idle"},'
            '"input_gate_state_sequence":[{"frame_index":20,"upper_gate_state":"closed","lower_gate_state":"open","water_state":"idle"}],'
            '"stable_input_gate_state":true,'
            '"stable_input_water_state":true,'
            '"stable_input_lock_state":true'
            '},'
            '"ship_behavior_context":{"latest_ship_instances":['
            "[TRUNCATED FOR QWEN3-VL SMOKE TRAINING]"
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]

        context = gate_state_context_from_prompt_messages(messages)

        self.assertEqual(
            context["current_state"],
            {
                "upper_gate_state": "closed",
                "lower_gate_state": "open",
                "water_state": "idle",
            },
        )
        self.assertTrue(context["stable_input_lock_state"])

    def test_gate_state_guard_repairs_current_state_only(self):
        prediction = {
            "current_state": {
                "upper_gate_state": "closed",
                "lower_gate_state": "closed",
                "water_state": "idle",
            },
            "future_state_10s": {
                "upper_gate_state": "closed",
                "lower_gate_state": "closed",
                "water_state": "idle",
            },
        }
        context = {
            "current_state": {
                "upper_gate_state": "closed",
                "lower_gate_state": "open",
                "water_state": "idle",
            },
            "stable_input_lock_state": True,
        }

        repaired, report = apply_gate_state_guard(prediction, context)

        self.assertTrue(report["applied"])
        self.assertEqual(repaired["current_state"]["lower_gate_state"], "open")
        self.assertEqual(repaired["future_state_10s"]["lower_gate_state"], "closed")
        self.assertEqual(prediction["current_state"]["lower_gate_state"], "closed")

    def test_gate_state_guard_never_repairs_future_gate_state(self):
        prediction = {
            "current_state": {
                "upper_gate_state": "open",
                "lower_gate_state": "closed",
                "water_state": "filling",
            },
            "future_state_10s": {
                "upper_gate_state": "open",
                "lower_gate_state": "closed",
                "water_state": "filling",
            },
        }
        context = {
            "current_state": {
                "upper_gate_state": "closed",
                "lower_gate_state": "open",
                "water_state": "idle",
            },
            "stable_input_lock_state": True,
        }

        repaired, report = apply_gate_state_guard(prediction, context)

        self.assertTrue(report["applied"])
        self.assertEqual(repaired["current_state"]["upper_gate_state"], "closed")
        self.assertEqual(repaired["current_state"]["lower_gate_state"], "open")
        self.assertEqual(repaired["current_state"]["water_state"], "idle")
        self.assertEqual(repaired["future_state_10s"]["upper_gate_state"], "open")
        self.assertEqual(repaired["future_state_10s"]["lower_gate_state"], "closed")

    def test_extracts_fusion_reasoning_context_from_truncated_prompt(self):
        prompt_text = (
            '{"response_contract":{},'
            '"fusion_reasoning_context":{'
            '"use_calibrated_2d_3d_fusion":true,'
            '"calibrated_cameras":["CAM_1","CAM_2"],'
            '"state_cameras_without_geometry":["CAM_3","CAM_8"]'
            '},'
            '"ship_behavior_context":{"latest_ship_instances":['
            "[TRUNCATED FOR QWEN3-VL SMOKE TRAINING]"
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]

        context = fusion_reasoning_context_from_prompt_messages(messages)

        self.assertEqual(
            context,
            {
                "use_calibrated_2d_3d_fusion": True,
                "calibrated_cameras": ["CAM_1", "CAM_2"],
                "state_cameras_without_geometry": ["CAM_3", "CAM_8"],
            },
        )

    def test_fusion_reasoning_guard_repairs_camera_layout_fields(self):
        prediction = {
            "fusion_reasoning": {
                "use_calibrated_2d_3d_fusion": True,
                "calibrated_cameras": ["CAM_3", "CAM_8"],
                "state_cameras_without_geometry": [],
            }
        }
        context = {
            "use_calibrated_2d_3d_fusion": True,
            "calibrated_cameras": ["CAM_1", "CAM_2", "CAM_4"],
            "state_cameras_without_geometry": ["CAM_3", "CAM_8"],
        }

        repaired, report = apply_fusion_reasoning_guard(prediction, context)

        self.assertTrue(report["applied"])
        self.assertEqual(
            repaired["fusion_reasoning"]["calibrated_cameras"],
            ["CAM_1", "CAM_2", "CAM_4"],
        )
        self.assertEqual(
            repaired["fusion_reasoning"]["state_cameras_without_geometry"],
            ["CAM_3", "CAM_8"],
        )
        self.assertEqual(
            prediction["fusion_reasoning"]["calibrated_cameras"],
            ["CAM_3", "CAM_8"],
        )

    def test_ship_behavior_guard_repairs_mooring_evidence_only(self):
        prediction = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Unladen_cargo_ship",
                        "ship_intentions": ["ship_berthed"],
                    }
                ],
                "mooring_or_berthing_confidence_evidence": {
                    "crew_count_2d": 0,
                    "ship_count_2d": 3,
                },
            }
        }
        context = {
            "input_mooring_evidence_counts": {
                "crew_count_2d": 2,
                "mooring_line_count_2d": 1,
                "ship_count_2d": 3,
                "ship_count_3d": 1,
                "mooring_confidence_boost_present": True,
                "weak_rule": "fixed weak rule",
            }
        }

        repaired, report = apply_ship_behavior_guard(prediction, context)

        self.assertTrue(report["applied"])
        self.assertEqual(
            repaired["ship_behavior"]["ship_intentions"],
            prediction["ship_behavior"]["ship_intentions"],
        )
        self.assertEqual(
            repaired["ship_behavior"]["mooring_or_berthing_confidence_evidence"],
            context["input_mooring_evidence_counts"],
        )
        self.assertNotIn(
            "weak_rule",
            prediction["ship_behavior"]["mooring_or_berthing_confidence_evidence"],
        )

    def test_ship_behavior_guard_canonicalizes_ship_category_spelling(self):
        prediction = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_Loaded_cargo_ship",
                        "ship_intentions": ["ship_berthed"],
                    },
                    {
                        "instance_token": "ship_b",
                        "category": "Unladen_cargo_fleet",
                        "ship_intentions": ["ship_entering_lock"],
                    },
                ]
            }
        }

        repaired, report = apply_ship_behavior_guard(prediction, {})

        self.assertTrue(report["applied"])
        self.assertEqual(
            repaired["ship_behavior"]["ship_intentions"][0]["category"],
            "Fully_loaded_cargo_ship",
        )
        self.assertEqual(
            repaired["ship_behavior"]["ship_intentions"][1]["category"],
            "Unladen_cargo_fleet",
        )
        self.assertEqual(
            prediction["ship_behavior"]["ship_intentions"][0]["category"],
            "Fully_Loaded_cargo_ship",
        )

    def test_ship_behavior_guard_repairs_category_from_context_by_token(self):
        prediction = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_ship",
                        "ship_intentions": ["ship_leaving_lock"],
                    },
                ]
            }
        }
        context = {
            "latest_ship_instances": [
                {
                    "instance_token": "ship_a",
                    "category": "Unladen_cargo_fleet",
                    "ship_intentions": ["ship_berthed"],
                }
            ]
        }

        repaired, report = apply_ship_behavior_guard(prediction, context)

        self.assertTrue(report["applied"])
        self.assertEqual(
            repaired["ship_behavior"]["ship_intentions"][0]["category"],
            "Unladen_cargo_fleet",
        )
        self.assertEqual(
            repaired["ship_behavior"]["ship_intentions"][0]["ship_intentions"],
            ["ship_leaving_lock"],
        )

    def test_schema_summary_counts_valid_json_and_missing_keys(self):
        reference = {"current_state": {}, "future_state_10s": {}}
        valid = schema_check({"current_state": {}, "future_state_10s": {}}, reference)
        missing = schema_check({"current_state": {}}, reference)
        invalid = schema_check(None, reference)

        self.assertTrue(valid["valid_json"])
        self.assertEqual(missing["missing_top_level_keys"], ["future_state_10s"])
        self.assertFalse(invalid["valid_json"])
        self.assertEqual(
            summarize_results(
                [
                    {"schema_check": valid},
                    {"schema_check": missing},
                    {"schema_check": invalid},
                ]
            ),
            {
                "num_samples": 3,
                "valid_json": 2,
                "exact_top_level_schema": 1,
                "exact_nested_schema": 1,
                "state_semantic_matches": {},
                "numeric_mae": {},
                "ship_behavior": {
                    "ship_intentions_exact": {"correct": 0, "total": 0},
                    "ship_intention_count_mae": 0.0,
                    "exact_item_match": {
                        "matched": 0,
                        "reference": 0,
                        "prediction": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                    "instance_token_match": {
                        "matched": 0,
                        "reference": 0,
                        "prediction": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                    "instance_intention_match": {
                        "matched": 0,
                        "reference": 0,
                        "prediction": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                    "intention_label_match": {},
                    "mooring_evidence_numeric_mae": {},
                    "mooring_evidence_boolean_matches": {},
                },
                "fusion_reasoning": {
                    "boolean_matches": {},
                    "camera_list_match": {},
                },
            },
        )

    def test_invalid_json_guard_refuses_to_write_bad_artifact(self):
        rows = [
            {"id": "ok", "schema_check": {"valid_json": True}},
            {"id": "bad", "schema_check": {"valid_json": False}},
        ]

        self.assertEqual(invalid_json_ids(rows), ["bad"])
        with self.assertRaises(SystemExit) as raised:
            require_valid_json_results(rows, output_path=Path("predictions.jsonl"))

        message = str(raised.exception)
        self.assertIn("invalid_json=1/2", message)
        self.assertIn("bad", message)
        self.assertIn("refusing to write predictions.jsonl", message)

    def test_schema_check_reports_nested_schema_errors(self):
        reference = {
            "current_state": {
                "water_level": 1.0,
                "lock_state": {"upper_gate_state": "closed"},
            },
            "ship_behavior": {
                "ship_intentions": [{"instance_token": "id"}],
            },
        }
        prediction = {
            "current_state": {
                "water_level": "1.0",
                "gate_state": "closed",
            },
            "ship_behavior": "none",
        }

        checked = schema_check(prediction, reference)

        self.assertEqual(checked["missing_top_level_keys"], [])
        self.assertEqual(checked["extra_top_level_keys"], [])
        self.assertIn("current_state.lock_state", checked["missing_nested_paths"])
        self.assertIn(
            "current_state.lock_state.upper_gate_state",
            checked["missing_nested_paths"],
        )
        self.assertIn("current_state.gate_state", checked["extra_nested_paths"])
        self.assertIn("current_state.water_level", checked["type_mismatch_paths"])
        self.assertIn("ship_behavior", checked["type_mismatch_paths"])

    def test_semantic_check_reports_state_matches_and_numeric_errors(self):
        reference = {
            "current_state": {
                "upper_gate_state": "closed",
                "lower_gate_state": "open",
                "water_state": "idle",
                "water_level": -7.5,
            },
            "future_state_10s": {
                "upper_gate_state": "closed",
                "lower_gate_state": "open",
                "water_state": "filling",
                "water_level": -7.0,
            },
            "future_water_level_delta": 0.5,
            "current_water_level_delta_from_first_selected_frame": 0.1,
            "water_surface_dynamics": {
                "current_water_state": "idle",
                "water_level_delta_from_first_selected_to_current": 0.1,
            },
        }
        prediction = {
            "current_state": {
                "upper_gate_state": "closed",
                "lower_gate_state": "closed",
                "water_state": "idle",
                "water_level": -7.4,
            },
            "future_state_10s": {
                "upper_gate_state": "closed",
                "lower_gate_state": "open",
                "water_state": "idle",
                "water_level": -6.8,
            },
            "future_water_level_delta": 0.2,
            "current_water_level_delta_from_first_selected_frame": 0.3,
            "water_surface_dynamics": {
                "current_water_state": "filling",
                "water_level_delta_from_first_selected_to_current": 0.4,
            },
        }

        checked = semantic_check(prediction, reference)

        self.assertTrue(checked["valid_json"])
        self.assertTrue(checked["state_matches"]["current_state.upper_gate_state"])
        self.assertFalse(checked["state_matches"]["current_state.lower_gate_state"])
        self.assertFalse(checked["state_matches"]["future_state_10s.water_state"])
        self.assertFalse(
            checked["state_matches"]["water_surface_dynamics.current_water_state"]
        )
        self.assertAlmostEqual(
            checked["water_level_absolute_errors"]["current_state.water_level"],
            0.1,
        )
        self.assertAlmostEqual(
            checked["water_level_absolute_errors"]["future_water_level_delta"],
            0.3,
        )
        self.assertAlmostEqual(
            checked["water_level_absolute_errors"][
                "current_water_level_delta_from_first_selected_frame"
            ],
            0.2,
        )
        self.assertAlmostEqual(
            checked["water_level_absolute_errors"][
                "water_surface_dynamics.water_level_delta_from_first_selected_to_current"
            ],
            0.3,
        )

    def test_semantic_check_reports_ship_behavior_metrics(self):
        reference = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_fleet",
                        "ship_intentions": ["ship_entering_lock"],
                    },
                    {
                        "instance_token": "ship_b",
                        "category": "Unladen_cargo_fleet",
                        "ship_intentions": ["ship_berthed"],
                    },
                ],
                "mooring_or_berthing_confidence_evidence": {
                    "crew_count_2d": 3,
                    "mooring_line_count_2d": 1,
                    "ship_count_2d": 5,
                    "ship_count_3d": 2,
                    "mooring_confidence_boost_present": True,
                },
            }
        }
        prediction = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_fleet",
                        "ship_intentions": ["ship_entering_lock"],
                    },
                    {
                        "instance_token": "ship_c",
                        "category": "Unladen_cargo_fleet",
                        "ship_intentions": ["ship_leaving_lock"],
                    },
                ],
                "mooring_or_berthing_confidence_evidence": {
                    "crew_count_2d": 1,
                    "mooring_line_count_2d": 0,
                    "ship_count_2d": 4,
                    "ship_count_3d": 2,
                    "mooring_confidence_boost_present": False,
                },
            }
        }

        checked = semantic_check(prediction, reference)
        intentions = checked["ship_behavior"]["ship_intentions"]
        mooring = checked["ship_behavior"]["mooring_evidence"]

        self.assertTrue(checked["ship_behavior"]["has_reference"])
        self.assertFalse(intentions["exact_items_match"])
        self.assertEqual(intentions["reference_count"], 2)
        self.assertEqual(intentions["prediction_count"], 2)
        self.assertEqual(intentions["matched_exact_items"], 1)
        self.assertEqual(intentions["matched_instance_tokens"], 1)
        self.assertEqual(intentions["matched_instance_intentions"], 1)
        self.assertEqual(
            intentions["reference_intention_label_counts"],
            {"ship_entering_lock": 1, "ship_berthed": 1},
        )
        self.assertEqual(
            intentions["matched_intention_label_counts"],
            {"ship_entering_lock": 1},
        )
        self.assertEqual(mooring["numeric_absolute_errors"]["crew_count_2d"], 2.0)
        self.assertEqual(mooring["numeric_absolute_errors"]["ship_count_3d"], 0.0)
        self.assertFalse(mooring["boolean_matches"]["mooring_confidence_boost_present"])

    def test_semantic_check_treats_load_status_as_same_ship_category(self):
        reference = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Unladen_cargo_ship",
                        "ship_intentions": ["ship_entering_lock"],
                    },
                    {
                        "instance_token": "ship_b",
                        "category": "Unladen_cargo_fleet",
                        "ship_intentions": ["ship_berthed"],
                    },
                ]
            }
        }
        prediction = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_ship",
                        "ship_intentions": ["ship_entering_lock"],
                    },
                    {
                        "instance_token": "ship_b",
                        "category": "Fully_loaded_cargo_fleet",
                        "ship_intentions": ["ship_berthed"],
                    },
                ]
            }
        }

        intentions = semantic_check(prediction, reference)["ship_behavior"][
            "ship_intentions"
        ]

        self.assertTrue(intentions["exact_items_match"])
        self.assertEqual(intentions["matched_exact_items"], 2)

    def test_semantic_check_keeps_cargo_and_container_categories_distinct(self):
        reference = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Unladen_container_ship",
                        "ship_intentions": ["ship_entering_lock"],
                    },
                ]
            }
        }
        prediction = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_ship",
                        "ship_intentions": ["ship_entering_lock"],
                    },
                ]
            }
        }

        intentions = semantic_check(prediction, reference)["ship_behavior"][
            "ship_intentions"
        ]

        self.assertFalse(intentions["exact_items_match"])
        self.assertEqual(intentions["matched_exact_items"], 0)

    def test_semantic_check_keeps_cargo_ship_and_fleet_categories_distinct(self):
        reference = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Unladen_cargo_fleet",
                        "ship_intentions": ["ship_entering_lock"],
                    },
                ]
            }
        }
        prediction = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_ship",
                        "ship_intentions": ["ship_entering_lock"],
                    },
                ]
            }
        }

        intentions = semantic_check(prediction, reference)["ship_behavior"][
            "ship_intentions"
        ]

        self.assertFalse(intentions["exact_items_match"])
        self.assertEqual(intentions["matched_exact_items"], 0)

    def test_semantic_check_canonicalizes_ship_category_spelling_for_exact_items(self):
        reference = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_fleet",
                        "ship_intentions": ["ship_entering_lock"],
                    }
                ]
            }
        }
        prediction = {
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_Loaded_cargo_fleet",
                        "ship_intentions": ["ship_entering_lock"],
                    }
                ]
            }
        }

        checked = semantic_check(prediction, reference)
        intentions = checked["ship_behavior"]["ship_intentions"]

        self.assertTrue(intentions["exact_items_match"])
        self.assertEqual(intentions["matched_exact_items"], 1)
        self.assertEqual(intentions["matched_instance_intentions"], 1)

    def test_semantic_check_reports_fusion_reasoning_metrics(self):
        reference = {
            "fusion_reasoning": {
                "use_calibrated_2d_3d_fusion": True,
                "calibrated_cameras": ["CAM_1", "CAM_2"],
                "state_cameras_without_geometry": ["CAM_3", "CAM_8"],
            }
        }
        prediction = {
            "fusion_reasoning": {
                "use_calibrated_2d_3d_fusion": True,
                "calibrated_cameras": ["CAM_2", "CAM_4"],
                "state_cameras_without_geometry": ["CAM_3", "CAM_8"],
            }
        }

        checked = semantic_check(prediction, reference)
        fusion = checked["fusion_reasoning"]

        self.assertTrue(fusion["has_reference"])
        self.assertTrue(fusion["boolean_matches"]["use_calibrated_2d_3d_fusion"])
        self.assertEqual(
            fusion["camera_list_checks"]["calibrated_cameras"],
            {
                "exact_order_match": False,
                "matched": 1,
                "reference": 2,
                "prediction": 2,
            },
        )
        self.assertTrue(
            fusion["camera_list_checks"]["state_cameras_without_geometry"][
                "exact_order_match"
            ]
        )

    def test_summary_includes_semantic_counts_and_numeric_mae(self):
        reference = {
            "current_state": {"upper_gate_state": "closed", "water_level": -7.5},
            "ship_behavior": {
                "ship_intentions": [
                    {
                        "instance_token": "ship_a",
                        "category": "Fully_loaded_cargo_fleet",
                        "ship_intentions": ["ship_entering_lock"],
                    }
                ],
                "mooring_or_berthing_confidence_evidence": {
                    "crew_count_2d": 2,
                    "mooring_confidence_boost_present": True,
                },
            },
            "fusion_reasoning": {
                "use_calibrated_2d_3d_fusion": True,
                "calibrated_cameras": ["CAM_1", "CAM_2"],
                "state_cameras_without_geometry": ["CAM_3", "CAM_8"],
            },
        }
        prediction = {
            "current_state": {"upper_gate_state": "open", "water_level": -7.0},
            "ship_behavior": {
                "ship_intentions": [],
                "mooring_or_berthing_confidence_evidence": {
                    "crew_count_2d": 0,
                    "mooring_confidence_boost_present": False,
                },
            },
            "fusion_reasoning": {
                "use_calibrated_2d_3d_fusion": False,
                "calibrated_cameras": ["CAM_2"],
                "state_cameras_without_geometry": ["CAM_3", "CAM_8"],
            },
        }
        result = {
            "reference": reference,
            "prediction_json": prediction,
            "schema_check": schema_check(prediction, reference),
            "semantic_check": semantic_check(prediction, reference),
        }

        summary = summarize_results([result])

        self.assertEqual(
            summary["state_semantic_matches"]["current_state.upper_gate_state"],
            {"correct": 0, "total": 1},
        )
        self.assertEqual(summary["numeric_mae"]["current_state.water_level"], 0.5)
        self.assertEqual(
            summary["ship_behavior"]["ship_intentions_exact"],
            {"correct": 0, "total": 1},
        )
        self.assertEqual(summary["ship_behavior"]["ship_intention_count_mae"], 1.0)
        self.assertEqual(
            summary["ship_behavior"]["instance_intention_match"]["recall"],
            0.0,
        )
        self.assertEqual(
            summary["ship_behavior"]["intention_label_match"]["ship_entering_lock"][
                "recall"
            ],
            0.0,
        )
        self.assertEqual(
            summary["ship_behavior"]["mooring_evidence_numeric_mae"]["crew_count_2d"],
            2.0,
        )
        self.assertEqual(
            summary["ship_behavior"]["mooring_evidence_boolean_matches"][
                "mooring_confidence_boost_present"
            ],
            {"correct": 0, "total": 1},
        )
        self.assertEqual(
            summary["fusion_reasoning"]["boolean_matches"][
                "use_calibrated_2d_3d_fusion"
            ],
            {"correct": 0, "total": 1},
        )
        self.assertEqual(
            summary["fusion_reasoning"]["camera_list_match"]["calibrated_cameras"][
                "recall"
            ],
            0.5,
        )
        self.assertEqual(
            summary["fusion_reasoning"]["camera_list_match"][
                "state_cameras_without_geometry"
            ]["exact_order"],
            {"correct": 1, "total": 1},
        )

    def test_load_prediction_results_recomputes_checks(self):
        row = {
            "reference": {"current_state": {"upper_gate_state": "closed"}},
            "prediction_json": {"current_state": {"upper_gate_state": "closed"}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            results = load_prediction_results(path)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["schema_check"]["valid_json"])
        self.assertTrue(
            results[0]["semantic_check"]["state_matches"][
                "current_state.upper_gate_state"
            ]
        )

    def test_load_prediction_results_can_apply_context_guards(self):
        row = {
            "id": "sample-1",
            "reference": {
                "current_state": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "open",
                    "water_state": "idle",
                },
                "future_state_10s": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "open",
                    "water_state": "idle",
                },
                "fusion_reasoning": {
                    "use_calibrated_2d_3d_fusion": True,
                    "calibrated_cameras": ["CAM_1"],
                    "state_cameras_without_geometry": ["CAM_3"],
                }
            },
            "prediction_json": {
                "current_state": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                },
                "future_state_10s": {
                    "upper_gate_state": "closed",
                    "lower_gate_state": "closed",
                    "water_state": "idle",
                },
                "fusion_reasoning": {
                    "use_calibrated_2d_3d_fusion": True,
                    "calibrated_cameras": ["CAM_3"],
                    "state_cameras_without_geometry": [],
                }
            },
        }
        contexts = {
            "sample-1": {
                "gate_state_context": {
                    "current_state": {
                        "upper_gate_state": "closed",
                        "lower_gate_state": "open",
                        "water_state": "idle",
                    },
                    "stable_input_lock_state": True,
                },
                "fusion_reasoning_context": {
                    "use_calibrated_2d_3d_fusion": True,
                    "calibrated_cameras": ["CAM_1"],
                    "state_cameras_without_geometry": ["CAM_3"],
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            results = load_prediction_results(
                path,
                context_by_id=contexts,
                use_gate_state_guard=True,
                use_fusion_reasoning_guard=True,
            )

        self.assertTrue(results[0]["gate_state_guard"]["applied"])
        self.assertTrue(results[0]["fusion_reasoning_guard"]["applied"])
        self.assertEqual(
            results[0]["semantic_check"]["fusion_reasoning"]["camera_list_checks"][
                "calibrated_cameras"
            ]["matched"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
