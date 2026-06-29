import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from tools.apply_berth_ship_intention_guard import (
    apply_leaving_phase_queue_guard,
    apply_lockage_phase_consistency_guard,
    apply_single_berth_single_ship_eval_token_alias,
    canonical_rtmdet_ship_category,
    dominant_rtmdet_category,
    filter_future_candidate_ship_intentions,
    filter_to_current_active_ship_intentions,
    lockage_flow_phase,
    prune_to_ideal_berth_count,
)
from tools.build_deployable_fused_baseline import (
    align_future_motion_flow_to_future_occupancy,
    align_input_motion_flow_to_ship_intentions,
    align_lock_occupancy_to_ship_intentions,
    apply_vlm_dynamic_ship_intention_fallback,
    fill_missing_current_motion_from_ship_intentions,
    defer_entering_phase_berthed_items,
    fused_metric_table,
    normalized_eval_splits,
    paths_for_splits,
    rtmdet_static_berth_candidate,
    rtmdet_static_berth_override_allowed,
    select_ship_prior_intentions,
    snap_berthed_motion_stitch_outliers,
    split_path_list,
    validate_vlm_semantic_json_rows,
)


class DeployableFusedBaselineTest(unittest.TestCase):
    def test_defer_entering_phase_berthed_items_requires_stable_dwell(self):
        frames = [
            {"relative_time_sec": 0.0, "lock_state": {"upper_gate_state": "closed", "lower_gate_state": "open"}},
            {"relative_time_sec": 5.0, "lock_state": {"upper_gate_state": "closed", "lower_gate_state": "open"}},
            {"relative_time_sec": 10.0, "lock_state": {"upper_gate_state": "closed", "lower_gate_state": "open"}},
        ]
        tracked_frames = [
            [{"track_token": "t1", "x": 5.0, "y": -20.0}],
            [{"track_token": "t1", "x": 5.0, "y": 2.0}],
            [{"track_token": "t1", "x": 5.0, "y": 4.0}],
        ]
        berths = [{"x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 10.0, "cx": 5.0, "cy": 5.0}]
        items = [{"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]}]

        guarded = defer_entering_phase_berthed_items(
            items,
            tracked_frames,
            {"t1": "ship_001"},
            berths,
            frames,
            "scene_2025_10_30_upstream_01",
            single_ship_only=True,
            min_dwell_frames=3,
            min_dwell_sec=8.0,
            max_dwell_displacement_m=3.0,
        )

        self.assertEqual(guarded[0]["ship_intentions"], ["ship_entering_lock"])

    def test_defer_entering_phase_berthed_items_keeps_stable_dwell(self):
        frames = [
            {"relative_time_sec": 0.0, "lock_state": {"upper_gate_state": "closed", "lower_gate_state": "open"}},
            {"relative_time_sec": 5.0, "lock_state": {"upper_gate_state": "closed", "lower_gate_state": "open"}},
            {"relative_time_sec": 12.0, "lock_state": {"upper_gate_state": "closed", "lower_gate_state": "open"}},
        ]
        tracked_frames = [
            [{"track_token": "t1", "x": 5.0, "y": 5.0}],
            [{"track_token": "t1", "x": 5.3, "y": 5.2}],
            [{"track_token": "t1", "x": 5.4, "y": 5.1}],
        ]
        berths = [{"x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 10.0, "cx": 5.0, "cy": 5.0}]
        items = [{"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]}]

        guarded = defer_entering_phase_berthed_items(
            items,
            tracked_frames,
            {"t1": "ship_001"},
            berths,
            frames,
            "scene_2025_10_30_upstream_01",
            single_ship_only=True,
            min_dwell_frames=3,
            min_dwell_sec=8.0,
            max_dwell_displacement_m=3.0,
        )

        self.assertEqual(guarded[0]["ship_intentions"], ["ship_berthed"])

    def test_defer_entering_phase_berthed_items_skips_multi_ship_by_default(self):
        frames = [
            {"relative_time_sec": 0.0, "lock_state": {"upper_gate_state": "closed", "lower_gate_state": "open"}},
            {"relative_time_sec": 5.0, "lock_state": {"upper_gate_state": "closed", "lower_gate_state": "open"}},
        ]
        tracked_frames = [
            [{"track_token": "t1", "x": 5.0, "y": -20.0}],
            [{"track_token": "t1", "x": 5.0, "y": 2.0}],
        ]
        berths = [{"x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 10.0, "cx": 5.0, "cy": 5.0}]
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_002", "ship_intentions": ["ship_berthed"]},
        ]

        guarded = defer_entering_phase_berthed_items(
            items,
            tracked_frames,
            {"t1": "ship_001"},
            berths,
            frames,
            "scene_2025_10_30_upstream_01",
            single_ship_only=True,
            min_dwell_frames=2,
            min_dwell_sec=0.0,
            max_dwell_displacement_m=999.0,
        )

        self.assertEqual(guarded, items)

    def test_combined_eval_splits_and_comma_paths(self):
        args = SimpleNamespace(split="test", eval_splits="val,test")

        self.assertEqual(normalized_eval_splits(args), ["val", "test"])
        self.assertEqual(
            split_path_list(Path("val.jsonl,test.jsonl")),
            [Path("val.jsonl"), Path("test.jsonl")],
        )
        self.assertEqual(
            paths_for_splits(
                None,
                ["val", "test"],
                lambda split: Path(f"{split}.json"),
                arg_name="--example",
            ),
            [Path("val.json"), Path("test.json")],
        )

    def test_fused_metric_table_combines_semantic_ship_and_world_state_metrics(self):
        route_summary = {
            "state_semantic_matches": {
                "current_state.upper_gate_state": {"correct": 24, "total": 24},
                "current_state.lower_gate_state": {"correct": 24, "total": 24},
                "current_state.water_state": {"correct": 24, "total": 24},
                "future_state_10s.upper_gate_state": {"correct": 23, "total": 24},
                "future_state_10s.lower_gate_state": {"correct": 24, "total": 24},
                "future_state_10s.water_state": {"correct": 24, "total": 24},
                "water_surface_dynamics.target_water_state": {
                    "correct": 24,
                    "total": 24,
                },
            },
            "ship_behavior": {
                "ship_intentions_exact": {"correct": 13, "total": 24},
                "instance_token_match": {"f1": 0.743},
                "instance_intention_match": {"f1": 0.514},
            },
        }
        world_current = {
            "occupied_slot_prf": {"f1": 0.8812},
            "vessel_motion_state_accuracy": 0.5136,
        }
        world_future = {
            "occupied_slot_prf": {"f1": 0.8966},
            "vessel_motion_state_accuracy": 0.6944,
        }

        metrics = fused_metric_table(route_summary, world_current, world_future)

        self.assertEqual(metrics["current_gate_water"], {"correct": 24, "total": 24})
        self.assertEqual(metrics["future_upper_gate"], {"correct": 23, "total": 24})
        self.assertEqual(metrics["future_lower_gate_water"], {"correct": 24, "total": 24})
        self.assertEqual(
            metrics["water_surface_target_state"],
            {"correct": 24, "total": 24},
        )
        self.assertEqual(metrics["current_occupied_f1"], 0.881)
        self.assertEqual(metrics["current_motion_acc"], 0.514)
        self.assertEqual(metrics["future_occupied_f1"], 0.897)
        self.assertEqual(metrics["future_motion_acc"], 0.694)
        self.assertEqual(metrics["ship_intentions_exact"], {"correct": 13, "total": 24})
        self.assertEqual(metrics["ship_token_f1"], 0.743)
        self.assertEqual(metrics["ship_intention_f1"], 0.514)

    def test_fused_metric_table_reports_vlm_ship_metrics_separately_from_deployable_branch(self):
        route_summary = {
            "state_semantic_matches": {},
            "ship_behavior": {
                "ship_intentions_exact": {"correct": 21, "total": 24},
                "instance_token_match": {"f1": 0.889},
                "instance_intention_match": {"f1": 0.812},
            },
        }
        vlm_ship_metrics = {
            "ship_intentions_exact": {"correct": 24, "total": 24},
            "ship_token_f1": 1.0,
            "ship_intention_f1": 1.0,
        }
        deployable_ship_metrics = {
            "ship_intentions_exact": {"correct": 21, "total": 24},
            "ship_token_f1": 0.889,
            "ship_intention_f1": 0.812,
        }

        metrics = fused_metric_table(
            route_summary,
            {},
            {},
            ship_metrics=vlm_ship_metrics,
            deployable_ship_metrics=deployable_ship_metrics,
        )

        self.assertEqual(metrics["ship_intentions_exact"], {"correct": 24, "total": 24})
        self.assertEqual(metrics["ship_token_f1"], 1.0)
        self.assertEqual(metrics["ship_intention_f1"], 1.0)
        self.assertEqual(
            metrics["deployable_ship_intentions_exact"],
            {"correct": 21, "total": 24},
        )
        self.assertEqual(metrics["deployable_ship_token_f1"], 0.889)
        self.assertEqual(metrics["deployable_ship_intention_f1"], 0.812)

    def test_ship_prior_mode_can_keep_vlm_when_vlm_has_more_ships(self):
        vlm_items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_002", "ship_intentions": ["ship_entering_lock"]},
        ]
        derived_items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
        ]

        selected, decision = select_ship_prior_intentions(
            vlm_items,
            derived_items,
            "vlm-count-fallback",
        )

        self.assertEqual(selected, vlm_items)
        self.assertEqual(decision, "keep_vlm_more_ships")

    def test_ship_prior_mode_uses_geometry_by_default_or_when_counts_do_not_improve(self):
        vlm_items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
        ]
        derived_items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_002", "ship_intentions": ["ship_leaving_lock"]},
        ]

        selected, decision = select_ship_prior_intentions(
            vlm_items,
            derived_items,
            "vlm-count-fallback",
        )
        self.assertEqual(selected, derived_items)
        self.assertEqual(decision, "use_deployable_geometry")

        selected, decision = select_ship_prior_intentions(
            vlm_items,
            derived_items,
            "replace",
        )
        self.assertEqual(selected, derived_items)
        self.assertEqual(decision, "use_deployable_geometry")

    def test_vlm_dynamic_ship_intention_fallback_restores_same_token_dynamic_label(self):
        rows = [
            {
                "id": "test:recognition:scene_a",
                "prediction_json_raw": {
                    "ship_behavior": {
                        "ship_intentions": [
                            {
                                "instance_token": "ship_001",
                                "ship_intentions": ["ship_entering_lock"],
                            }
                        ]
                    }
                },
                "prediction_json": {
                    "ship_behavior": {
                        "ship_intentions": [
                            {
                                "instance_token": "ship_001",
                                "ship_intentions": ["ship_berthed"],
                            },
                            {
                                "instance_token": "ship_002",
                                "ship_intentions": ["ship_berthed"],
                            },
                        ]
                    }
                },
                "schema_check": {
                    "valid_json": True,
                    "missing_top_level_keys": [],
                    "extra_top_level_keys": [],
                    "missing_nested_paths": [],
                    "extra_nested_paths": [],
                    "type_mismatch_paths": [],
                },
            }
        ]

        report = apply_vlm_dynamic_ship_intention_fallback(rows)

        self.assertEqual(report["changed_items"], 1)
        self.assertEqual(
            rows[0]["prediction_json"]["ship_behavior"]["ship_intentions"][0][
                "ship_intentions"
            ],
            ["ship_entering_lock"],
        )
        self.assertEqual(
            rows[0]["prediction_json"]["ship_behavior"]["ship_intentions"][1][
                "ship_intentions"
            ],
            ["ship_berthed"],
        )

    def test_vlm_dynamic_ship_intention_fallback_ignores_non_dynamic_vlm_label(self):
        rows = [
            {
                "id": "test:recognition:scene_a",
                "prediction_json_raw": {
                    "ship_behavior": {
                        "ship_intentions": [
                            {
                                "instance_token": "ship_001",
                                "ship_intentions": ["ship_berthed"],
                            }
                        ]
                    }
                },
                "prediction_json": {
                    "ship_behavior": {
                        "ship_intentions": [
                            {
                                "instance_token": "ship_001",
                                "ship_intentions": ["ship_berthed"],
                            }
                        ]
                    }
                },
                "schema_check": {
                    "valid_json": True,
                    "missing_top_level_keys": [],
                    "extra_top_level_keys": [],
                    "missing_nested_paths": [],
                    "extra_nested_paths": [],
                    "type_mismatch_paths": [],
                },
            }
        ]

        report = apply_vlm_dynamic_ship_intention_fallback(rows)

        self.assertEqual(report["changed_items"], 0)
        self.assertEqual(
            rows[0]["prediction_json"]["ship_behavior"]["ship_intentions"][0][
                "ship_intentions"
            ],
            ["ship_berthed"],
        )

    def test_fused_metric_table_adds_current_only_rows_without_future_denominator(self):
        rows = [
            {
                "id": "test:prediction:scene_a",
                "semantic_check": {
                    "state_matches": {
                        "current_state.upper_gate_state": True,
                        "current_state.lower_gate_state": True,
                        "current_state.water_state": True,
                        "future_state_10s.upper_gate_state": True,
                        "future_state_10s.lower_gate_state": True,
                        "future_state_10s.water_state": True,
                        "water_surface_dynamics.target_water_state": True,
                    }
                },
            },
            {
                "id": "test:prediction:scene_b",
                "semantic_check": {
                    "state_matches": {
                        "current_state.upper_gate_state": True,
                        "current_state.lower_gate_state": False,
                        "current_state.water_state": True,
                        "future_state_10s.upper_gate_state": False,
                        "future_state_10s.lower_gate_state": True,
                        "future_state_10s.water_state": True,
                        "water_surface_dynamics.target_water_state": True,
                    }
                },
            },
            {
                "id": "test:recognition:scene_current_only",
                "semantic_check": {
                    "state_matches": {
                        "current_state.upper_gate_state": True,
                        "current_state.lower_gate_state": True,
                        "current_state.water_state": True,
                        "water_surface_dynamics.current_water_state": True,
                    }
                },
            },
        ]

        metrics = fused_metric_table(
            {"state_semantic_matches": {}, "ship_behavior": {}},
            {},
            {},
            rows=rows,
        )

        self.assertEqual(metrics["current_gate_water"], {"correct": 2, "total": 3})
        self.assertEqual(metrics["future_upper_gate"], {"correct": 1, "total": 2})
        self.assertEqual(metrics["future_lower_gate_water"], {"correct": 2, "total": 2})
        self.assertEqual(metrics["water_surface_target_state"], {"correct": 3, "total": 3})

    def test_fused_builder_rejects_invalid_vlm_semantic_json_by_default(self):
        rows = [
            {"id": "valid", "schema_check": {"valid_json": True}},
            {"id": "invalid", "schema_check": {"valid_json": False}},
        ]

        with self.assertRaises(SystemExit) as raised:
            validate_vlm_semantic_json_rows(
                rows,
                allow_invalid=False,
                output_path=Path("fused.jsonl"),
            )

        self.assertIn("invalid", str(raised.exception))
        self.assertIn("refusing to write fused.jsonl", str(raised.exception))

    def test_fused_builder_can_explicitly_allow_invalid_vlm_semantic_json(self):
        validate_vlm_semantic_json_rows(
            [{"id": "invalid", "schema_check": {"valid_json": False}}],
            allow_invalid=True,
            output_path=Path("fused.jsonl"),
        )

    def test_rtmdet_static_berth_candidate_requires_inside_multicamera_static_motion(self):
        args = SimpleNamespace(static_2d_motion_threshold=0.02)
        feature = {
            "end_inside_berth": True,
            "cameras_supported": ["CAM_1", "CAM_2"],
            "rtmdet_2d_motion": {
                "camera_count_with_motion": 2,
                "max_normalized_displacement": 0.01,
            },
        }

        self.assertTrue(rtmdet_static_berth_candidate(feature, args))

        feature["rtmdet_2d_motion"]["max_normalized_displacement"] = 0.03
        self.assertFalse(rtmdet_static_berth_candidate(feature, args))

        feature["rtmdet_2d_motion"]["max_normalized_displacement"] = 0.01
        feature["cameras_supported"] = ["CAM_1"]
        self.assertFalse(rtmdet_static_berth_candidate(feature, args))

    def test_rtmdet_static_berth_override_does_not_replace_leaving(self):
        self.assertTrue(rtmdet_static_berth_override_allowed(["ship_entering_lock"]))
        self.assertFalse(rtmdet_static_berth_override_allowed(["ship_leaving_lock"]))
        self.assertFalse(rtmdet_static_berth_override_allowed(["ship_berthed"]))

    def test_world_state_occupancy_aligns_to_final_ship_intentions(self):
        occupancy = {
            "current": {
                "berth_slots": [
                    {"region_id": "berth_slot_01", "occupied": True, "ship_count": 1, "ship_tokens": ["ship_001"]},
                    {"region_id": "berth_slot_02", "occupied": True, "ship_count": 1, "ship_tokens": ["hydro_track_002"]},
                    {"region_id": "berth_slot_03", "occupied": False, "ship_count": 0, "ship_tokens": []},
                ],
            },
            "future_10s": {
                "berth_slots": [
                    {"region_id": "berth_slot_01", "occupied": True, "ship_count": 1, "ship_tokens": ["ship_001"]},
                    {"region_id": "berth_slot_02", "occupied": True, "ship_count": 1, "ship_tokens": ["hydro_track_002"]},
                    {"region_id": "berth_slot_03", "occupied": False, "ship_count": 0, "ship_tokens": []},
                ],
            },
        }
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_002", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_003", "ship_intentions": ["ship_leaving_lock"]},
            {"instance_token": "ship_004", "ship_intentions": ["ship_entering_lock"]},
        ]
        context = {
            "ship_002": {"latest_berth_index": 2, "open_gate_candidate": False},
            "ship_003": {"latest_berth_index": 1, "open_gate_candidate": False},
            "ship_004": {
                "latest_berth_index": 0,
                "nearest_berth_index": 0,
                "nearest_berth_distance_m": 0.0,
                "open_gate_candidate": True,
            },
        }

        report = align_lock_occupancy_to_ship_intentions(
            occupancy,
            items,
            context,
            [{"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}] * 3,
        )

        current_slots = occupancy["current"]["berth_slots"]
        future_slots = occupancy["future_10s"]["berth_slots"]
        self.assertEqual(report["tokens_removed"], 2)
        self.assertEqual(current_slots[1]["ship_tokens"], ["ship_003"])
        self.assertEqual(current_slots[2]["ship_tokens"], ["ship_002"])
        self.assertFalse(future_slots[1]["occupied"])
        self.assertEqual(future_slots[2]["ship_tokens"], ["ship_002"])

    def test_input_motion_alignment_uses_final_intentions_conservatively(self):
        flow = [
            {
                "instance_token": "ship_001",
                "motion_state": "ship_entering_lock",
                "direction_label": "moving_to_upper_gate",
                "delta_xy": [1.0, 8.0],
                "end_speed_mps": 0.4,
            },
            {
                "instance_token": "ship_002",
                "motion_state": "ship_berthed",
                "direction_label": "static_or_settled",
                "delta_xy": [0.0, -10.0],
                "end_speed_mps": 0.0,
            },
            {
                "instance_token": "ship_003",
                "motion_state": "ship_entering_lock",
                "direction_label": "moving_to_lower_gate",
                "delta_xy": [0.0, -4.0],
                "end_speed_mps": 0.3,
            },
            {
                "instance_token": "instance_scene_ship_004",
                "motion_state": "ship_entering_lock",
                "direction_label": "moving_to_lower_gate",
                "delta_xy": [0.0, -5.0],
                "end_speed_mps": 0.3,
            },
        ]
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_002", "ship_intentions": ["ship_leaving_lock"]},
            {"instance_token": "ship_003", "ship_intentions": ["ship_leaving_lock"]},
        ]

        changed = align_input_motion_flow_to_ship_intentions(flow, items)

        self.assertEqual(changed, 3)
        self.assertEqual(flow[0]["motion_state"], "ship_berthed")
        self.assertEqual(flow[1]["motion_state"], "ship_leaving_lock")
        self.assertEqual(flow[1]["direction_label"], "moving_to_lower_gate")
        self.assertEqual(
            flow[2]["motion_state"],
            "ship_entering_lock",
            "raw moving labels are not overwritten unless they were settled/berthed",
        )
        self.assertEqual(flow[3]["motion_state"], "ship_static")

    def test_current_motion_stitch_fills_missing_final_ship_tokens(self):
        flow = []
        ship_items = [
            {
                "instance_token": "ship_001",
                "category": "cargo",
                "ship_intentions": ["ship_entering_lock"],
            },
            {
                "instance_token": "ship_002",
                "category": "cargo",
                "ship_intentions": ["ship_berthed"],
            },
        ]

        report = fill_missing_current_motion_from_ship_intentions(flow, ship_items)

        self.assertEqual(report["filled_missing_final_ship_items"], 2)
        self.assertEqual(flow[0]["motion_state"], "ship_entering_lock")
        self.assertEqual(flow[0]["direction_label"], "from_ship_intention_fallback")
        self.assertEqual(flow[1]["motion_state"], "ship_berthed")
        self.assertEqual(flow[1]["direction_label"], "static_or_settled")

    def test_current_motion_stitch_snaps_berthed_static_and_outlier_motion(self):
        flow = [
            {
                "instance_token": "ship_static",
                "motion_state": "ship_static",
                "direction_label": "static_or_settled",
                "delta_xy": [0.0, 0.0],
                "end_speed_mps": 0.0,
            },
            {
                "instance_token": "ship_outlier",
                "motion_state": "ship_moving",
                "direction_label": "moving_to_lower_gate",
                "delta_xy": [1.0, -6.0],
                "end_speed_mps": 6.5,
            },
            {
                "instance_token": "ship_slow_vlm",
                "motion_state": "ship_entering_lock",
                "direction_label": "moving_to_upper_gate",
                "delta_xy": [0.0, 2.0],
                "end_speed_mps": 0.1,
            },
            {
                "instance_token": "ship_dynamic",
                "motion_state": "ship_entering_lock",
                "direction_label": "moving_to_upper_gate",
                "delta_xy": [0.0, 2.0],
                "end_speed_mps": 0.1,
            },
        ]
        raw_labels = {
            "ship_slow_vlm": "ship_berthed",
            "ship_dynamic": "ship_berthed",
        }
        final_labels = {
            "ship_static": "ship_berthed",
            "ship_outlier": "ship_berthed",
            "ship_slow_vlm": "ship_berthed",
            "ship_dynamic": "ship_entering_lock",
        }

        report = snap_berthed_motion_stitch_outliers(
            flow,
            raw_labels,
            final_labels,
            SimpleNamespace(
                motion_stitch_vlm_slow_speed_mps=0.2,
                motion_stitch_high_speed_outlier_mps=5.0,
            ),
        )

        self.assertEqual(report["final_berthed_static_items"], 1)
        self.assertEqual(report["final_berthed_high_speed_outlier_items"], 1)
        self.assertEqual(report["vlm_slow_berthed_items"], 1)
        self.assertEqual(flow[0]["motion_state"], "ship_berthed")
        self.assertEqual(flow[1]["motion_state"], "ship_berthed")
        self.assertEqual(flow[2]["motion_state"], "ship_berthed")
        self.assertEqual(
            flow[3]["motion_state"],
            "ship_entering_lock",
            "raw VLM berthed cannot override a final dynamic ship branch",
        )

    def test_future_motion_static_berthed_boundary_follows_future_occupancy(self):
        flow = [
            {
                "instance_token": "ship_001",
                "motion_state": "ship_static",
                "direction_label": "static_or_settled",
                "delta_xy": [0.0, 0.0],
                "end_speed_mps": 0.0,
            },
            {
                "instance_token": "ship_002",
                "motion_state": "ship_berthed",
                "direction_label": "static_or_settled",
                "delta_xy": [1.0, 2.0],
                "end_speed_mps": 0.1,
            },
            {
                "instance_token": "ship_003",
                "motion_state": "ship_leaving_lock",
                "direction_label": "moving_to_upper_gate",
                "delta_xy": [0.0, 8.0],
                "end_speed_mps": 1.2,
            },
        ]
        occupancy = {
            "berth_slots": [
                {
                    "region_id": "berth_slot_01",
                    "occupied": True,
                    "ship_count": 1,
                    "ship_tokens": ["ship_001"],
                }
            ]
        }

        changed = align_future_motion_flow_to_future_occupancy(flow, occupancy)

        self.assertEqual(changed, 2)
        self.assertEqual(flow[0]["motion_state"], "ship_berthed")
        self.assertEqual(flow[1]["motion_state"], "ship_static")
        self.assertEqual(flow[1]["delta_xy"], [0.0, 0.0])
        self.assertEqual(
            flow[2]["motion_state"],
            "ship_leaving_lock",
            "future phase labels are handled by separate transition logic",
        )

    def test_lockage_flow_phase_uses_route_side_and_open_gate(self):
        upstream_upper = [{"lock_state": {"upper_gate_state": "open", "lower_gate_state": "closed"}}]
        upstream_lower = [{"lock_state": {"upper_gate_state": "closed", "lower_gate_state": "open"}}]
        downstream_upper = [{"lock_state": {"upper_gate_state": "open", "lower_gate_state": "closed"}}]

        self.assertEqual(
            lockage_flow_phase("scene_upstream_seg001", upstream_upper),
            "ship_leaving_lock",
        )
        self.assertEqual(
            lockage_flow_phase("scene_upstream_seg001", upstream_lower),
            "ship_entering_lock",
        )
        self.assertEqual(
            lockage_flow_phase("scene_downstream_seg001", downstream_upper),
            "ship_entering_lock",
        )

    def test_rtmdet_category_override_requires_consensus(self):
        counts = Counter(
            {
                "Fully_loaded_cargo_ship": 9,
                "Fully_loaded_container_ship": 3,
            }
        )
        weights = Counter(
            {
                "Fully_loaded_cargo_ship": 6.5,
                "Fully_loaded_container_ship": 2.1,
            }
        )

        self.assertEqual(
            dominant_rtmdet_category(counts, weights),
            "Fully_loaded_cargo_ship",
        )

        counts["Fully_loaded_container_ship"] = 7
        self.assertIsNone(dominant_rtmdet_category(counts, weights))

    def test_rtmdet_unknown_category_is_not_forced_to_cargo(self):
        self.assertIsNone(canonical_rtmdet_ship_category("Unknown_vessel"))
        self.assertIsNone(canonical_rtmdet_ship_category("Tugboat"))
        self.assertEqual(
            canonical_rtmdet_ship_category("Unladen_container_ship"),
            "Unladen_cargo_ship",
        )

    def test_ideal_berth_prune_drops_short_hydro_fragment(self):
        items = [
            {"instance_token": "ship_001"},
            {"instance_token": "ship_002"},
            {"instance_token": "hydro_track_003"},
        ]
        berths = [
            {"x_min": 40, "x_max": 60, "y_min": 0, "y_max": 50, "cx": 50, "cy": 25},
            {"x_min": 40, "x_max": 60, "y_min": 60, "y_max": 110, "cx": 50, "cy": 85},
        ]
        tracked_frames = [
            [
                {"track_token": "t1", "x": 50, "y": 20, "score": 0.5},
                {"track_token": "t2", "x": 50, "y": 80, "score": 0.5},
            ],
            [
                {"track_token": "t1", "x": 50, "y": 22, "score": 0.6},
                {"track_token": "t2", "x": 50, "y": 82, "score": 0.6},
                {"track_token": "hydro_track_003", "x": 55, "y": 90, "score": 0.2},
            ],
        ]

        pruned = prune_to_ideal_berth_count(
            items,
            tracked_frames,
            {"t1": "ship_001", "t2": "ship_002"},
            berths,
        )

        self.assertEqual([item["instance_token"] for item in pruned], ["ship_001", "ship_002"])

    def test_ideal_berth_prune_keeps_open_gate_over_whole_chamber_recovery(self):
        items = [
            {"instance_token": "ship_001"},
            {"instance_token": "ship_002"},
            {"instance_token": "hydro_track_003"},
        ]
        berths = [
            {"x_min": 40, "x_max": 60, "y_min": 0, "y_max": 50, "cx": 50, "cy": 25},
            {"x_min": 40, "x_max": 60, "y_min": 60, "y_max": 110, "cx": 50, "cy": 85},
        ]
        tracked_frames = [
            [
                {"track_token": "t1", "x": 50, "y": 20, "score": 0.5},
                {
                    "track_token": "t2",
                    "x": 50,
                    "y": -10,
                    "score": 0.4,
                    "detection_source": "rtmdet_open_gate_recovery",
                    "support_camera_count": 4,
                },
                {
                    "track_token": "hydro_track_003",
                    "x": 50,
                    "y": 80,
                    "score": 0.9,
                    "detection_source": "rtmdet_multicamera_recovery",
                    "support_camera_count": 6,
                },
            ]
        ]

        pruned = prune_to_ideal_berth_count(
            items,
            tracked_frames,
            {"t1": "ship_001", "t2": "ship_002"},
            berths,
        )

        self.assertEqual([item["instance_token"] for item in pruned], ["ship_001", "ship_002"])

    def test_future_candidate_filter_drops_unmapped_rtmdet_recovery(self):
        items = [
            {"instance_token": "ship_001"},
            {"instance_token": "hydro_track_002"},
            {"instance_token": "ship_003"},
        ]
        tracked_frames = [
            [
                {"track_token": "t1", "x": 50, "y": 20, "score": 0.5},
                {
                    "track_token": "hydro_track_002",
                    "x": 50,
                    "y": 80,
                    "score": 0.9,
                    "detection_source": "rtmdet_multicamera_recovery",
                },
                {
                    "track_token": "t3",
                    "x": 50,
                    "y": 5,
                    "score": 0.6,
                    "detection_source": "rtmdet_open_gate_recovery",
                },
            ]
        ]

        filtered = filter_future_candidate_ship_intentions(
            items,
            tracked_frames,
            {"t1": "ship_001", "t3": "ship_003"},
        )

        self.assertEqual([item["instance_token"] for item in filtered], ["ship_001", "ship_003"])

    def test_current_active_filter_drops_stale_input_window_track(self):
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_002"},
        ]
        tracked_frames = [
            [{"track_token": "t1"}, {"track_token": "t2"}],
            [{"track_token": "t2"}],
        ]

        filtered = filter_to_current_active_ship_intentions(
            items,
            tracked_frames,
            {"t1": "ship_001", "t2": "ship_002"},
        )

        self.assertEqual([item["instance_token"] for item in filtered], ["ship_002"])

    def test_current_active_filter_keeps_stale_leaving_ship(self):
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_leaving_lock"]},
            {"instance_token": "ship_002", "ship_intentions": ["ship_berthed"]},
        ]
        tracked_frames = [
            [{"track_token": "t1"}, {"track_token": "t2"}],
            [],
        ]

        filtered = filter_to_current_active_ship_intentions(
            items,
            tracked_frames,
            {"t1": "ship_001", "t2": "ship_002"},
        )

        self.assertEqual([item["instance_token"] for item in filtered], ["ship_001"])

    def test_current_active_filter_keeps_recent_stable_berthed_occlusion(self):
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_002", "ship_intentions": ["ship_berthed"]},
        ]
        tracked_frames = [
            [{"track_token": "t1", "x": 50.0, "y": 20.0}, {"track_token": "t2", "x": 80.0, "y": 20.0}],
            [{"track_token": "t1", "x": 50.4, "y": 20.3}, {"track_token": "t2", "x": 83.5, "y": 20.0}],
            [{"track_token": "t2", "x": 86.5, "y": 20.0}],
        ]

        filtered = filter_to_current_active_ship_intentions(
            items,
            tracked_frames,
            {"t1": "ship_001", "t2": "ship_002"},
            berths=[{"x_min": 45, "x_max": 55, "y_min": 10, "y_max": 30}],
        )

        self.assertEqual([item["instance_token"] for item in filtered], ["ship_001", "ship_002"])

    def test_current_active_filter_drops_recent_moving_berthed_occlusion(self):
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
        ]
        tracked_frames = [
            [{"track_token": "t1", "x": 50.0, "y": 12.0}],
            [{"track_token": "t1", "x": 50.0, "y": 26.0}],
            [],
        ]

        filtered = filter_to_current_active_ship_intentions(
            items,
            tracked_frames,
            {"t1": "ship_001"},
            berths=[{"x_min": 45, "x_max": 55, "y_min": 10, "y_max": 30}],
        )

        self.assertEqual(filtered, [])

    def test_current_active_filter_keeps_recent_berthed_track_with_current_rtmdet_support(self):
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_002", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_003", "ship_intentions": ["ship_berthed"]},
        ]
        tracked_frames = [
            [
                {
                    "track_token": "t1",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 10.0,
                    "size": [2.0, 2.0, 2.0],
                    "yaw": 0.0,
                },
                {
                    "track_token": "t2",
                    "x": 0.0,
                    "y": 0.0,
                },
                {
                    "track_token": "t3",
                    "x": 0.0,
                    "y": 0.0,
                },
            ],
            [
                {"track_token": "t2", "x": 0.0, "y": 0.0},
                {"track_token": "t3", "x": 0.0, "y": 0.0},
            ],
        ]
        frames = [
            {},
            {
                "images": {
                    "CAM_1": {
                        "is_calibrated": True,
                        "file_name": "samples/CAM_1/current.png",
                        "width": 100,
                        "height": 100,
                        "calibration": {
                            "camera_intrinsic": [
                                [10.0, 0.0, 50.0],
                                [0.0, 10.0, 50.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "rotation": [1.0, 0.0, 0.0, 0.0],
                            "translation": [0.0, 0.0, 0.0],
                        },
                    }
                }
            },
        ]
        rtmdet_by_path = {
            str(Path("data") / "samples/CAM_1/current.png"): [
                {
                    "bbox": [48.8, 48.8, 51.2, 51.2],
                    "score": 0.9,
                    "label_name": "Unladen_cargo_ship",
                }
            ]
        }

        filtered = filter_to_current_active_ship_intentions(
            items,
            tracked_frames,
            {"t1": "ship_001", "t2": "ship_002", "t3": "ship_003"},
            frames=frames,
            berths=[{"x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1}],
            rtmdet_by_path=rtmdet_by_path,
            data_root=Path("data"),
        )

        self.assertEqual(
            [item["instance_token"] for item in filtered],
            ["ship_001", "ship_002", "ship_003"],
        )

    def test_current_active_filter_keeps_current_recovery_track(self):
        items = [
            {"instance_token": "ship_001"},
        ]
        tracked_frames = [
            [],
            [
                {
                    "track_token": "hydro_track_001",
                    "detection_source": "rtmdet_multicamera_recovery",
                }
            ],
        ]

        filtered = filter_to_current_active_ship_intentions(
            items,
            tracked_frames,
            {"hydro_track_001": "ship_001"},
        )

        self.assertEqual([item["instance_token"] for item in filtered], ["ship_001"])

    def test_single_berth_single_ship_alias_maps_remaining_track_token(self):
        items = [
            {
                "instance_token": "hydro_track_002",
                "category": "Unladen_cargo_ship",
                "ship_intentions": ["ship_entering_lock"],
            }
        ]
        frames = [
            {
                "instances_3d": [
                    {
                        "instance_token": "instance_scene_a_ship_001",
                        "category": "Unladen_cargo_fleet",
                    }
                ]
            }
        ]
        berths = [{"x_min": 40, "x_max": 60, "y_min": 0, "y_max": 100}]

        aliased = apply_single_berth_single_ship_eval_token_alias(
            items,
            frames,
            berths,
            eval_token_map=True,
        )

        self.assertEqual(aliased[0]["instance_token"], "instance_scene_a_ship_001")
        self.assertEqual(aliased[0]["category"], "Unladen_cargo_ship")
        self.assertEqual(aliased[0]["ship_intentions"], ["ship_entering_lock"])

    def test_single_berth_alias_is_eval_only(self):
        items = [{"instance_token": "hydro_track_002"}]
        frames = [
            {
                "instances_3d": [
                    {
                        "instance_token": "instance_scene_a_ship_001",
                        "category": "Unladen_cargo_fleet",
                    }
                ]
            }
        ]
        berths = [{"x_min": 40, "x_max": 60, "y_min": 0, "y_max": 100}]

        aliased = apply_single_berth_single_ship_eval_token_alias(
            items,
            frames,
            berths,
            eval_token_map=False,
        )

        self.assertEqual(aliased, items)

    def test_leaving_phase_guard_marks_front_queue_berths(self):
        berths = [
            {"x_min": 40, "x_max": 60, "y_min": 0, "y_max": 50, "cx": 50, "cy": 25},
            {"x_min": 40, "x_max": 60, "y_min": 60, "y_max": 110, "cx": 50, "cy": 85},
            {"x_min": 40, "x_max": 60, "y_min": 120, "y_max": 170, "cx": 50, "cy": 145},
            {"x_min": 40, "x_max": 60, "y_min": 180, "y_max": 230, "cx": 50, "cy": 205},
        ]
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_002", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_003", "ship_intentions": ["ship_entering_lock"]},
        ]
        tracked_frames = [
            [
                {"track_token": "t1", "x": 50, "y": 25},
                {"track_token": "t2", "x": 50, "y": 85},
                {"track_token": "t3", "x": 50, "y": 145},
            ]
        ]
        frames = [{"lock_state": {"upper_gate_state": "open", "lower_gate_state": "closed"}}]

        guarded = apply_leaving_phase_queue_guard(
            items,
            tracked_frames,
            {"t1": "ship_001", "t2": "ship_002", "t3": "ship_003"},
            berths,
            frames,
            "scene_upstream_seg001",
        )

        labels = {item["instance_token"]: item["ship_intentions"] for item in guarded}
        self.assertEqual(labels["ship_003"], ["ship_leaving_lock"])
        self.assertEqual(labels["ship_002"], ["ship_leaving_lock"])
        self.assertEqual(labels["ship_001"], ["ship_berthed"])

    def test_leaving_phase_guard_keeps_second_berth_when_open_end_occupied(self):
        berths = [
            {"x_min": 40, "x_max": 60, "y_min": 0, "y_max": 50, "cx": 50, "cy": 25},
            {"x_min": 40, "x_max": 60, "y_min": 60, "y_max": 110, "cx": 50, "cy": 85},
        ]
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_entering_lock"]},
            {"instance_token": "ship_002", "ship_intentions": ["ship_berthed"]},
        ]
        tracked_frames = [
            [
                {"track_token": "t1", "x": 50, "y": 85},
                {"track_token": "t2", "x": 50, "y": 25},
            ]
        ]
        frames = [{"lock_state": {"upper_gate_state": "open", "lower_gate_state": "closed"}}]

        guarded = apply_leaving_phase_queue_guard(
            items,
            tracked_frames,
            {"t1": "ship_001", "t2": "ship_002"},
            berths,
            frames,
            "scene_upstream_seg001",
        )

        labels = {item["instance_token"]: item["ship_intentions"] for item in guarded}
        self.assertEqual(labels["ship_001"], ["ship_leaving_lock"])
        self.assertEqual(labels["ship_002"], ["ship_berthed"])

    def test_leaving_phase_guard_keeps_stationary_berthed_front_berth(self):
        berths = [
            {"x_min": 40, "x_max": 60, "y_min": 0, "y_max": 50, "cx": 50, "cy": 25},
        ]
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
        ]
        tracked_frames = [
            [{"track_token": "t1", "x": 50.0, "y": 25.0}],
            [{"track_token": "t1", "x": 50.5, "y": 25.5}],
        ]
        frames = [{"lock_state": {"upper_gate_state": "closed", "lower_gate_state": "open"}}]

        guarded = apply_leaving_phase_queue_guard(
            items,
            tracked_frames,
            {"t1": "ship_001"},
            berths,
            frames,
            "scene_downstream_seg001",
        )

        self.assertEqual(guarded[0]["ship_intentions"], ["ship_berthed"])

    def test_phase_consistency_guard_flips_opposite_moving_label(self):
        berths = [
            {"x_min": 40, "x_max": 60, "y_min": 0, "y_max": 50, "cx": 50, "cy": 25},
        ]
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_leaving_lock"]},
        ]
        tracked_frames = [
            [{"track_token": "t1", "x": 50.0, "y": 10.0}],
            [{"track_token": "t1", "x": 50.0, "y": 40.0}],
        ]
        frames = [{"lock_state": {"upper_gate_state": "open", "lower_gate_state": "closed"}}]

        guarded = apply_lockage_phase_consistency_guard(
            items,
            tracked_frames,
            {"t1": "ship_001"},
            berths,
            frames,
            "scene_downstream_seg001",
        )

        self.assertEqual(guarded[0]["ship_intentions"], ["ship_entering_lock"])

    def test_phase_consistency_guard_marks_small_in_berth_outlier_with_berthed_peers_as_berthed(self):
        berths = [
            {"x_min": 40, "x_max": 60, "y_min": 0, "y_max": 50, "cx": 50, "cy": 25},
            {"x_min": 40, "x_max": 60, "y_min": 60, "y_max": 110, "cx": 50, "cy": 85},
            {"x_min": 40, "x_max": 60, "y_min": 120, "y_max": 170, "cx": 50, "cy": 145},
        ]
        items = [
            {"instance_token": "ship_001", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_002", "ship_intentions": ["ship_berthed"]},
            {"instance_token": "ship_003", "ship_intentions": ["ship_leaving_lock"]},
        ]
        tracked_frames = [
            [
                {"track_token": "t1", "x": 50.0, "y": 25.0},
                {"track_token": "t2", "x": 50.0, "y": 85.0},
                {"track_token": "t3", "x": 50.0, "y": 145.0},
            ],
            [
                {"track_token": "t1", "x": 50.0, "y": 25.0},
                {"track_token": "t2", "x": 50.0, "y": 85.0},
                {"track_token": "t3", "x": 50.0, "y": 137.0},
            ],
        ]
        frames = [{"lock_state": {"upper_gate_state": "open", "lower_gate_state": "closed"}}]

        guarded = apply_lockage_phase_consistency_guard(
            items,
            tracked_frames,
            {"t1": "ship_001", "t2": "ship_002", "t3": "ship_003"},
            berths,
            frames,
            "scene_downstream_seg001",
        )

        self.assertEqual(guarded[2]["ship_intentions"], ["ship_berthed"])


if __name__ == "__main__":
    unittest.main()
