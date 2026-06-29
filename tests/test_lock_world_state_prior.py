import importlib.util
import unittest
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "apply_lock_world_state_prior",
    Path(__file__).resolve().parent.parent / "tools" / "apply_lock_world_state_prior.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class TestApplyLockWorldStatePrior(unittest.TestCase):
    def test_replace_injects_world_state_without_recomputing_legacy_schema(self):
        rows = [
            {
                "id": "test:prediction:scene_a",
                "prediction_json": {"current_state": {"upper_gate_state": "open"}},
                "reference": {"current_state": {"upper_gate_state": "open"}},
                "schema_check": {"stale": True},
            }
        ]
        states = {
            "scene_a": {
                "scene_token": "scene_a",
                "lock_occupancy": {"current": {"num_ships": 1}},
                "vessel_motion_flow": {"input_window": []},
            }
        }

        report = _MODULE.apply_lock_world_state_prior(rows, states)

        pred = rows[0]["prediction_json"]
        self.assertEqual(pred["lock_occupancy"]["current"]["num_ships"], 1)
        self.assertEqual(pred["vessel_motion_flow"]["input_window"], [])
        self.assertEqual(rows[0]["schema_check"], {"stale": True})
        self.assertEqual(rows[0]["lock_world_state_prior"]["fields"], ["lock_occupancy", "vessel_motion_flow"])
        self.assertEqual(report["prior_applied_rows"], 1)
        self.assertEqual(report["recomputed_rows"], 0)

    def test_fill_keeps_existing_world_state_field(self):
        rows = [
            {
                "id": "test:prediction:scene_a",
                "prediction_json": {
                    "lock_occupancy": {"current": {"num_ships": 9}},
                },
            }
        ]
        states = {
            "scene_a": {
                "scene_token": "scene_a",
                "lock_occupancy": {"current": {"num_ships": 1}},
                "vessel_motion_flow": {"input_window": []},
            }
        }

        report = _MODULE.apply_lock_world_state_prior(rows, states, mode="fill")

        pred = rows[0]["prediction_json"]
        self.assertEqual(pred["lock_occupancy"]["current"]["num_ships"], 9)
        self.assertEqual(pred["vessel_motion_flow"]["input_window"], [])
        self.assertEqual(report["changed_fields"]["lock_occupancy"], 0)
        self.assertEqual(report["changed_fields"]["vessel_motion_flow"], 1)

    def test_recomputes_schema_when_reference_has_world_state_fields(self):
        rows = [
            {
                "id": "test:prediction:scene_a",
                "prediction_json": {"current_state": {"upper_gate_state": "open"}},
                "reference": {
                    "current_state": {"upper_gate_state": "open"},
                    "lock_occupancy": {"current": {"num_ships": 1}},
                    "vessel_motion_flow": {"input_window": []},
                },
            }
        ]
        states = {
            "scene_a": {
                "scene_token": "scene_a",
                "lock_occupancy": {"current": {"num_ships": 1}},
                "vessel_motion_flow": {"input_window": []},
            }
        }

        report = _MODULE.apply_lock_world_state_prior(rows, states)

        self.assertEqual(report["recomputed_rows"], 1)
        self.assertTrue(rows[0]["schema_check"]["valid_json"])
        self.assertEqual(rows[0]["schema_check"]["missing_top_level_keys"], [])


if __name__ == "__main__":
    unittest.main()
