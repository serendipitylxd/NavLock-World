import importlib.util
import unittest
from pathlib import Path

from navlock_world.lock_world_state import derive_sequence_world_state

# Load the deriver tool module by path (tools/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "derive_world_state_from_detections",
    Path(__file__).resolve().parent.parent / "tools" / "derive_world_state_from_detections.py",
)
_DERIVER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DERIVER)
derive_prediction_from_input = _DERIVER.derive_prediction_from_input


def berth(x_min, y_min, x_max, y_max):
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "cx": (x_min + x_max) / 2,
        "cy": (y_min + y_max) / 2,
    }


BERTHS = [
    berth(40.0, 30.0, 52.0, 95.0),
    berth(50.0, 105.0, 62.0, 165.0),
]


def ship(token, x, y, category="Unladen_cargo_ship"):
    return {"instance_token": token, "category": category, "translation": [x, y, -1.0]}


def frame(t, ships):
    return {
        "sample_token": f"sample_{t}",
        "relative_time_sec": float(t),
        "lock_state": {"upper_gate_state": "open", "lower_gate_state": "closed", "water_state": "idle"},
        "instances_3d": ships,
    }


def fake_sequence():
    s2_y = [100.0, 130.0, 160.0, 190.0, 220.0, 250.0]
    frames = [frame(t, [ship("s1", 46.0, 62.0), ship("s2", 56.0, s2_y[t])]) for t in range(6)]
    return {
        "scene_token": "scene_fake_0001",
        "has_prediction_target": True,
        "frames": frames,
        "prediction_input_frame_indices": [0, 1, 2, 3],
        "prediction_target_frame_indices": [4, 5],
    }


def fake_current_only_sequence():
    seq = fake_sequence()
    seq["scene_token"] = "scene_current_only"
    seq["has_prediction_target"] = False
    seq["prediction_target_frame_indices"] = []
    return seq


class TestWorldStateDeriver(unittest.TestCase):
    def setUp(self):
        self.seq = fake_sequence()
        self.pred = derive_prediction_from_input(
            self.seq,
            BERTHS,
            future_motion_mode="persistence",
        )
        self.gt = derive_sequence_world_state(self.seq, BERTHS)

    def test_non_leaky_uses_only_input_frames(self):
        # future == current and target == input (persistence), so the deriver never
        # reads the target frames -> non-leaky.
        occ = self.pred["lock_occupancy"]
        self.assertEqual(occ["future_10s"], occ["current"])
        flow = self.pred["vessel_motion_flow"]
        self.assertEqual(flow["target_window"], flow["input_window"])

    def test_current_matches_gt_exactly(self):
        # The observed parts are derived from the same input frames as the GT.
        self.assertEqual(
            self.pred["lock_occupancy"]["current"],
            self.gt["lock_occupancy"]["current"],
        )
        self.assertEqual(
            self.pred["vessel_motion_flow"]["input_window"],
            self.gt["vessel_motion_flow"]["input_window"],
        )

    def test_scene_token_preserved(self):
        self.assertEqual(self.pred["scene_token"], "scene_fake_0001")

    def test_current_only_gt_omits_future_fields(self):
        gt = derive_sequence_world_state(fake_current_only_sequence(), BERTHS)

        self.assertIn("current", gt["lock_occupancy"])
        self.assertIn("input_window", gt["vessel_motion_flow"])
        self.assertNotIn("future_10s", gt["lock_occupancy"])
        self.assertNotIn("target_window", gt["vessel_motion_flow"])

    def test_settle_aware_future_motion_marks_near_berth_mover_static(self):
        pred = derive_prediction_from_input(self.seq, BERTHS)
        target_flow = pred["vessel_motion_flow"]["target_window"]
        s2 = next(item for item in target_flow if item["instance_token"] == "s2")
        self.assertEqual(s2["motion_state"], "ship_static")
        self.assertEqual(s2["direction_label"], "static_or_settled")
        self.assertEqual(s2["delta_xy"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
