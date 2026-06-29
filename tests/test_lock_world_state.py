import unittest

from navlock_world.lock_world_state import derive_sequence_world_state


def berth(x_min, y_min, x_max, y_max):
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "cx": (x_min + x_max) / 2,
        "cy": (y_min + y_max) / 2,
    }


# Two berths along the chamber Y axis (upper gate = high Y, lower gate = low Y).
BERTHS = [
    berth(40.0, 30.0, 52.0, 95.0),   # berth_slot_01, cy = 62.5
    berth(50.0, 105.0, 62.0, 165.0),  # berth_slot_02, cy = 135
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
    # s1 sits still inside berth_slot_01; s2 moves up the chamber (increasing Y).
    s2_y = [100.0, 130.0, 160.0, 190.0, 220.0, 250.0]
    frames = [
        frame(t, [ship("s1", 46.0, 62.0), ship("s2", 56.0, s2_y[t])])
        for t in range(6)
    ]
    return {
        "scene_token": "scene_fake_0001",
        "frames": frames,
        "prediction_input_frame_indices": [0, 1, 2, 3],
        "prediction_target_frame_indices": [4, 5],
    }


class TestLockWorldState(unittest.TestCase):
    def setUp(self):
        self.state = derive_sequence_world_state(fake_sequence(), BERTHS)

    def test_output_structure(self):
        occ = self.state["lock_occupancy"]
        self.assertIn("current", occ)
        self.assertIn("future_10s", occ)
        flow = self.state["vessel_motion_flow"]
        self.assertIn("input_window", flow)
        self.assertIn("target_window", flow)
        self.assertEqual(self.state["scene_token"], "scene_fake_0001")

    def test_current_occupancy(self):
        current = self.state["lock_occupancy"]["current"]
        self.assertEqual(current["num_ships"], 2)
        slot1 = next(s for s in current["berth_slots"] if s["region_id"] == "berth_slot_01")
        self.assertTrue(slot1["occupied"])
        self.assertIn("s1", slot1["ship_tokens"])

    def test_berthed_ship(self):
        flow = self.state["vessel_motion_flow"]["input_window"]
        s1 = next(f for f in flow if f["instance_token"] == "s1")
        self.assertEqual(s1["motion_state"], "ship_berthed")

    def test_moving_ship_direction(self):
        flow = self.state["vessel_motion_flow"]["input_window"]
        s2 = next(f for f in flow if f["instance_token"] == "s2")
        self.assertIn(s2["direction_label"], {"moving_to_upper_gate", "moving_to_lower_gate"})
        self.assertEqual(s2["direction_label"], "moving_to_upper_gate")


if __name__ == "__main__":
    unittest.main()
