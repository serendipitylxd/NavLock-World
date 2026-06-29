import unittest

from navlock_world.berth_ship_intentions import derive_ship_intentions


def berth(berth_id, x_min, y_min, x_max, y_max):
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "cx": (x_min + x_max) / 2,
        "cy": (y_min + y_max) / 2,
    }


# Two berths along the chamber Y axis. Upper gate = high Y, lower gate = low Y.
BERTHS = [
    berth("berth_001", 40.0, 30.0, 52.0, 95.0),   # cy = 62.5 (low Y, lower gate)
    berth("berth_002", 50.0, 105.0, 62.0, 165.0),  # cy = 135 (high Y, upper gate)
]


def frame(time, instances, upper="open", lower="closed"):
    return {
        "relative_time_sec": time,
        "lock_state": {"upper_gate_state": upper, "lower_gate_state": lower},
        "instances_3d": instances,
    }


def ship(token, x, y, category="Unladen_cargo_ship"):
    return {"instance_token": token, "category": category, "translation": [x, y, -1.0]}


class TestBerthShipIntentions(unittest.TestCase):
    def test_stationary_in_box_is_berthed(self):
        frames = [frame(t, [ship("s1", 46.0, 62.0)]) for t in range(5)]

        items = derive_ship_intentions(frames, BERTHS)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["instance_token"], "s1")
        self.assertEqual(items[0]["ship_intentions"], ["ship_berthed"])

    def test_moving_away_from_open_gate_is_entering(self):
        # Upper gate open -> open end is high Y; moving toward low Y = entering.
        frames = [
            frame(0, [ship("s2", 56.0, 200.0)]),
            frame(1, [ship("s2", 54.0, 160.0)]),
            frame(2, [ship("s2", 50.0, 100.0)]),
        ]

        items = derive_ship_intentions(frames, BERTHS)

        self.assertEqual(items[0]["ship_intentions"], ["ship_entering_lock"])

    def test_moving_toward_open_gate_is_leaving(self):
        # Upper gate open -> open end is high Y; moving toward high Y = leaving.
        frames = [
            frame(0, [ship("s3", 56.0, 120.0)]),
            frame(1, [ship("s3", 56.0, 160.0)]),
            frame(2, [ship("s3", 56.0, 200.0)]),
        ]

        items = derive_ship_intentions(frames, BERTHS)

        self.assertEqual(items[0]["ship_intentions"], ["ship_leaving_lock"])

    def test_single_berth_lower_gate_direction_uses_gate_state(self):
        # Lower gate open -> open end is low Y; moving toward high Y = entering,
        # even when the single berth makes min_y == mean_y.
        single_berth = [BERTHS[0]]
        frames = [
            frame(0, [ship("s5", 46.0, 20.0)], upper="closed", lower="open"),
            frame(1, [ship("s5", 46.0, 40.0)], upper="closed", lower="open"),
            frame(2, [ship("s5", 46.0, 70.0)], upper="closed", lower="open"),
        ]

        items = derive_ship_intentions(frames, single_berth)

        self.assertEqual(items[0]["ship_intentions"], ["ship_entering_lock"])

    def test_entered_and_parked_is_berthed(self):
        # Large net displacement but settled at a berth centre for the last
        # several frames (low end-of-window speed) -> future berthed.
        ys = [200.0, 150.0, 100.0, 62.0, 62.0, 62.0, 62.0, 62.0]
        frames = [frame(t, [ship("s4", 46.0, y)]) for t, y in enumerate(ys)]

        items = derive_ship_intentions(frames, BERTHS)

        self.assertEqual(items[0]["ship_intentions"], ["ship_berthed"])

    def test_footbridge_is_excluded(self):
        frames = [
            frame(t, [
                ship("s1", 46.0, 62.0),
                ship("bridge", 46.0, 62.0, category="Lock_footbridge"),
            ])
            for t in range(5)
        ]

        items = derive_ship_intentions(frames, BERTHS)

        tokens = {item["instance_token"] for item in items}
        self.assertEqual(tokens, {"s1"})

    def test_no_berths_returns_empty(self):
        frames = [frame(0, [ship("s1", 46.0, 62.0)])]
        self.assertEqual(derive_ship_intentions(frames, []), [])


if __name__ == "__main__":
    unittest.main()
