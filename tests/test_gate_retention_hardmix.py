import unittest

from tools.build_gate_retention_hardmix import (
    gate_bucket,
    gate_transition_labels,
    repeat_for_bucket,
)


class TestGateRetentionHardmix(unittest.TestCase):
    def test_focuses_open_closing_and_closed_opening_pairs(self):
        current = {
            "upper_gate_state": "open",
            "lower_gate_state": "closed",
            "water_state": "idle",
        }
        future = {
            "upper_gate_state": "closing",
            "lower_gate_state": "opening",
            "water_state": "idle",
        }

        labels = gate_transition_labels(current, future)
        bucket = gate_bucket(current, future, labels)
        repeat = repeat_for_bucket(
            bucket=bucket,
            stable_repeat=2,
            transition_repeat=4,
            focus_transition_repeat=8,
            active_repeat=3,
        )

        self.assertEqual(
            labels,
            ["upper_open_to_closing", "lower_closed_to_opening"],
        )
        self.assertEqual(
            bucket,
            "focus_transition_lower_closed_to_opening+upper_open_to_closing",
        )
        self.assertEqual(repeat, 8)

    def test_repeats_stable_motion_labels_more_than_plain_stable_samples(self):
        current = {
            "upper_gate_state": "opening",
            "lower_gate_state": "closed",
            "water_state": "idle",
        }
        future = dict(current)

        labels = gate_transition_labels(current, future)
        bucket = gate_bucket(current, future, labels)
        repeat = repeat_for_bucket(
            bucket=bucket,
            stable_repeat=2,
            transition_repeat=4,
            focus_transition_repeat=8,
            active_repeat=3,
        )

        self.assertEqual(labels, [])
        self.assertEqual(bucket, "stable_motion_label")
        self.assertEqual(repeat, 3)


if __name__ == "__main__":
    unittest.main()
