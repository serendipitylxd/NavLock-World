import importlib.util
import unittest
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "analyze_rtmdet_ship_intention_support",
    Path(__file__).resolve().parent.parent / "tools" / "analyze_rtmdet_ship_intention_support.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class TestRtmDetShipIntentionSupport(unittest.TestCase):
    def test_best_rtmdet_match_returns_highest_supported_iou(self):
        match = _MODULE.best_rtmdet_match(
            (0.0, 0.0, 10.0, 10.0),
            [
                {"bbox": (30.0, 30.0, 40.0, 40.0), "score": 0.9},
                {"bbox": (1.0, 1.0, 9.0, 9.0), "score": 0.8},
                {"bbox": (2.0, 2.0, 8.0, 8.0), "score": 0.7},
            ],
            support_iou_threshold=0.3,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match["index"], 1)
        self.assertGreater(match["iou"], 0.6)

    def test_best_rtmdet_match_rejects_low_iou(self):
        match = _MODULE.best_rtmdet_match(
            (0.0, 0.0, 10.0, 10.0),
            [{"bbox": (20.0, 20.0, 30.0, 30.0), "score": 0.9}],
            support_iou_threshold=0.3,
        )

        self.assertIsNone(match)

    def test_motion_summary_for_camera_normalizes_by_image_diagonal(self):
        motion = _MODULE.motion_summary_for_camera(
            "CAM_6",
            [
                {
                    "frame_index": 1,
                    "center": (0.0, 0.0),
                    "width": 3,
                    "height": 4,
                    "score": 0.8,
                    "iou": 0.6,
                },
                {
                    "frame_index": 2,
                    "center": (3.0, 4.0),
                    "width": 3,
                    "height": 4,
                    "score": 0.9,
                    "iou": 0.8,
                },
            ],
        )

        self.assertEqual(motion["pixel_displacement"], 5.0)
        self.assertEqual(motion["normalized_displacement"], 1.0)
        self.assertEqual(motion["mean_iou"], 0.7)

    def test_ship_intention_errors_splits_correct_wrong_missed_extra(self):
        row = {
            "prediction_json": {
                "ship_behavior": {
                    "ship_intentions": [
                        {
                            "instance_token": "ship_1",
                            "category": "Cargo",
                            "ship_intentions": ["ship_berthed"],
                        },
                        {
                            "instance_token": "ship_2",
                            "category": "Cargo",
                            "ship_intentions": ["ship_entering_lock"],
                        },
                        {
                            "instance_token": "ship_extra",
                            "category": "Cargo",
                            "ship_intentions": ["ship_leaving_lock"],
                        },
                    ]
                }
            },
            "reference": {
                "ship_behavior": {
                    "ship_intentions": [
                        {
                            "instance_token": "ship_1",
                            "category": "Cargo",
                            "ship_intentions": ["ship_berthed"],
                        },
                        {
                            "instance_token": "ship_2",
                            "category": "Cargo",
                            "ship_intentions": ["ship_leaving_lock"],
                        },
                        {
                            "instance_token": "ship_missed",
                            "category": "Cargo",
                            "ship_intentions": ["ship_entering_lock"],
                        },
                    ]
                }
            },
        }

        errors = _MODULE.ship_intention_errors(row)

        self.assertEqual([item["instance_token"] for item in errors["correct"]], ["ship_1"])
        self.assertEqual([item["instance_token"] for item in errors["wrong"]], ["ship_2"])
        self.assertEqual(errors["wrong"][0]["reference"], "ship_leaving_lock")
        self.assertEqual(errors["wrong"][0]["predicted"], "ship_entering_lock")
        self.assertEqual([item["instance_token"] for item in errors["missed"]], ["ship_missed"])
        self.assertEqual([item["instance_token"] for item in errors["extra"]], ["ship_extra"])


if __name__ == "__main__":
    unittest.main()
