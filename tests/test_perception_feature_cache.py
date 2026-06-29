import unittest

from tools.build_perception_feature_cache import (
    RTMDET_CLASSES,
    SHIP_2D_CLASSES,
    _summarize_labels,
)


class TestPerceptionFeatureCache(unittest.TestCase):
    def test_rtmdet_only_vessel_categories_are_not_counted_as_ship_detections(self):
        label_by_name = {name: index for index, name in enumerate(RTMDET_CLASSES)}
        summary = _summarize_labels(
            [
                (label_by_name["Fully_loaded_cargo_ship"], 0.9),
                (label_by_name["Tugboat"], 0.8),
                (label_by_name["Unknown_vessel"], 0.7),
            ],
            RTMDET_CLASSES,
            SHIP_2D_CLASSES,
        )

        self.assertEqual(summary["counts_by_class"]["Tugboat"], 1)
        self.assertEqual(summary["counts_by_class"]["Unknown_vessel"], 1)
        self.assertEqual(summary["num_ship_detections"], 1)


if __name__ == "__main__":
    unittest.main()
