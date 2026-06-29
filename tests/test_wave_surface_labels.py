import unittest
from pathlib import Path

from tools.build_wave_surface_labels import build_wave_labels


class TestWaveSurfaceLabels(unittest.TestCase):
    def test_build_wave_labels_uses_target_camera_regions(self):
        labels = build_wave_labels(data_root=Path("data"), split="val")

        self.assertGreater(len(labels), 0)
        for item in labels:
            self.assertTrue(item["wave_expected"])
            self.assertFalse(item["image_verified"])
            self.assertFalse(item["image_level_waterline_annotation_required"])
            if item["water_state"] == "filling":
                self.assertEqual(item["camera"], "CAM_3")
                self.assertEqual(item["region_id"], "upper_gate_left_in_chamber")
                self.assertIn("/CAM_3/", item["image_path"])
            elif item["water_state"] == "emptying":
                self.assertEqual(item["camera"], "CAM_8")
                self.assertEqual(item["region_id"], "lower_gate_right_outside_chamber")
                self.assertIn("/CAM_8/", item["image_path"])
            else:
                self.fail(f"unexpected water_state: {item['water_state']}")


if __name__ == "__main__":
    unittest.main()
