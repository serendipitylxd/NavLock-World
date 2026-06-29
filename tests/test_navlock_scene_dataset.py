import unittest

from navlock_world.datasets import NavLockSceneDataset, navlock_scene_collate


class TestNavLockSceneDataset(unittest.TestCase):
    def test_recognition_sample_contains_multimodal_fields(self):
        dataset = NavLockSceneDataset(data_root="data", split="train", mode="recognition")
        self.assertEqual(len(dataset), 459)

        sample = dataset[0]
        self.assertEqual(sample["mode"], "recognition")
        self.assertGreaterEqual(sample["num_frames"], 1)

        frame = sample["frames"][0]
        self.assertEqual(sorted(frame["images"].keys()), [f"CAM_{i}" for i in range(1, 9)])
        self.assertIn("path", frame["images"]["CAM_1"])
        self.assertEqual(frame["images"]["CAM_3"]["camera_role"], "state")
        self.assertFalse(frame["images"]["CAM_3"]["is_calibrated"])
        self.assertEqual(frame["images"]["CAM_8"]["camera_role"], "state")
        self.assertFalse(frame["images"]["CAM_8"]["is_calibrated"])

        self.assertEqual(frame["lidar"]["channel"], "LIDAR_TOP")
        self.assertIn("path", frame["lidar"])
        self.assertIn("upper_gate_state", frame["lock_state_labels"])
        self.assertIn("water_state", frame["lock_state_labels"])
        self.assertIn("water_state", frame["lock_state"])
        self.assertIn("water_level", frame["lock_state"])
        self.assertIn("water_level", frame)
        self.assertIsInstance(frame["water_level"], float)
        self.assertNotIn("water_label", frame["lock_state"])
        self.assertIn("water_state_labels", sample["metadata"]["label_schema"])
        self.assertIn("water_level", sample["metadata"]["label_schema"])
        self.assertNotIn("water_labels", sample["metadata"]["label_schema"])

    def test_prediction_mode_filters_scenes_without_target(self):
        dataset = NavLockSceneDataset(data_root="data", split="train", mode="prediction")
        self.assertEqual(len(dataset), 199)

        sample = dataset[0]
        self.assertTrue(sample["has_prediction_target"])
        self.assertGreater(len(sample["prediction"]["input_frames"]), 0)
        self.assertGreater(len(sample["prediction"]["target_frames"]), 0)
        self.assertLessEqual(
            max(frame["relative_time_sec"] for frame in sample["prediction"]["input_frames"]),
            50.0,
        )
        self.assertGreater(
            min(frame["relative_time_sec"] for frame in sample["prediction"]["target_frames"]),
            50.0,
        )

    def test_collate_adds_variable_length_masks(self):
        dataset = NavLockSceneDataset(data_root="data", split="train", mode="all")
        batch = navlock_scene_collate([dataset[0], dataset[1]])

        self.assertEqual(len(batch["scene_tokens"]), 2)
        self.assertEqual(len(batch["frame_masks"]), 2)
        self.assertEqual(len(batch["prediction_input_masks"]), 2)
        self.assertEqual(len(batch["prediction_target_masks"]), 2)


if __name__ == "__main__":
    unittest.main()
