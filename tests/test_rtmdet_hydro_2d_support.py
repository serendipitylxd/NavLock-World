import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from navlock_world.projection import (
    bbox_iou,
    camera_ray_to_lidar,
    project_lidar_box_to_image,
    project_lidar_point_to_image,
    triangulate_lidar_rays,
)


_SPEC = importlib.util.spec_from_file_location(
    "analyze_rtmdet_hydro_2d_support",
    Path(__file__).resolve().parent.parent / "tools" / "analyze_rtmdet_hydro_2d_support.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class TestProjectionHelpers(unittest.TestCase):
    def test_project_lidar_box_with_identity_camera_pose(self):
        calibration = {
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "camera_intrinsic": [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 50.0],
                [0.0, 0.0, 1.0],
            ],
        }

        bbox = project_lidar_box_to_image([0.0, 0.0, 10.0, 2.0, 2.0, 2.0, 0.0], calibration, 100, 100)

        self.assertIsNotNone(bbox)
        self.assertLess(bbox[0], 50.0)
        self.assertLess(bbox[1], 50.0)
        self.assertGreater(bbox[2], 50.0)
        self.assertGreater(bbox[3], 50.0)

    def test_bbox_iou(self):
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)

    def test_camera_ray_projection_and_triangulation(self):
        calibration = {
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "camera_intrinsic": [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 50.0],
                [0.0, 0.0, 1.0],
            ],
        }

        point = project_lidar_point_to_image([0.0, 0.0, 10.0], calibration, 100, 100)
        self.assertEqual(point, (50.0, 50.0))
        ray = camera_ray_to_lidar((50.0, 50.0), calibration)
        self.assertIsNotNone(ray)
        self.assertAlmostEqual(ray[0][0], 0.0)
        self.assertAlmostEqual(ray[1][2], 1.0)

        triangulated = triangulate_lidar_rays(
            [
                ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                ([1.0, 0.0, 0.0], [-1.0, 0.0, 10.0]),
            ]
        )
        self.assertIsNotNone(triangulated)
        point_3d, residual = triangulated
        self.assertAlmostEqual(point_3d[0], 0.0, places=5)
        self.assertAlmostEqual(point_3d[2], 10.0, places=5)
        self.assertLess(residual, 1e-5)


class TestRtmDetHydroSupportAnalyzer(unittest.TestCase):
    def test_load_rtmdet_ship_boxes_excludes_rtmdet_only_vessel_categories(self):
        predictions = [
            {
                "img_path": "data/cam.png",
                "pred_instances": {
                    "scores": torch.tensor([0.9, 0.8, 0.7]),
                    "labels": torch.tensor([5, 12, 13]),
                    "bboxes": torch.tensor(
                        [
                            [0.0, 0.0, 10.0, 10.0],
                            [10.0, 10.0, 20.0, 20.0],
                            [20.0, 20.0, 30.0, 30.0],
                        ]
                    ),
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rtmdet.pkl"
            with path.open("wb") as f:
                pickle.dump(predictions, f)

            boxes = _MODULE.load_rtmdet_ship_boxes(path, score_threshold=0.1)

        self.assertEqual(
            [box["label_name"] for box in boxes["data/cam.png"]],
            ["Unladen_cargo_ship"],
        )

    def test_hydro_ship_detections_filters_class_and_score(self):
        detections = _MODULE.hydro_ship_detections(
            {
                "boxes": [[1, 2, 3, 4, 5, 6, 0], [7, 8, 9, 1, 1, 1, 0]],
                "label_names": ["Unladen_cargo_ship", "Lock_footbridge"],
                "scores": [0.2, 0.9],
            },
            score_threshold=0.15,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label_name"], "Unladen_cargo_ship")

    def test_matched_rtmdet_indices_greedy_matches_once(self):
        matched = _MODULE.matched_rtmdet_indices(
            [
                {"bbox": (0.0, 0.0, 10.0, 10.0)},
                {"bbox": (20.0, 20.0, 30.0, 30.0)},
            ],
            [
                {"bbox": (1.0, 1.0, 9.0, 9.0), "score": 0.9},
                {"bbox": (21.0, 21.0, 29.0, 29.0), "score": 0.8},
                {"bbox": (50.0, 50.0, 60.0, 60.0), "score": 0.7},
            ],
            support_iou_threshold=0.3,
        )

        self.assertEqual(matched, {0, 1})


if __name__ == "__main__":
    unittest.main()
