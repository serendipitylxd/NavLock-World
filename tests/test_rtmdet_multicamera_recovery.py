import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from navlock_world.lock_world_state import load_lock_chamber_bounds


_SPEC = importlib.util.spec_from_file_location(
    "recover_rtmdet_multicamera_3d",
    Path(__file__).resolve().parent.parent / "tools" / "recover_rtmdet_multicamera_3d.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _calibration(translation):
    return {
        "translation": translation,
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "camera_intrinsic": [
            [10.0, 0.0, 50.0],
            [0.0, 10.0, 50.0],
            [0.0, 0.0, 1.0],
        ],
    }


class TestRtmDetMulticameraRecovery(unittest.TestCase):
    def setUp(self):
        self.frame = {
            "images": {
                "CAM_1": {
                    "file_name": "cam1.png",
                    "is_calibrated": True,
                    "width": 100,
                    "height": 100,
                    "calibration": _calibration([0.0, 0.0, 0.0]),
                },
                "CAM_2": {
                    "file_name": "cam2.png",
                    "is_calibrated": True,
                    "width": 100,
                    "height": 100,
                    "calibration": _calibration([10.0, 0.0, 0.0]),
                },
            }
        }
        self.rtmdet_by_path = {
            "data/cam1.png": [
                {
                    "bbox": [45.0, 45.0, 55.0, 55.0],
                    "score": 0.9,
                    "label_name": "Unladen_cargo_ship",
                }
            ],
            "data/cam2.png": [
                {
                    "bbox": [35.0, 45.0, 45.0, 55.0],
                    "score": 0.8,
                    "label_name": "Unladen_cargo_ship",
                }
            ],
        }
        self.chamber = {
            "x_min": -5.0,
            "x_max": 5.0,
            "y_min": -5.0,
            "y_max": 5.0,
            "y_mean": 0.0,
        }

    def test_recovers_when_multicamera_count_exceeds_hydro_count(self):
        recovered = _MODULE.recover_frame_detections(
            self.frame,
            [],
            self.rtmdet_by_path,
            data_root=Path("data"),
            chamber=self.chamber,
            min_cameras=2,
            max_ray_residual_m=1.0,
            cluster_distance_m=5.0,
            existing_distance_m=5.0,
        )

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["support_camera_count"], 2)
        self.assertEqual(recovered[0]["rtmdet_lock_ship_count"], 1)
        self.assertEqual(recovered[0]["hydro_lock_ship_count"], 0)
        self.assertEqual(recovered[0]["recovery_missing_count"], 1)
        self.assertAlmostEqual(recovered[0]["x"], 0.0, places=5)
        self.assertAlmostEqual(recovered[0]["y"], 0.0, places=5)

    def test_does_not_recover_when_hydro_count_matches_multicamera_count(self):
        recovered = _MODULE.recover_frame_detections(
            self.frame,
            [
                {
                    "detection_id": 1,
                    "category": "Unladen_cargo_ship",
                    "x": 1.0,
                    "y": 0.0,
                    "z": 10.0,
                    "size": [60.0, 12.0, 6.0],
                    "yaw": 0.0,
                    "score": 0.9,
                }
            ],
            self.rtmdet_by_path,
            data_root=Path("data"),
            chamber=self.chamber,
            min_cameras=2,
            max_ray_residual_m=1.0,
            cluster_distance_m=5.0,
            existing_distance_m=5.0,
        )

        self.assertEqual(recovered, [])

    def test_camera_consensus_count_uses_unique_in_chamber_boxes(self):
        count = _MODULE.rtmdet_in_chamber_camera_consensus_count(
            self.frame,
            [],
            self.rtmdet_by_path,
            data_root=Path("data"),
            chamber=self.chamber,
            min_cameras=2,
            candidate_min_cameras=2,
            max_ray_residual_m=1.0,
            cluster_distance_m=5.0,
        )

        self.assertEqual(count, 1)

    def test_rtmdet_only_vessel_categories_do_not_drive_recovery_or_count(self):
        for label_name in ("Tugboat", "Unknown_vessel"):
            with self.subTest(label_name=label_name):
                rtmdet_by_path = {
                    path: [dict(box, label_name=label_name)]
                    for path, boxes in self.rtmdet_by_path.items()
                    for box in boxes
                }

                recovered = _MODULE.recover_frame_detections(
                    self.frame,
                    [],
                    rtmdet_by_path,
                    data_root=Path("data"),
                    chamber=self.chamber,
                    min_cameras=2,
                    max_ray_residual_m=1.0,
                    cluster_distance_m=5.0,
                    existing_distance_m=5.0,
                    allow_unknown_vessel_recovery=True,
                )
                count = _MODULE.rtmdet_in_chamber_camera_consensus_count(
                    self.frame,
                    [],
                    rtmdet_by_path,
                    data_root=Path("data"),
                    chamber=self.chamber,
                    min_cameras=2,
                    candidate_min_cameras=2,
                    max_ray_residual_m=1.0,
                    cluster_distance_m=5.0,
                    allow_unknown_vessel_recovery=True,
                )

                self.assertEqual(recovered, [])
                self.assertIsNone(count)

    def test_does_not_recover_outside_chamber(self):
        outside = dict(self.chamber)
        outside["x_min"] = 20.0
        outside["x_max"] = 30.0

        recovered = _MODULE.recover_frame_detections(
            self.frame,
            [],
            self.rtmdet_by_path,
            data_root=Path("data"),
            chamber=outside,
            min_cameras=2,
            max_ray_residual_m=1.0,
            cluster_distance_m=5.0,
            existing_distance_m=5.0,
        )

        self.assertEqual(recovered, [])

    def test_loads_physical_lock_chamber_bounds(self):
        payload = {
            "regions": [
                {
                    "name": "lock_chamber",
                    "x_range": [62.7, 39.7],
                    "y_range": [17.2, 307.2],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "boundary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            bounds = load_lock_chamber_bounds(path)

        self.assertEqual(
            bounds,
            {
                "x_min": 39.7,
                "x_max": 62.7,
                "y_min": 17.2,
                "y_max": 307.2,
                "y_mean": 162.2,
            },
        )

    def test_recovers_open_lower_gate_candidate_from_gate_camera_group(self):
        frame = {
            "lock_state": {"upper_gate_state": "closed", "lower_gate_state": "open"},
            "images": {
                **self.frame["images"],
                "CAM_4": {
                    "file_name": "cam4.png",
                    "is_calibrated": True,
                    "width": 100,
                    "height": 100,
                    "calibration": _calibration([0.0, 10.0, 0.0]),
                },
            },
        }
        rtmdet_by_path = {
            **self.rtmdet_by_path,
            "data/cam4.png": [
                {
                    "bbox": [45.0, 35.0, 55.0, 45.0],
                    "score": 0.7,
                    "label_name": "Unladen_cargo_ship",
                }
            ],
        }

        recovered = _MODULE.recover_open_gate_frame_detections(
            frame,
            [],
            rtmdet_by_path,
            data_root=Path("data"),
            chamber=self.chamber,
            min_cameras=3,
            max_ray_residual_m=1.0,
            cluster_distance_m=5.0,
            existing_distance_m=5.0,
            gate_zone_length_m=10.0,
        )

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["detection_source"], "rtmdet_open_gate_recovery")
        self.assertEqual(recovered[0]["open_gate_state"], "lower_gate_state")
        self.assertEqual(recovered[0]["support_camera_count"], 3)


if __name__ == "__main__":
    unittest.main()
