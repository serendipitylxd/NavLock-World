import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.render_lidar_views_for_vlm import (
    DEFAULT_POINT_CLOUD_RANGE,
    collect_lidar_records,
    render_record,
)


class TestLidarViewRenderer(unittest.TestCase):
    def test_render_record_writes_bev_and_range_view_pngs(self):
        points = np.array(
            [
                [10.0, 20.0, -1.0, 0.0, 0.0],
                [20.0, 80.0, 1.0, 0.0, 0.0],
                [50.0, 160.0, 3.0, 0.0, 0.0],
                [90.0, 300.0, 8.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lidar_path = tmp_path / "lidar.bin"
            points.tofile(lidar_path)

            result = render_record(
                record={
                    "split": "val",
                    "sample_token": "sample_test",
                    "scene_token": "scene_test",
                    "frame_index": 1,
                    "relative_time_sec": 10.0,
                    "lidar_path": str(lidar_path),
                },
                output_root=tmp_path / "views",
                point_cloud_range=DEFAULT_POINT_CLOUD_RANGE,
                num_point_features=5,
                bev_size=(64, 128),
                range_size=(96, 32),
                overwrite=True,
            )

            self.assertEqual(result["status"], "rendered")
            bev = cv2.imread(result["bev_path"], cv2.IMREAD_COLOR)
            range_view = cv2.imread(result["range_view_path"], cv2.IMREAD_COLOR)
            self.assertIsNotNone(bev)
            self.assertIsNotNone(range_view)
            self.assertEqual(bev.shape[:2], (128, 64))
            self.assertEqual(range_view.shape[:2], (32, 96))
            self.assertGreater(int(bev.sum()), 0)
            self.assertGreater(int(range_view.sum()), 0)

    def test_collect_lidar_records_deduplicates_frames(self):
        row = {
            "split": "val",
            "scene_token": "scene_test",
            "input": {
                "frames": [
                    {
                        "sample_token": "sample_test",
                        "frame_index": 0,
                        "relative_time_sec": 0.0,
                        "lidar": {"path": "data/samples/LIDAR_TOP/a.bin"},
                    },
                    {
                        "sample_token": "sample_test",
                        "frame_index": 0,
                        "relative_time_sec": 0.0,
                        "lidar": {"path": "data/samples/LIDAR_TOP/a.bin"},
                    },
                ]
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            records = collect_lidar_records([path])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample_token"], "sample_test")


if __name__ == "__main__":
    unittest.main()
