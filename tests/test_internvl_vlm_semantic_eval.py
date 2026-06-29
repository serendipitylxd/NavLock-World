import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.evaluate_internvl_vlm_semantic import (
    dynamic_preprocess,
    load_eval_items,
    repair_json_prefix,
)


class TestInternVLRouteBEval(unittest.TestCase):
    def test_load_eval_items_prepares_qwen_style_messages(self):
        item = {
            "id": "sample-1",
            "metadata": {"split": "test"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file:///tmp/a.jpg"},
                        {"type": "image", "image": "file:///tmp/b.jpg"},
                        {
                            "type": "text",
                            "text": (
                                '{"water_level_context":{"current_water_level":1.0}}'
                                + "abcdef" * 20
                            ),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": '{"current_state":{"water_level":1.0}}',
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            loaded = load_eval_items(
                path,
                max_samples=1,
                skip_samples=0,
                max_images_per_sample=1,
                max_text_chars=80,
            )

        self.assertEqual(loaded[0]["id"], "sample-1")
        self.assertEqual(loaded[0]["image_paths"], ["/tmp/a.jpg"])
        self.assertIn("TRUNCATED", loaded[0]["question_text"])
        self.assertEqual(loaded[0]["water_level_context"]["current_water_level"], 1.0)
        self.assertEqual(loaded[0]["reference"]["current_state"]["water_level"], 1.0)

    def test_dynamic_preprocess_respects_tile_cap(self):
        image = Image.new("RGB", (640, 360), color=(10, 20, 30))

        tiles = dynamic_preprocess(
            image,
            image_size=448,
            max_num=1,
            use_thumbnail=True,
        )

        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].size, (448, 448))

    def test_repair_json_prefix_closes_missing_top_level_brace(self):
        text = (
            '{"current_state":{"upper_gate_state":"closed"},'
            '"fusion_reasoning":{"calibrated_cameras":["CAM_8"],'
            '"state_cameras_without_geometry":["CAM_3"]}'
        )

        repaired = repair_json_prefix(text)

        self.assertIsNotNone(repaired)
        self.assertEqual(repaired["current_state"]["upper_gate_state"], "closed")
        self.assertEqual(
            repaired["fusion_reasoning"]["state_cameras_without_geometry"], ["CAM_3"]
        )

    def test_repair_json_prefix_drops_runaway_number(self):
        text = (
            '{"current_state":{"upper_gate_state":"open","water_state":"filling"},'
            '"future_state_10s":{"upper_gate_state":"open"},'
            '"water_surface_dynamics":{"water_level_delta":0' + "0" * 200
        )

        repaired = repair_json_prefix(text)

        self.assertIsNotNone(repaired)
        self.assertEqual(repaired["current_state"]["water_state"], "filling")
        self.assertEqual(repaired["future_state_10s"]["upper_gate_state"], "open")

    def test_repair_json_prefix_returns_none_without_object(self):
        self.assertIsNone(repair_json_prefix("no json here"))


if __name__ == "__main__":
    unittest.main()
