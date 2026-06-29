import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.train_qwen3vl_lora_smoke import (
    Qwen3VLSFTDataset,
    apply_overrides,
    move_trainable_params_to_model_device,
    normalize_file_uris,
    prepare_messages_item,
    validate_training_device,
)


class TestQwen3VLLoRASmoke(unittest.TestCase):
    def test_apply_overrides_parses_json_values(self):
        cfg = {"learning_rate": 1e-4}

        apply_overrides(
            cfg,
            [
                "learning_rate=0.000001",
                "eval_only=true",
                "optim=paged_adamw_8bit",
                "init_adapter_dir=outputs/qwen3vl_4b_lora_schema_paths8_noamp_v5",
            ],
        )

        self.assertEqual(cfg["learning_rate"], 0.000001)
        self.assertIs(cfg["eval_only"], True)
        self.assertEqual(cfg["optim"], "paged_adamw_8bit")
        self.assertEqual(
            cfg["init_adapter_dir"],
            "outputs/qwen3vl_4b_lora_schema_paths8_noamp_v5",
        )

    def test_normalize_file_uris_for_processor(self):
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "file:///tmp/navlock image.png",
                            "max_pixels": 65536,
                        },
                        {"type": "text", "text": "Return JSON."},
                    ],
                },
                {"role": "assistant", "content": "{}"},
            ]
        }

        normalized = normalize_file_uris(item)

        self.assertEqual(
            normalized["messages"][0]["content"][0]["image"],
            "/tmp/navlock image.png",
        )
        self.assertEqual(item["messages"][0]["content"][0]["image"], "file:///tmp/navlock image.png")

    def test_dataset_applies_uri_normalization(self):
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "file:///tmp/a.png"}],
                },
                {"role": "assistant", "content": "{}"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            dataset = Qwen3VLSFTDataset(path)

        self.assertEqual(dataset[0]["messages"][0]["content"][0]["image"], "/tmp/a.png")

    def test_dataset_can_skip_initial_samples(self):
        def make_item(name):
            return {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": name}]},
                    {"role": "assistant", "content": "{}"},
                ]
            }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(
                "\n".join(json.dumps(make_item(name)) for name in ["a", "b", "c"])
                + "\n",
                encoding="utf-8",
            )
            dataset = Qwen3VLSFTDataset(path, skip_samples=1, max_samples=1)

        self.assertEqual(dataset[0]["messages"][0]["content"][0]["text"], "b")

    def test_prepare_messages_limits_images_and_text_for_smoke(self):
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file:///tmp/a.png"},
                        {"type": "image", "image": "file:///tmp/b.png"},
                        {"type": "text", "text": "abcdefghij"},
                    ],
                },
                {"role": "assistant", "content": "{}"},
            ]
        }

        prepared = prepare_messages_item(
            item,
            max_images_per_sample=1,
            max_text_chars=4,
        )
        user_content = prepared["messages"][0]["content"]

        self.assertEqual(
            [part["image"] for part in user_content if part["type"] == "image"],
            ["/tmp/a.png"],
        )
        self.assertTrue(user_content[-1]["text"].startswith("abcd"))
        self.assertIn("TRUNCATED", user_content[-1]["text"])
        self.assertEqual(
            prepared["messages"][1]["content"],
            [{"type": "text", "text": "{}"}],
        )

    def test_move_trainable_params_noops_without_cuda_base_params(self):
        model = torch.nn.Linear(2, 2)
        before_device = next(model.parameters()).device

        move_trainable_params_to_model_device(model)

        self.assertEqual(next(model.parameters()).device, before_device)

    def test_validate_training_device_requires_cuda_by_default(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires a visible CUDA"):
                validate_training_device({})

    def test_validate_training_device_can_allow_cpu_diagnostic(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            validate_training_device({"allow_cpu_training": True})


if __name__ == "__main__":
    unittest.main()
