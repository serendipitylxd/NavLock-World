#!/usr/bin/env python3
"""Run a minimal Qwen3-VL LoRA/QLoRA smoke fine-tuning job."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, unquote

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/qwen3vl_4b_lora_smoke.yaml",
        help="YAML config for the smoke run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, JSONL, processor, and one collated batch without loading the model.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML config value. Values are parsed as JSON when possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    apply_overrides(cfg, args.overrides)
    processor = AutoProcessor.from_pretrained(
        cfg["model_name_or_path"], local_files_only=True
    )
    dataset = Qwen3VLSFTDataset(
        path=Path(cfg["train_file"]),
        max_samples=int(cfg.get("max_samples") or 0) or None,
        skip_samples=int(cfg.get("skip_samples") or 0),
        max_images_per_sample=optional_int(cfg, "max_images_per_sample"),
        max_text_chars=optional_int(cfg, "max_text_chars"),
    )
    eval_dataset = None
    if cfg.get("eval_file"):
        eval_dataset = Qwen3VLSFTDataset(
            path=Path(cfg["eval_file"]),
            max_samples=int(cfg.get("max_eval_samples") or 0) or None,
            skip_samples=int(cfg.get("skip_eval_samples") or 0),
            max_images_per_sample=optional_int(cfg, "max_images_per_sample"),
            max_text_chars=optional_int(cfg, "max_text_chars"),
        )
    collator = Qwen3VLSFTCollator(processor)
    batch = collator([dataset[0]])
    print(f"train_file={cfg['train_file']}")
    print(f"num_samples={len(dataset)}")
    print(f"first_batch_input_shape={tuple(batch['input_ids'].shape)}")
    print(f"first_batch_num_supervised_tokens={(batch['labels'] != -100).sum().item()}")
    if eval_dataset is not None:
        eval_batch = collator([eval_dataset[0]])
        print(f"eval_file={cfg['eval_file']}")
        print(f"num_eval_samples={len(eval_dataset)}")
        print(f"first_eval_batch_input_shape={tuple(eval_batch['input_ids'].shape)}")

    if args.dry_run:
        print("dry_run=ok")
        return

    validate_training_device(cfg)
    model = build_training_model(cfg)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        max_steps=int(cfg["max_steps"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(
            cfg.get("per_device_eval_batch_size", cfg["per_device_train_batch_size"])
        ),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_steps=int(cfg["warmup_steps"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        adam_beta1=float(cfg.get("adam_beta1", 0.9)),
        adam_beta2=float(cfg.get("adam_beta2", 0.999)),
        adam_epsilon=float(cfg.get("adam_epsilon", 1e-8)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        optim=cfg.get("optim", "adamw_torch"),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg["save_total_limit"]),
        overwrite_output_dir=bool(cfg.get("overwrite_output_dir", False)),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(cfg.get("eval_steps", cfg["save_steps"])),
        do_eval=eval_dataset is not None,
        bf16=bool(cfg.get("bf16", False)),
        fp16=bool(cfg.get("fp16", False)),
        remove_unused_columns=False,
        report_to=[],
        seed=int(cfg.get("seed", 42)),
        logging_nan_inf_filter=bool(cfg.get("logging_nan_inf_filter", False)),
        skip_memory_metrics=bool(cfg.get("skip_memory_metrics", True)),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    if bool(cfg.get("eval_only", False)):
        metrics = trainer.evaluate()
        print("eval_only=ok")
        print(json.dumps(metrics, sort_keys=True))
        return
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])
    print(f"saved={cfg['output_dir']}")


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = ["model_name_or_path", "train_file", "output_dir", "max_steps"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing required config keys: {missing}")
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use KEY=VALUE format: {override}")
        key, raw_value = override.split("=", 1)
        if not key:
            raise ValueError(f"override key cannot be empty: {override}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        cfg[key] = value


def optional_int(cfg: dict[str, Any], key: str) -> Optional[int]:
    if key not in cfg or cfg[key] is None or cfg[key] == "":
        return None
    return int(cfg[key])


def validate_training_device(cfg: dict[str, Any]) -> None:
    if bool(cfg.get("allow_cpu_training", False)):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen3-VL LoRA/QLoRA training requires a visible CUDA device. "
            "Run dry-run only, fix the NVIDIA driver/GPU runtime, or set "
            "allow_cpu_training=true for a deliberate CPU-only diagnostic."
        )


def load_model(cfg: dict[str, Any]) -> Qwen3VLForConditionalGeneration:
    quantization_config = None
    if bool(cfg.get("use_4bit", True)):
        compute_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[cfg.get("bnb_4bit_compute_dtype", "bfloat16")]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    return Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_name_or_path"],
        local_files_only=True,
        torch_dtype=torch.bfloat16 if bool(cfg.get("bf16", True)) else torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )


def build_training_model(cfg: dict[str, Any]) -> Any:
    model = load_model(cfg)
    use_gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if cfg.get("kbit_prepare_mode") == "no_fp32_cast":
        model = prepare_model_for_kbit_training_no_fp32_cast(
            model, use_gradient_checkpointing=use_gradient_checkpointing
        )
    else:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gradient_checkpointing
        )
    init_adapter_dir = cfg.get("init_adapter_dir")
    if init_adapter_dir:
        model = PeftModel.from_pretrained(
            model,
            init_adapter_dir,
            is_trainable=True,
        )
        move_trainable_params_to_model_device(model)
        return model
    return get_peft_model(model, build_lora_config(cfg))


def prepare_model_for_kbit_training_no_fp32_cast(
    model: torch.nn.Module, use_gradient_checkpointing: bool
) -> torch.nn.Module:
    """Prepare quantized training without PEFT's memory-heavy fp32 upcast."""
    for param in model.parameters():
        param.requires_grad = False
    if not use_gradient_checkpointing:
        return model
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return model
    input_embeddings = getattr(model, "get_input_embeddings", lambda: None)()
    if input_embeddings is not None:
        input_embeddings.register_forward_hook(
            lambda _module, _inputs, output: output.requires_grad_(True)
        )
    return model


def move_trainable_params_to_model_device(model: torch.nn.Module) -> None:
    target_device = next(
        (
            param.device
            for param in model.parameters()
            if param.device.type == "cuda" and not param.requires_grad
        ),
        None,
    )
    if target_device is None:
        if not torch.cuda.is_available():
            return
        target_device = torch.device("cuda:0")
    for param in model.parameters():
        if param.requires_grad and param.device != target_device:
            param.data = param.data.to(target_device)
            if param.grad is not None:
                param.grad = param.grad.to(target_device)


def build_lora_config(cfg: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["lora_target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


class Qwen3VLSFTDataset(Dataset):
    def __init__(
        self,
        path: Path,
        max_samples: Optional[int] = None,
        skip_samples: int = 0,
        max_images_per_sample: Optional[int] = None,
        max_text_chars: Optional[int] = None,
    ) -> None:
        self.items = [
            prepare_messages_item(
                json.loads(line),
                max_images_per_sample=max_images_per_sample,
                max_text_chars=max_text_chars,
            )
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if skip_samples:
            self.items = self.items[skip_samples:]
        if max_samples is not None:
            self.items = self.items[:max_samples]
        if not self.items:
            raise ValueError(f"no samples loaded from {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class Qwen3VLSFTCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = processor.tokenizer.eos_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(example["messages"]) for example in examples]
        input_ids = pad_sequence(
            [item["input_ids"] for item in encoded],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in encoded],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [item["labels"] for item in encoded],
            batch_first=True,
            padding_value=-100,
        )
        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [item[key] for item in encoded if key in item]
            if values:
                batch[key] = torch.cat(values, dim=0)
        return batch

    def _encode(self, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompt_messages = [messages[0]]
        prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = squeeze_batch(full)
        prompt_len = int(prompt["input_ids"].shape[-1])
        labels = encoded["input_ids"].clone()
        labels[:prompt_len] = -100
        encoded["labels"] = labels
        return encoded


def squeeze_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    squeezed = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key in {"input_ids", "attention_mask"}:
            squeezed[key] = value.squeeze(0)
        else:
            squeezed[key] = value
    return squeezed


def prepare_messages_item(
    item: dict[str, Any],
    max_images_per_sample: Optional[int] = None,
    max_text_chars: Optional[int] = None,
) -> dict[str, Any]:
    normalized = copy.deepcopy(item)
    for message in normalized["messages"]:
        content = message.get("content", [])
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
            continue
        if not isinstance(content, list):
            continue
        num_images = 0
        kept_content = []
        for part in content:
            if not isinstance(part, dict):
                kept_content.append(part)
                continue
            if part.get("type") == "image":
                num_images += 1
                if (
                    max_images_per_sample is not None
                    and num_images > max_images_per_sample
                ):
                    continue
                image = part.get("image")
                if isinstance(image, str) and image.startswith("file://"):
                    part["image"] = unquote(urlparse(image).path)
            elif part.get("type") == "text" and max_text_chars is not None:
                text = part.get("text")
                if isinstance(text, str) and len(text) > max_text_chars:
                    part["text"] = (
                        text[:max_text_chars]
                        + "\n\n[TRUNCATED FOR QWEN3-VL SMOKE TRAINING]"
                    )
            kept_content.append(part)
        message["content"] = kept_content
    return normalized


def normalize_file_uris(item: dict[str, Any]) -> dict[str, Any]:
    return prepare_messages_item(item)


if __name__ == "__main__":
    main()
