#!/usr/bin/env python3
"""LoRA fine-tune InternVL3.5 on NavLock VLM semantic Qwen-style chat JSONL.

InternVL has no HF processor chat-template, so this replicates ``model.chat``'s
prompt construction (the ``internvl2_5`` conversation template plus the
``<img><IMG_CONTEXT>...</img>`` image-token expansion) inside a data collator,
masks everything before the assistant answer, and trains LoRA adapters on the
Qwen3 language backbone. It mirrors ``train_qwen3vl_lora_smoke`` for config
handling and the 4-bit / no-fp32-cast memory path so the three VLM semantic models can
be trained under one controlled recipe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from scripts.evaluate_internvl_vlm_semantic import build_transform, load_image
from scripts.train_qwen3vl_lora_smoke import (
    apply_overrides,
    build_lora_config,
    load_config,
    move_trainable_params_to_model_device,
    optional_int,
    prepare_messages_item,
    prepare_model_for_kbit_training_no_fp32_cast,
    validate_training_device,
)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/internvl3_5_4b_lora_vlm_semantic.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--override", action="append", default=[], dest="overrides")
    return parser.parse_args()


def model_compute_dtype(cfg: dict[str, Any]) -> torch.dtype:
    if cfg.get("fp16"):
        return torch.float16
    if cfg.get("bf16"):
        return torch.bfloat16
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cfg.get("bnb_4bit_compute_dtype", "float16")]


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    apply_overrides(cfg, args.overrides)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    transform = build_transform(int(cfg.get("image_size", 448)))

    # num_image_token depends only on image/patch size + downsample ratio; read it
    # from the config so the collator does not need the model loaded for dry-run.
    import json as _json

    model_config = _json.loads(
        (Path(cfg["model_name_or_path"]) / "config.json").read_text(encoding="utf-8")
    )
    image_size = int(model_config.get("force_image_size", 448))
    patch_size = int(model_config["vision_config"]["patch_size"])
    downsample_ratio = float(model_config["downsample_ratio"])
    num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio**2))

    collator = InternVLSFTCollator(
        tokenizer=tokenizer,
        model_path=cfg["model_name_or_path"],
        transform=transform,
        num_image_token=num_image_token,
        img_context_token_id=img_context_token_id,
        image_size=int(cfg.get("image_size", 448)),
        max_tiles_per_image=int(cfg.get("max_tiles_per_image", 1)),
        dtype=model_compute_dtype(cfg),
    )

    dataset = InternVLSFTDataset(
        path=Path(cfg["train_file"]),
        max_samples=int(cfg.get("max_samples") or 0) or None,
        skip_samples=int(cfg.get("skip_samples") or 0),
        max_images_per_sample=optional_int(cfg, "max_images_per_sample"),
        max_text_chars=optional_int(cfg, "max_text_chars"),
    )
    eval_dataset = None
    if cfg.get("eval_file"):
        eval_dataset = InternVLSFTDataset(
            path=Path(cfg["eval_file"]),
            max_samples=int(cfg.get("max_eval_samples") or 0) or None,
            skip_samples=int(cfg.get("skip_eval_samples") or 0),
            max_images_per_sample=optional_int(cfg, "max_images_per_sample"),
            max_text_chars=optional_int(cfg, "max_text_chars"),
        )

    batch = collator([dataset[0]])
    print(f"train_file={cfg['train_file']}")
    print(f"num_samples={len(dataset)}")
    print(f"num_image_token={num_image_token}")
    print(f"first_batch_input_shape={tuple(batch['input_ids'].shape)}")
    print(f"first_batch_pixel_shape={tuple(batch['pixel_values'].shape)}")
    print(
        f"first_batch_num_supervised_tokens={(batch['labels'] != -100).sum().item()}"
    )
    print(
        f"first_batch_num_img_context_tokens="
        f"{(batch['input_ids'] == img_context_token_id).sum().item()}"
    )

    if args.dry_run:
        print("dry_run=ok")
        return

    validate_training_device(cfg)
    model = build_training_model(cfg)
    # The InternVLChatModel.forward reads ``self.img_context_token_id`` on the
    # base model; set it there, not on the PeftModel wrapper.
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    base_model.img_context_token_id = img_context_token_id
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
        adam_epsilon=float(cfg.get("adam_epsilon", 1e-8)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
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
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"saved={cfg['output_dir']}")


def load_internvl_model(cfg: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {
        "dtype": model_compute_dtype(cfg),
        "low_cpu_mem_usage": True,
        "use_flash_attn": False,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if bool(cfg.get("use_4bit", True)):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=model_compute_dtype(cfg),
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", False)),
        )
    return AutoModel.from_pretrained(cfg["model_name_or_path"], **kwargs)


def build_training_model(cfg: dict[str, Any]) -> Any:
    model = load_internvl_model(cfg)
    use_gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
    if use_gradient_checkpointing:
        language_model = getattr(model, "language_model", model)
        if hasattr(language_model, "gradient_checkpointing_enable"):
            language_model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model.config, "llm_config"):
            model.config.llm_config.use_cache = False
    if cfg.get("kbit_prepare_mode") == "no_fp32_cast":
        model = prepare_model_for_kbit_training_no_fp32_cast(
            model, use_gradient_checkpointing=use_gradient_checkpointing
        )
    elif bool(cfg.get("use_4bit", True)):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gradient_checkpointing
        )
    init_adapter_dir = cfg.get("init_adapter_dir")
    if init_adapter_dir:
        model = PeftModel.from_pretrained(model, init_adapter_dir, is_trainable=True)
        move_trainable_params_to_model_device(model)
        return model
    return get_peft_model(model, build_internvl_lora_config(cfg))


def build_internvl_lora_config(cfg: dict[str, Any]) -> LoraConfig:
    # No task_type: PeftModelForCausalLM would inject ``inputs_embeds`` which the
    # custom InternVLChatModel.forward does not accept. A task-type-less PeftModel
    # passes the batch (pixel_values/image_flags/labels) straight through, and the
    # base model still computes the LM loss.
    return LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["lora_target_modules"]),
        bias="none",
    )


class InternVLSFTDataset(Dataset):
    def __init__(
        self,
        path: Path,
        max_samples: Optional[int],
        skip_samples: int,
        max_images_per_sample: Optional[int],
        max_text_chars: Optional[int],
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


class InternVLSFTCollator:
    def __init__(
        self,
        tokenizer: Any,
        model_path: str,
        transform: Any,
        num_image_token: int,
        img_context_token_id: int,
        image_size: int,
        max_tiles_per_image: int,
        dtype: torch.dtype,
    ) -> None:
        self.tokenizer = tokenizer
        self.transform = transform
        self.num_image_token = num_image_token
        self.img_context_token_id = img_context_token_id
        self.image_size = image_size
        self.max_tiles_per_image = max_tiles_per_image
        self.dtype = dtype
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self._get_conv_template, self._system_message = _load_conv_template(model_path)

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
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        pixel_values = [item["pixel_values"] for item in encoded if item["pixel_values"] is not None]
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
            batch["image_flags"] = torch.cat(
                [item["image_flags"] for item in encoded if item["image_flags"] is not None],
                dim=0,
            )
        return batch

    def _encode(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        user_message = messages[0]
        answer = _content_text(messages[1]["content"])
        image_paths, question_text = _split_user_content(user_message["content"])

        tiles_per_image = []
        pixel_chunks = []
        for path in image_paths:
            tiles = load_image(
                path,
                transform=self.transform,
                image_size=self.image_size,
                max_num=self.max_tiles_per_image,
            )
            tiles_per_image.append(tiles.shape[0])
            pixel_chunks.append(tiles)

        image_lines = "".join(
            f"Image-{i}: <image>\n" for i in range(1, len(image_paths) + 1)
        )
        question = image_lines + question_text

        prompt_text = self._build_prompt(question, None)
        full_text = self._build_prompt(question, answer)
        prompt_text = self._expand_image_tokens(prompt_text, tiles_per_image)
        full_text = self._expand_image_tokens(full_text, tiles_per_image)

        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        full = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]
        labels = input_ids.clone()
        labels[: prompt_ids.shape[0]] = -100

        if pixel_chunks:
            pixel_values = torch.cat(pixel_chunks, dim=0).to(self.dtype)
            image_flags = torch.ones((pixel_values.shape[0], 1), dtype=torch.long)
        else:
            pixel_values = None
            image_flags = None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }

    def _build_prompt(self, question: str, answer: Optional[str]) -> str:
        template = self._get_conv_template()
        template.system_message = self._system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], answer)
        return template.get_prompt()

    def _expand_image_tokens(self, text: str, tiles_per_image: list[int]) -> str:
        for tiles in tiles_per_image:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * tiles
                + IMG_END_TOKEN
            )
            text = text.replace("<image>", image_tokens, 1)
        return text


def _load_conv_template(model_path: str):
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "internvl_conversation", os.path.join(model_path, "conversation.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import json as _json

    config = _json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    template_name = config["template"]
    system_message = module.get_conv_template(template_name).system_message
    return (lambda: module.get_conv_template(template_name)), system_message


def _split_user_content(content: Any) -> tuple[list[str], str]:
    image_paths = []
    texts = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image":
            image = part.get("image")
            if isinstance(image, str) and image.startswith("file://"):
                image = unquote(urlparse(image).path)
            image_paths.append(image)
        elif part.get("type") == "text":
            texts.append(str(part.get("text", "")))
    return image_paths, "\n".join(texts)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


if __name__ == "__main__":
    main()
