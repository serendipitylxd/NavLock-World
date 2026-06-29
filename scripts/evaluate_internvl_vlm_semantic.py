#!/usr/bin/env python3
"""Evaluate InternVL on NavLock VLM semantic Qwen-style chat JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from scripts.evaluate_qwen3vl_lora_adapter import (
    apply_fusion_reasoning_guard,
    apply_gate_state_guard,
    apply_ship_behavior_guard,
    apply_water_level_guard,
    content_to_text,
    fusion_reasoning_context_from_prompt_messages,
    gate_state_context_from_prompt_messages,
    schema_check,
    semantic_check,
    ship_behavior_context_from_prompt_messages,
    split_prompt_and_reference,
    summarize_results,
    try_parse_json_object,
    water_level_context_from_prompt_messages,
    write_jsonl,
)
from scripts.train_qwen3vl_lora_smoke import prepare_messages_item


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_JSON_CONSTRAINT_PROMPT = """FINAL InternVL output-control override:
- Return one complete raw JSON object only.
- Follow compact_response_template field names and do not add commentary.
- Do not invent camera names or expand camera lists. If fusion_reasoning is present, calibrated_cameras must use only CAM_1, CAM_2, CAM_4, CAM_5, CAM_6, CAM_7, and state_cameras_without_geometry must use only CAM_3, CAM_8.
- For ship_behavior.ship_intentions, use only ship instance tokens present in ship_behavior_context.latest_ship_instances. If latest_ship_instances is missing, empty, truncated, or unclear, ship_behavior.ship_intentions must be [].
- Never create numbered ship sequences such as ship_1, ship_2, ... or CAM_1, CAM_2, ... beyond the allowed camera list.
- Keep arrays short and close every JSON object/array before stopping."""


def repair_json_prefix(text: str) -> Optional[dict[str, Any]]:
    """Recover the longest valid JSON-object prefix from a truncated InternVL output.

    InternVL zero-shot sometimes runs away on the trailing optional
    ``fusion_reasoning`` camera lists (or a runaway number) and never closes the
    top-level object. Every current/future gate, water, and ship field is
    generated before that trailing section, so closing the open brackets — or, if
    that fails, cutting back to the last complete delimiter — recovers a valid
    object with the critical fields intact. Returns ``None`` when nothing parses.
    """
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]

    stack: list[str] = []
    in_str = False
    esc = False
    last_delim: Optional[tuple[int, list[str]]] = None
    for index, char in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif char == "\\":
                esc = True
            elif char == '"':
                in_str = False
        elif char == '"':
            in_str = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()
            last_delim = (index + 1, list(stack))
        elif char == ",":
            last_delim = (index, list(stack))

    def close(prefix: str, open_stack: list[str], open_str: bool) -> str:
        out = prefix
        if open_str:
            out += '"'
        for bracket in reversed(open_stack):
            out += "}" if bracket == "{" else "]"
        return out

    candidates = [close(body.rstrip().rstrip(","), list(stack), in_str)]
    if last_delim is not None:
        cut, delim_stack = last_delim
        candidates.append(close(body[:cut].rstrip().rstrip(","), delim_stack, False))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="models/InternVL3_5-4B-Instruct",
        help="Local InternVL model directory.",
    )
    parser.add_argument(
        "--input-file",
        default="outputs/vlm_semantic/qwen3vl_4b/navlock_qwen3vl_4b_test.jsonl",
        help="Qwen-style VLM semantic JSONL with user images/text and assistant reference.",
    )
    parser.add_argument(
        "--output",
        default="outputs/internvl3_5_4b_eval/predictions_test24.jsonl",
        help="Output predictions JSONL.",
    )
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-text-chars", type=int, default=6000)
    parser.add_argument("--max-images-per-sample", type=int, default=4)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Optional generation repetition_penalty for InternVL.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Optional generation no_repeat_ngram_size for InternVL.",
    )
    parser.add_argument(
        "--max-tiles-per-image",
        type=int,
        default=1,
        help=(
            "InternVL dynamic image tiles per selected image. Keep at 1 for "
            "controlled 4-image VLM semantic runs on RTX 4080 16GB."
        ),
    )
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--use-internvl-json-constraint-prompt",
        action="store_true",
        help=(
            "Prepend InternVL-specific output-control rules to reduce runaway "
            "camera/ship arrays and avoid truncated JSON."
        ),
    )
    parser.add_argument(
        "--repair-truncated-json",
        action="store_true",
        help=(
            "When strict JSON parsing fails, recover the longest valid prefix by "
            "closing unterminated trailing camera/number arrays. The critical "
            "gate/water/ship fields precede the runaway, so they are preserved."
        ),
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional LoRA adapter directory to merge before evaluation.",
    )
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--apply-water-level-guard", action="store_true")
    parser.add_argument("--apply-gate-state-guard", action="store_true")
    parser.add_argument("--apply-fusion-reasoning-guard", action="store_true")
    parser.add_argument("--apply-ship-behavior-guard", action="store_true")
    parser.add_argument(
        "--score-file",
        default=None,
        help="Existing predictions JSONL to summarize without loading InternVL.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.score_file:
        rows = load_existing_predictions(Path(args.score_file))
        print(json.dumps(summarize_results(rows), ensure_ascii=True, sort_keys=True))
        return

    items = load_eval_items(
        Path(args.input_file),
        max_samples=args.max_samples if args.max_samples > 0 else None,
        skip_samples=args.skip_samples,
        max_images_per_sample=args.max_images_per_sample,
        max_text_chars=args.max_text_chars,
    )
    print(f"model_path={args.model_path}")
    print(f"input_file={args.input_file}")
    print(f"num_samples={len(items)}")
    print(f"first_id={items[0]['id']}")
    print(f"first_num_images={len(items[0]['image_paths'])}")
    print(f"first_text_chars={len(items[0]['question_text'])}")

    if args.dry_run:
        print("dry_run=ok")
        return

    if not args.trust_remote_code:
        raise RuntimeError(
            "InternVL GitHub-format checkpoints require --trust-remote-code. "
            "Only use this after reviewing/approving the local model code."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible. Run this script in the GPU-visible environment."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model = load_model(args)
    transform = build_transform(args.image_size)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.repetition_penalty is not None:
        generation_config["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size > 0:
        generation_config["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    results = []
    for index, item in enumerate(items, start=1):
        result = generate_one(
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            item=item,
            generation_config=generation_config,
            image_size=args.image_size,
            max_tiles_per_image=args.max_tiles_per_image,
            dtype=model_dtype(args),
            use_json_constraint_prompt=args.use_internvl_json_constraint_prompt,
            repair_truncated_json=args.repair_truncated_json,
            use_water_level_guard=args.apply_water_level_guard,
            use_gate_state_guard=args.apply_gate_state_guard,
            use_fusion_reasoning_guard=args.apply_fusion_reasoning_guard,
            use_ship_behavior_guard=args.apply_ship_behavior_guard,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "index": index,
                    "id": item["id"],
                    "valid_json": result["schema_check"]["valid_json"],
                    "json_repaired": result["json_repaired"],
                    "exact_nested_schema": is_exact_nested_schema(
                        result["schema_check"]
                    ),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )

    write_jsonl(Path(args.output), results)
    summary = summarize_results(results)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    print(f"saved={args.output}")


def load_existing_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no predictions loaded from {path}")
    return rows


def load_eval_items(
    path: Path,
    max_samples: Optional[int],
    skip_samples: int,
    max_images_per_sample: Optional[int],
    max_text_chars: Optional[int],
) -> list[dict[str, Any]]:
    items = []
    for raw_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or raw_index < skip_samples:
            continue
        raw = json.loads(line)
        prepared = prepare_messages_item(
            raw,
            max_images_per_sample=max_images_per_sample,
            max_text_chars=max_text_chars,
        )
        prompt_messages, reference = split_prompt_and_reference(prepared["messages"])
        user_content = prompt_messages[0].get("content", [])
        image_paths = [
            part["image"]
            for part in user_content
            if isinstance(part, dict) and part.get("type") == "image"
        ]
        question_text = content_to_text(user_content)
        items.append(
            {
                "id": raw.get("id"),
                "metadata": raw.get("metadata", {}),
                "prompt_messages": prompt_messages,
                "image_paths": image_paths,
                "question_text": question_text,
                "water_level_context": water_level_context_from_prompt_messages(
                    prompt_messages
                ),
                "gate_state_context": gate_state_context_from_prompt_messages(
                    prompt_messages
                ),
                "fusion_reasoning_context": fusion_reasoning_context_from_prompt_messages(
                    prompt_messages
                ),
                "ship_behavior_context": ship_behavior_context_from_prompt_messages(
                    prompt_messages
                ),
                "reference": reference,
            }
        )
        if max_samples is not None and len(items) >= max_samples:
            break
    if not items:
        raise ValueError(f"no samples loaded from {path}")
    return items


def load_model(args: argparse.Namespace) -> Any:
    kwargs: dict[str, Any] = {
        "dtype": model_dtype(args),
        "low_cpu_mem_usage": True,
        "use_flash_attn": False,
        "trust_remote_code": True,
        "device_map": args.device_map,
        "local_files_only": True,
    }
    if args.load_in_8bit:
        kwargs["load_in_8bit"] = True
    model = AutoModel.from_pretrained(args.model_path, **kwargs).eval()
    if getattr(args, "adapter_dir", None):
        from peft import PeftModel

        # Merge the LoRA adapter back into the InternVLChatModel so model.chat()
        # (a base-model method, absent on the PeftModel wrapper) stays usable.
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        model = model.merge_and_unload()
        model = model.eval()
    return model


def model_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.fp16:
        return torch.float16
    if args.bf16:
        return torch.bfloat16
    return torch.float32


def generate_one(
    model: Any,
    tokenizer: Any,
    transform: Any,
    item: dict[str, Any],
    generation_config: dict[str, Any],
    image_size: int,
    max_tiles_per_image: int,
    dtype: torch.dtype,
    use_json_constraint_prompt: bool,
    use_water_level_guard: bool,
    use_gate_state_guard: bool,
    use_fusion_reasoning_guard: bool,
    use_ship_behavior_guard: bool,
    repair_truncated_json: bool = False,
) -> dict[str, Any]:
    question, pixel_values, num_patches_list = build_question_and_pixels(
        item,
        transform=transform,
        image_size=image_size,
        max_tiles_per_image=max_tiles_per_image,
        dtype=dtype,
        use_json_constraint_prompt=use_json_constraint_prompt,
    )
    with torch.inference_mode():
        generated_text = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list if num_patches_list else None,
        )

    raw_prediction = try_parse_json_object(generated_text)
    json_repaired = False
    if raw_prediction is None and repair_truncated_json:
        repaired = repair_json_prefix(generated_text)
        if repaired is not None:
            raw_prediction = repaired
            json_repaired = True
    parsed_prediction = raw_prediction
    guard_report = None
    gate_guard_report = None
    fusion_guard_report = None
    ship_guard_report = None
    if use_water_level_guard:
        parsed_prediction, guard_report = apply_water_level_guard(
            raw_prediction,
            item.get("water_level_context", {}),
        )
    if use_gate_state_guard:
        parsed_prediction, gate_guard_report = apply_gate_state_guard(
            parsed_prediction,
            item.get("gate_state_context", {}),
        )
    if use_fusion_reasoning_guard:
        parsed_prediction, fusion_guard_report = apply_fusion_reasoning_guard(
            parsed_prediction,
            item.get("fusion_reasoning_context", {}),
        )
    if use_ship_behavior_guard:
        parsed_prediction, ship_guard_report = apply_ship_behavior_guard(
            parsed_prediction,
            item.get("ship_behavior_context", {}),
        )

    result = {
        "id": item.get("id"),
        "metadata": item.get("metadata", {}),
        "reference": item["reference"],
        "prediction_text": generated_text,
        "prediction_json": parsed_prediction,
        "json_repaired": json_repaired,
        "schema_check": schema_check(parsed_prediction, item["reference"]),
        "semantic_check": semantic_check(parsed_prediction, item["reference"]),
        "internvl_prompt": {
            "num_images": len(item["image_paths"]),
            "max_tiles_per_image": max_tiles_per_image,
            "num_patches_list": num_patches_list,
            "question_text_chars": len(item["question_text"]),
            "use_json_constraint_prompt": use_json_constraint_prompt,
        },
    }
    if (
        use_water_level_guard
        or use_gate_state_guard
        or use_fusion_reasoning_guard
        or use_ship_behavior_guard
    ):
        result["prediction_json_raw"] = raw_prediction
    if use_water_level_guard:
        result["water_level_guard"] = guard_report
    if use_gate_state_guard:
        result["gate_state_guard"] = gate_guard_report
    if use_fusion_reasoning_guard:
        result["fusion_reasoning_guard"] = fusion_guard_report
    if use_ship_behavior_guard:
        result["ship_behavior_guard"] = ship_guard_report
    return result


def build_question_and_pixels(
    item: dict[str, Any],
    transform: Any,
    image_size: int,
    max_tiles_per_image: int,
    dtype: torch.dtype,
    use_json_constraint_prompt: bool = False,
) -> tuple[str, Optional[torch.Tensor], list[int]]:
    image_paths = item.get("image_paths", [])
    question_text = item["question_text"]
    if use_json_constraint_prompt:
        question_text = question_text + "\n\n" + INTERNVL_JSON_CONSTRAINT_PROMPT
    if not image_paths:
        return question_text, None, []

    pixel_values_list = []
    num_patches_list = []
    image_lines = []
    for index, image_path in enumerate(image_paths, start=1):
        pixel_values = load_image(
            image_path,
            transform=transform,
            image_size=image_size,
            max_num=max_tiles_per_image,
        )
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        image_lines.append(f"Image-{index}: <image>")

    pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype).cuda()
    question = "\n".join(image_lines) + "\n" + question_text
    return question, pixel_values, num_patches_list


def build_transform(input_size: int) -> Any:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image(
    image_file: str,
    transform: Any,
    image_size: int,
    max_num: int,
) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    images = dynamic_preprocess(
        image,
        image_size=image_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    return torch.stack([transform(tile) for tile in images])


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for index in range(blocks):
        box = (
            (index % (target_width // image_size)) * image_size,
            (index // (target_width // image_size)) * image_size,
            ((index % (target_width // image_size)) + 1) * image_size,
            ((index // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def is_exact_nested_schema(schema: dict[str, Any]) -> bool:
    return (
        schema.get("valid_json")
        and not schema.get("missing_top_level_keys")
        and not schema.get("extra_top_level_keys")
        and not schema.get("missing_nested_paths")
        and not schema.get("extra_nested_paths")
        and not schema.get("type_mismatch_paths")
    )


if __name__ == "__main__":
    main()
