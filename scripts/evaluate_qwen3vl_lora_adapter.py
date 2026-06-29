#!/usr/bin/env python3
"""Run lightweight generation checks for a Qwen3-VL LoRA adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor

from scripts.train_qwen3vl_lora_smoke import (
    load_model,
    optional_int,
    prepare_messages_item,
)

NAVLOCK_SHIP_CATEGORIES = (
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Unladen_container_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
    "Tugboat",
    "Unknown_vessel",
)
NAVLOCK_SHIP_CATEGORY_BY_LOWER = {
    category.lower(): category for category in NAVLOCK_SHIP_CATEGORIES
}
NAVLOCK_SHIP_EVAL_CATEGORY_ALIASES = {
    "Fully_loaded_cargo_ship": "cargo_ship",
    "Unladen_cargo_ship": "cargo_ship",
    "Fully_loaded_container_ship": "container_vessel",
    "Unladen_container_ship": "container_vessel",
    "Fully_loaded_cargo_fleet": "cargo_fleet",
    "Unladen_cargo_fleet": "cargo_fleet",
    "Tugboat": "tugboat",
    "Unknown_vessel": "unknown_vessel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/qwen3vl_4b_lora_small.yaml",
        help="Training config that defines model, adapter, and data defaults.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Qwen3-VL JSONL file. Defaults to eval_file, then train_file from config.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="LoRA adapter directory. Defaults to output_dir from config.",
    )
    parser.add_argument(
        "--output",
        default="outputs/qwen3vl_4b_lora_small_eval/predictions.jsonl",
        help="Output JSONL with prompts, references, generations, and schema checks.",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append rows to --output instead of overwriting it.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based input row offset for chunked evaluation.",
    )
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--retry-invalid-json",
        type=int,
        default=0,
        help=(
            "Regenerate rows whose decoded text does not parse as JSON. "
            "Retries keep the same image/text budget."
        ),
    )
    parser.add_argument(
        "--invalid-json-retry-repetition-penalty",
        type=float,
        default=1.05,
        help=(
            "Optional repetition_penalty used only on invalid-JSON retry attempts. "
            "Set to 1.0 to retry with unchanged generation kwargs."
        ),
    )
    parser.add_argument(
        "--fail-on-invalid-json",
        action="store_true",
        help=(
            "Exit before writing output/rescored-output if any row remains invalid "
            "JSON after optional retries."
        ),
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=None,
        help="Override config max_text_chars for prompt truncation during evaluation.",
    )
    parser.add_argument(
        "--max-images-per-sample",
        type=int,
        default=None,
        help="Override config max_images_per_sample during evaluation.",
    )
    parser.add_argument(
        "--apply-water-level-guard",
        action="store_true",
        help=(
            "Postprocess generated JSON with water_level_context from the prompt. "
            "This keeps raw output in prediction_json_raw and scores prediction_json."
        ),
    )
    parser.add_argument(
        "--apply-gate-state-guard",
        action="store_true",
        help=(
            "Postprocess deterministic gate/current-state fields from "
            "gate_state_context. This only repairs current_state; future gate "
            "states are predictions and are not postprocessed."
        ),
    )
    parser.add_argument(
        "--apply-fusion-reasoning-guard",
        action="store_true",
        help=(
            "Postprocess fusion_reasoning camera-layout fields from "
            "fusion_reasoning_context. Raw output is kept in prediction_json_raw."
        ),
    )
    parser.add_argument(
        "--apply-ship-behavior-guard",
        action="store_true",
        help=(
            "Postprocess ship category spelling and mooring evidence fields. "
            "This does not modify ship intention labels."
        ),
    )
    parser.add_argument(
        "--score-file",
        default=None,
        help="Existing predictions JSONL to rescore without loading the model.",
    )
    parser.add_argument(
        "--context-file",
        default=None,
        help=(
            "Qwen3-VL JSONL with prompts used to provide guard contexts when "
            "rescoring --score-file."
        ),
    )
    parser.add_argument(
        "--rescored-output",
        default=None,
        help="Optional JSONL path for rescored/guarded rows when using --score-file.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail before model loading if CUDA is not available.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-sample generation progress bar.",
    )
    parser.add_argument(
        "--ban-exclamation-token",
        action="store_true",
        help="Forbid exclamation-mark token sequences during generation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.score_file:
        context_by_id = {}
        if args.context_file:
            context_by_id = load_guard_contexts(
                Path(args.context_file),
                max_images_per_sample=args.max_images_per_sample,
                max_text_chars=args.max_text_chars,
            )
        results = load_prediction_results(
            Path(args.score_file),
            context_by_id=context_by_id,
            use_water_level_guard=args.apply_water_level_guard,
            use_gate_state_guard=args.apply_gate_state_guard,
            use_fusion_reasoning_guard=args.apply_fusion_reasoning_guard,
            use_ship_behavior_guard=args.apply_ship_behavior_guard,
        )
        summary = summarize_results(results)
        if args.fail_on_invalid_json:
            require_valid_json_results(
                results,
                output_path=Path(args.rescored_output) if args.rescored_output else None,
            )
        if args.rescored_output:
            write_jsonl(Path(args.rescored_output), results)
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
        return

    cfg = load_yaml(Path(args.config))
    input_file = Path(args.input_file or cfg.get("eval_file") or cfg["train_file"])
    adapter_dir = Path(args.adapter_dir or cfg["output_dir"])
    max_samples = args.max_samples if args.max_samples > 0 else None
    items = load_eval_items(
        input_file,
        start_index=max(0, args.start_index),
        max_samples=max_samples,
        max_images_per_sample=(
            args.max_images_per_sample
            if args.max_images_per_sample is not None
            else optional_int(cfg, "max_images_per_sample")
        ),
        max_text_chars=(
            args.max_text_chars
            if args.max_text_chars is not None
            else optional_int(cfg, "max_text_chars")
        ),
    )

    print(f"input_file={input_file}")
    print(f"adapter_dir={adapter_dir}")
    print(f"num_samples={len(items)}")
    first = items[0]
    print(f"first_id={first.get('id')}")
    print(f"first_reference_schema_keys={sorted(first['reference'].keys())}")

    if args.dry_run:
        print("dry_run=ok")
        return

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("--require-cuda was set, but torch.cuda.is_available() is false")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(torch.cuda.current_device())}")

    processor = AutoProcessor.from_pretrained(
        cfg["model_name_or_path"], local_files_only=True
    )
    model = load_adapter_model(cfg, adapter_dir)
    items_iter = tqdm(
        items,
        desc="generate",
        unit="sample",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    results = []
    for item in items_iter:
        results.append(
            generate_one_with_invalid_json_retries(
                model=model,
                processor=processor,
                item=item,
                max_new_tokens=args.max_new_tokens,
                max_invalid_json_retries=max(0, args.retry_invalid_json),
                invalid_json_retry_repetition_penalty=(
                    args.invalid_json_retry_repetition_penalty
                ),
                use_water_level_guard=args.apply_water_level_guard,
                use_gate_state_guard=args.apply_gate_state_guard,
                use_fusion_reasoning_guard=args.apply_fusion_reasoning_guard,
                use_ship_behavior_guard=args.apply_ship_behavior_guard,
                ban_exclamation_token=args.ban_exclamation_token,
            )
        )
    summary = summarize_results(results)
    if args.fail_on_invalid_json:
        require_valid_json_results(results, output_path=Path(args.output))
    write_jsonl(Path(args.output), results, append=args.append_output)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    print(f"saved={args.output}")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_eval_items(
    path: Path,
    max_samples: Optional[int],
    max_images_per_sample: Optional[int],
    max_text_chars: Optional[int],
    start_index: int = 0,
) -> list[dict[str, Any]]:
    items = []
    for row_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if row_index < start_index:
            continue
        if not line.strip():
            continue
        raw = json.loads(line)
        prepared = prepare_messages_item(
            raw,
            max_images_per_sample=max_images_per_sample,
            max_text_chars=max_text_chars,
        )
        prompt_messages, reference = split_prompt_and_reference(prepared["messages"])
        items.append(
            {
                "id": raw.get("id"),
                "metadata": raw.get("metadata", {}),
                "prompt_messages": prompt_messages,
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


def load_guard_contexts(
    path: Path,
    max_images_per_sample: Optional[int],
    max_text_chars: Optional[int],
) -> dict[str, dict[str, Any]]:
    contexts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        prepared = prepare_messages_item(
            raw,
            max_images_per_sample=max_images_per_sample,
            max_text_chars=max_text_chars,
        )
        prompt_messages, _ = split_prompt_and_reference(prepared["messages"])
        contexts[raw.get("id")] = {
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
        }
    return contexts


def split_prompt_and_reference(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("expected the final message to be the assistant reference")
    reference_text = content_to_text(messages[-1].get("content", ""))
    reference = parse_json_object(reference_text)
    return messages[:-1], reference


def water_level_context_from_prompt_messages(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    for message in messages:
        if message.get("role") != "user":
            continue
        text = content_to_text(message.get("content", ""))
        context = extract_named_json_value(text, "water_level_context")
        if isinstance(context, dict):
            return context
        compact_summary = extract_named_json_value(text, "compact_input_summary")
        if isinstance(compact_summary, dict):
            return water_level_context_from_compact_summary(compact_summary)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        context = payload.get("water_level_context")
        if isinstance(context, dict):
            return context
        compact_summary = payload.get("compact_input_summary")
        if isinstance(compact_summary, dict):
            return water_level_context_from_compact_summary(compact_summary)
    return {}


def gate_state_context_from_prompt_messages(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    for message in messages:
        if message.get("role") != "user":
            continue
        text = content_to_text(message.get("content", ""))
        context = extract_named_json_value(text, "gate_state_context")
        if isinstance(context, dict):
            return context
        compact_summary = extract_named_json_value(text, "compact_input_summary")
        if isinstance(compact_summary, dict):
            return gate_state_context_from_compact_summary(compact_summary)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        context = payload.get("gate_state_context")
        if isinstance(context, dict):
            return context
        compact_summary = payload.get("compact_input_summary")
        if isinstance(compact_summary, dict):
            return gate_state_context_from_compact_summary(compact_summary)
    return {}


def fusion_reasoning_context_from_prompt_messages(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    for message in messages:
        if message.get("role") != "user":
            continue
        text = content_to_text(message.get("content", ""))
        context = extract_named_json_value(text, "fusion_reasoning_context")
        if isinstance(context, dict):
            return context
        input_payload = extract_named_json_value(text, "input")
        if isinstance(input_payload, dict):
            return fusion_reasoning_context_from_input(input_payload)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        context = payload.get("fusion_reasoning_context")
        if isinstance(context, dict):
            return context
        input_payload = payload.get("input")
        if isinstance(input_payload, dict):
            return fusion_reasoning_context_from_input(input_payload)
    return {}


def ship_behavior_context_from_prompt_messages(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    for message in messages:
        if message.get("role") != "user":
            continue
        text = content_to_text(message.get("content", ""))
        context = extract_named_json_value(text, "ship_behavior_context")
        if isinstance(context, dict):
            return context
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        context = payload.get("ship_behavior_context")
        if isinstance(context, dict):
            return context
    return {}


def fusion_reasoning_context_from_input(input_payload: dict[str, Any]) -> dict[str, Any]:
    camera_layout = input_payload.get("camera_layout")
    if not isinstance(camera_layout, dict):
        return {}
    calibrated_cameras = normalize_string_list(
        camera_layout.get("calibrated_geometry_cameras")
    )
    state_cameras = normalize_string_list(
        camera_layout.get("uncalibrated_state_cameras")
    )
    if not calibrated_cameras and not state_cameras:
        return {}
    return {
        "use_calibrated_2d_3d_fusion": bool(calibrated_cameras),
        "calibrated_cameras": calibrated_cameras,
        "state_cameras_without_geometry": state_cameras,
    }


def extract_named_json_value(text: str, key: str) -> Any:
    marker = json.dumps(key, ensure_ascii=True) + ":"
    index = text.find(marker)
    if index < 0:
        return None
    start = index + len(marker)
    while start < len(text) and text[start].isspace():
        start += 1
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value


def water_level_context_from_compact_summary(
    compact_summary: dict[str, Any],
) -> dict[str, Any]:
    current_state = compact_summary.get("current_state_from_last_input_frame")
    if not isinstance(current_state, dict):
        current_state = {}
    sequence = compact_summary.get("input_lock_state_sequence")
    if not isinstance(sequence, list):
        sequence = []
    water_level_sequence = [
        {
            "frame_index": state.get("frame_index"),
            "relative_time_sec": state.get("relative_time_sec"),
            "water_state": state.get("water_state"),
            "water_level": state.get("water_level"),
        }
        for state in sequence
        if isinstance(state, dict) and is_number(state.get("water_level"))
    ]
    levels = [state["water_level"] for state in water_level_sequence]
    current_water_level = current_state.get("water_level")
    return {
        "current_water_level": current_water_level,
        "current_water_state": current_state.get("water_state"),
        "current_water_level_available": is_number(current_water_level),
        "input_water_level_sequence": water_level_sequence,
        "input_water_level_range": [min(levels), max(levels)] if levels else None,
        "input_water_level_delta": compact_summary.get("input_water_level_delta"),
        "stable_input_water_level": bool(levels)
        and max(levels) - min(levels) <= 1e-6,
    }


def gate_state_context_from_compact_summary(
    compact_summary: dict[str, Any],
) -> dict[str, Any]:
    current_state = compact_summary.get("current_state_from_last_input_frame")
    if not isinstance(current_state, dict):
        current_state = {}
    sequence = compact_summary.get("input_lock_state_sequence")
    if not isinstance(sequence, list):
        sequence = []
    gate_sequence = [
        {
            "frame_index": state.get("frame_index"),
            "relative_time_sec": state.get("relative_time_sec"),
            "upper_gate_state": state.get("upper_gate_state"),
            "lower_gate_state": state.get("lower_gate_state"),
            "water_state": state.get("water_state"),
        }
        for state in sequence
        if isinstance(state, dict)
    ]
    current = {
        key: current_state.get(key)
        for key in ("upper_gate_state", "lower_gate_state", "water_state")
        if current_state.get(key) is not None
    }
    stable_input_gate_state = bool(gate_sequence) and all(
        state.get("upper_gate_state") == current.get("upper_gate_state")
        and state.get("lower_gate_state") == current.get("lower_gate_state")
        for state in gate_sequence
    )
    stable_input_water_state = bool(gate_sequence) and all(
        state.get("water_state") == current.get("water_state")
        for state in gate_sequence
    )
    return {
        "current_state": current,
        "input_gate_state_sequence": gate_sequence,
        "stable_input_gate_state": stable_input_gate_state,
        "stable_input_water_state": stable_input_water_state,
        "stable_input_lock_state": (
            stable_input_gate_state
            and stable_input_water_state
            and current.get("water_state") == "idle"
        ),
    }


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def parse_json_object(text: str) -> dict[str, Any]:
    parsed = json.loads(extract_json_text(text))
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    return stripped[start : end + 1]


def load_adapter_model(
    cfg: dict[str, Any],
    adapter_dir: Path,
) -> Any:
    base_model = load_model(cfg)
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model.eval()
    return model


def generate_one(
    model: Any,
    processor: Any,
    item: dict[str, Any],
    max_new_tokens: int,
    use_water_level_guard: bool = False,
    use_gate_state_guard: bool = False,
    use_fusion_reasoning_guard: bool = False,
    use_ship_behavior_guard: bool = False,
    generation_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    encoded = processor.apply_chat_template(
        item["prompt_messages"],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    encoded = move_to_model_device(encoded, model)
    input_len = int(encoded["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            **(generation_kwargs or {}),
        )
    generated_text = processor.tokenizer.decode(
        generated[0, input_len:],
        skip_special_tokens=True,
    )
    raw_prediction = try_parse_json_object(generated_text)
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
        "schema_check": schema_check(parsed_prediction, item["reference"]),
        "semantic_check": semantic_check(parsed_prediction, item["reference"]),
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


def generate_one_with_invalid_json_retries(
    model: Any,
    processor: Any,
    item: dict[str, Any],
    max_new_tokens: int,
    max_invalid_json_retries: int,
    invalid_json_retry_repetition_penalty: Optional[float],
    use_water_level_guard: bool = False,
    use_gate_state_guard: bool = False,
    use_fusion_reasoning_guard: bool = False,
    use_ship_behavior_guard: bool = False,
    ban_exclamation_token: bool = False,
) -> dict[str, Any]:
    attempts = []
    result: Optional[dict[str, Any]] = None
    total_attempts = max(1, max_invalid_json_retries + 1)
    for attempt_index in range(total_attempts):
        generation_kwargs = {}
        if attempt_index > 0:
            if (
                invalid_json_retry_repetition_penalty is not None
                and invalid_json_retry_repetition_penalty != 1.0
            ):
                generation_kwargs["repetition_penalty"] = (
                    invalid_json_retry_repetition_penalty
                )
        if attempt_index > 0 or ban_exclamation_token:
            bad_words_ids = repeated_punctuation_bad_words_ids(processor.tokenizer)
            if bad_words_ids:
                generation_kwargs["bad_words_ids"] = bad_words_ids
                generation_kwargs["renormalize_logits"] = True
        result = generate_one(
            model=model,
            processor=processor,
            item=item,
            max_new_tokens=max_new_tokens,
            use_water_level_guard=use_water_level_guard,
            use_gate_state_guard=use_gate_state_guard,
            use_fusion_reasoning_guard=use_fusion_reasoning_guard,
            use_ship_behavior_guard=use_ship_behavior_guard,
            generation_kwargs=generation_kwargs,
        )
        valid_json = bool(result.get("schema_check", {}).get("valid_json"))
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "valid_json": valid_json,
                "repetition_penalty": generation_kwargs.get("repetition_penalty"),
                "bad_words_ids": bool(generation_kwargs.get("bad_words_ids")),
                "prediction_text_prefix": (result.get("prediction_text") or "")[:80],
            }
        )
        if valid_json:
            break
    if result is None:
        raise RuntimeError("generation produced no result")
    if max_invalid_json_retries > 0:
        result["invalid_json_retry"] = {
            "max_retries": max_invalid_json_retries,
            "retries_used": len(attempts) - 1,
            "attempts": attempts,
        }
    return result


def repeated_punctuation_bad_words_ids(tokenizer: Any) -> list[list[int]]:
    bad_words: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for text in ("!", " !", "!!", "!!!", "\n!"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        key = tuple(int(token_id) for token_id in token_ids)
        if key in seen:
            continue
        seen.add(key)
        bad_words.append(list(key))
    return bad_words


def apply_water_level_guard(
    prediction: Optional[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "changes": [],
    }
    current_water_level = context.get("current_water_level")
    if (
        prediction is None
        or not isinstance(context, dict)
        or not is_number(current_water_level)
    ):
        return prediction, report

    repaired = copy.deepcopy(prediction)
    changes: list[dict[str, Any]] = []

    def set_if_present(path: str, value: Any) -> None:
        parent_path, key = path.rsplit(".", 1) if "." in path else ("", path)
        parent = get_path(repaired, parent_path) if parent_path else repaired
        if not isinstance(parent, dict) or key not in parent:
            return
        old = parent.get(key)
        if old == value:
            return
        parent[key] = value
        changes.append({"path": path, "old": old, "new": value})

    set_if_present("current_state.water_level", current_water_level)

    current_water_state = get_path(repaired, "current_state.water_state") or context.get(
        "current_water_state"
    )
    set_if_present("water_surface_dynamics.current_water_state", current_water_state)
    water_surface = repaired.get("water_surface_dynamics")
    if (
        "future_state_10s" not in repaired
        and isinstance(water_surface, dict)
        and current_water_state is not None
    ):
        old = water_surface.get("current_water_state")
        if old != current_water_state:
            water_surface["current_water_state"] = current_water_state
            changes.append(
                {
                    "path": "water_surface_dynamics.current_water_state",
                    "old": old,
                    "new": current_water_state,
                }
            )
    future_water_state = (
        get_path(repaired, "future_state_10s.water_state")
        or get_path(repaired, "water_surface_dynamics.target_water_state")
    )
    if (
        context.get("stable_input_water_level") is True
        and current_water_state == "idle"
        and future_water_state == "idle"
    ):
        set_if_present("future_state_10s.water_level", current_water_level)
        set_if_present("future_water_level_delta", 0.0)
        set_if_present(
            "water_surface_dynamics.water_level_delta_from_last_input_to_target",
            0.0,
        )

    report["changes"] = changes
    report["applied"] = bool(changes)
    return (repaired if changes else prediction), report


def apply_gate_state_guard(
    prediction: Optional[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "changes": [],
    }
    if prediction is None or not isinstance(context, dict):
        return prediction, report
    current_context = context.get("current_state")
    if not isinstance(current_context, dict):
        return prediction, report

    repaired = copy.deepcopy(prediction)
    changes: list[dict[str, Any]] = []

    def set_if_present(path: str, value: Any) -> None:
        if value is None:
            return
        parent_path, key = path.rsplit(".", 1) if "." in path else ("", path)
        parent = get_path(repaired, parent_path) if parent_path else repaired
        if not isinstance(parent, dict) or key not in parent:
            return
        old = parent.get(key)
        if old == value:
            return
        parent[key] = value
        changes.append({"path": path, "old": old, "new": value})

    for key in ("upper_gate_state", "lower_gate_state", "water_state"):
        set_if_present(f"current_state.{key}", current_context.get(key))

    report["changes"] = changes
    report["applied"] = bool(changes)
    return (repaired if changes else prediction), report


def apply_fusion_reasoning_guard(
    prediction: Optional[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "changes": [],
    }
    if prediction is None or not isinstance(context, dict):
        return prediction, report
    expected = {
        "use_calibrated_2d_3d_fusion": context.get("use_calibrated_2d_3d_fusion"),
        "calibrated_cameras": normalize_string_list(
            context.get("calibrated_cameras")
        ),
        "state_cameras_without_geometry": normalize_string_list(
            context.get("state_cameras_without_geometry")
        ),
    }
    if expected["use_calibrated_2d_3d_fusion"] is None:
        return prediction, report
    if not expected["calibrated_cameras"] and not expected["state_cameras_without_geometry"]:
        return prediction, report

    repaired = copy.deepcopy(prediction)
    fusion = repaired.get("fusion_reasoning")
    if not isinstance(fusion, dict):
        old_value = fusion
        fusion = {}
        repaired["fusion_reasoning"] = fusion
        report["changes"].append(
            {"path": "fusion_reasoning", "old": old_value, "new": {}}
        )

    for key, value in expected.items():
        old_value = fusion.get(key)
        if old_value == value:
            continue
        fusion[key] = value
        report["changes"].append(
            {"path": f"fusion_reasoning.{key}", "old": old_value, "new": value}
        )

    report["applied"] = bool(report["changes"])
    return (repaired if report["changes"] else prediction), report


def apply_ship_behavior_guard(
    prediction: Optional[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "changes": [],
    }
    if prediction is None:
        return prediction, report

    repaired = copy.deepcopy(prediction)
    ship_behavior = repaired.get("ship_behavior")
    if isinstance(ship_behavior, dict):
        canonicalize_ship_behavior_categories(ship_behavior, report["changes"])
        apply_ship_behavior_context_categories(
            ship_behavior,
            context,
            report["changes"],
        )

    evidence = (
        context.get("input_mooring_evidence_counts")
        if isinstance(context, dict)
        else None
    )
    if not isinstance(evidence, dict) or not evidence:
        report["applied"] = bool(report["changes"])
        return (repaired if report["changes"] else prediction), report

    if not isinstance(ship_behavior, dict):
        old_value = ship_behavior
        ship_behavior = {}
        repaired["ship_behavior"] = ship_behavior
        report["changes"].append(
            {"path": "ship_behavior", "old": old_value, "new": {}}
        )
    mooring = ship_behavior.get("mooring_or_berthing_confidence_evidence")
    if not isinstance(mooring, dict):
        old_value = mooring
        mooring = {}
        ship_behavior["mooring_or_berthing_confidence_evidence"] = mooring
        report["changes"].append(
            {
                "path": "ship_behavior.mooring_or_berthing_confidence_evidence",
                "old": old_value,
                "new": {},
            }
        )

    for key, value in evidence.items():
        old_value = mooring.get(key)
        if old_value == value:
            continue
        mooring[key] = value
        report["changes"].append(
            {
                "path": (
                    "ship_behavior.mooring_or_berthing_confidence_evidence."
                    f"{key}"
                ),
                "old": old_value,
                "new": value,
            }
        )

    report["applied"] = bool(report["changes"])
    return (repaired if report["changes"] else prediction), report


def canonicalize_ship_behavior_categories(
    ship_behavior: dict[str, Any],
    changes: list[dict[str, Any]],
) -> None:
    intentions = ship_behavior.get("ship_intentions")
    if not isinstance(intentions, list):
        return
    for index, item in enumerate(intentions):
        if not isinstance(item, dict):
            continue
        old_category = item.get("category")
        new_category = canonical_ship_category(old_category)
        if new_category == old_category:
            continue
        item["category"] = new_category
        changes.append(
            {
                "path": f"ship_behavior.ship_intentions[{index}].category",
                "old": old_category,
                "new": new_category,
            }
        )


def apply_ship_behavior_context_categories(
    ship_behavior: dict[str, Any],
    context: dict[str, Any],
    changes: list[dict[str, Any]],
) -> None:
    expected_by_token = ship_category_by_token_from_context(context)
    if not expected_by_token:
        return
    intentions = ship_behavior.get("ship_intentions")
    if not isinstance(intentions, list):
        return
    for index, item in enumerate(intentions):
        if not isinstance(item, dict):
            continue
        token = item.get("instance_token")
        if token is None:
            continue
        expected_category = expected_by_token.get(str(token))
        if expected_category is None:
            continue
        old_category = item.get("category")
        if old_category == expected_category:
            continue
        item["category"] = expected_category
        changes.append(
            {
                "path": f"ship_behavior.ship_intentions[{index}].category",
                "old": old_category,
                "new": expected_category,
                "source": "ship_behavior_context.latest_ship_instances",
            }
        )


def ship_category_by_token_from_context(context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    items = context.get("latest_ship_instances")
    if not isinstance(items, list):
        return {}
    out = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        token = item.get("instance_token")
        if token is None:
            continue
        category = canonical_ship_category(item.get("category"))
        if category is None:
            continue
        out[str(token)] = category
    return out


def canonical_ship_category(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return NAVLOCK_SHIP_CATEGORY_BY_LOWER.get(value.lower(), value)


def load_prediction_results(
    path: Path,
    context_by_id: Optional[dict[str, dict[str, Any]]] = None,
    use_water_level_guard: bool = False,
    use_gate_state_guard: bool = False,
    use_fusion_reasoning_guard: bool = False,
    use_ship_behavior_guard: bool = False,
) -> list[dict[str, Any]]:
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reference = row.get("reference", {})
        prediction = row.get("prediction_json")
        contexts = (context_by_id or {}).get(row.get("id"), {})
        raw_prediction = prediction
        water_report = None
        gate_report = None
        fusion_report = None
        ship_report = None
        if use_water_level_guard:
            prediction, water_report = apply_water_level_guard(
                prediction,
                contexts.get("water_level_context", {}),
            )
        if use_gate_state_guard:
            prediction, gate_report = apply_gate_state_guard(
                prediction,
                contexts.get("gate_state_context", {}),
            )
        if use_fusion_reasoning_guard:
            prediction, fusion_report = apply_fusion_reasoning_guard(
                prediction,
                contexts.get("fusion_reasoning_context", {}),
            )
        if use_ship_behavior_guard:
            prediction, ship_report = apply_ship_behavior_guard(
                prediction,
                contexts.get("ship_behavior_context", {}),
            )
        if (
            use_water_level_guard
            or use_gate_state_guard
            or use_fusion_reasoning_guard
            or use_ship_behavior_guard
        ):
            row.setdefault("prediction_json_raw", raw_prediction)
            row["prediction_json"] = prediction
        if use_water_level_guard:
            row["water_level_guard"] = water_report
        if use_gate_state_guard:
            row["gate_state_guard"] = gate_report
        if use_fusion_reasoning_guard:
            row["fusion_reasoning_guard"] = fusion_report
        if use_ship_behavior_guard:
            row["ship_behavior_guard"] = ship_report
        row["schema_check"] = schema_check(prediction, reference)
        row["semantic_check"] = semantic_check(prediction, reference)
        results.append(row)
    if not results:
        raise ValueError(f"no predictions loaded from {path}")
    return results


def move_to_model_device(batch: dict[str, Any], model: Any) -> dict[str, Any]:
    device = next(model.parameters()).device
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def try_parse_json_object(text: str) -> Optional[dict[str, Any]]:
    try:
        return parse_json_object(text)
    except (json.JSONDecodeError, ValueError):
        return None


def schema_check(
    prediction: Optional[dict[str, Any]],
    reference: dict[str, Any],
) -> dict[str, Any]:
    if prediction is None:
        return {
            "valid_json": False,
            "missing_top_level_keys": sorted(reference.keys()),
            "extra_top_level_keys": [],
            "missing_nested_paths": sorted(flatten_schema_paths(reference)),
            "extra_nested_paths": [],
            "type_mismatch_paths": [],
        }
    reference_keys = set(reference.keys())
    prediction_keys = set(prediction.keys())
    nested = compare_nested_schema(prediction, reference)
    return {
        "valid_json": True,
        "missing_top_level_keys": sorted(reference_keys - prediction_keys),
        "extra_top_level_keys": sorted(prediction_keys - reference_keys),
        **nested,
    }


def semantic_check(
    prediction: Optional[dict[str, Any]],
    reference: dict[str, Any],
) -> dict[str, Any]:
    state_paths = [
        "current_state.upper_gate_state",
        "current_state.lower_gate_state",
        "current_state.water_state",
        "future_state_10s.upper_gate_state",
        "future_state_10s.lower_gate_state",
        "future_state_10s.water_state",
        "water_surface_dynamics.target_water_state",
        "water_surface_dynamics.current_water_state",
    ]
    numeric_paths = [
        "current_state.water_level",
        "current_water_level_delta_from_first_selected_frame",
        "future_state_10s.water_level",
        "future_water_level_delta",
        "water_surface_dynamics.water_level_delta_from_last_input_to_target",
        "water_surface_dynamics.water_level_delta_from_first_selected_to_current",
    ]
    if prediction is None:
        prediction = {}
        return {
            "valid_json": False,
            "state_matches": {
                path: False
                for path in state_paths
                if has_path(reference, path)
            },
            "water_level_absolute_errors": {},
            "ship_behavior": ship_behavior_check(
                prediction.get("ship_behavior"),
                reference.get("ship_behavior"),
            ),
            "fusion_reasoning": fusion_reasoning_check(
                prediction.get("fusion_reasoning"),
                reference.get("fusion_reasoning"),
            ),
        }
    return {
        "valid_json": True,
        "state_matches": {
            path: get_path(prediction, path) == get_path(reference, path)
            for path in state_paths
            if has_path(reference, path)
        },
        "water_level_absolute_errors": {
            path: abs(float(get_path(prediction, path)) - float(get_path(reference, path)))
            for path in numeric_paths
            if is_number(get_path(prediction, path)) and is_number(get_path(reference, path))
        },
        "ship_behavior": ship_behavior_check(
            prediction.get("ship_behavior"),
            reference.get("ship_behavior"),
        ),
        "fusion_reasoning": fusion_reasoning_check(
            prediction.get("fusion_reasoning"),
            reference.get("fusion_reasoning"),
        ),
    }


def ship_behavior_check(
    prediction: Any,
    reference: Any,
) -> dict[str, Any]:
    if not isinstance(reference, dict):
        return empty_ship_behavior_check()
    if not isinstance(prediction, dict):
        prediction = {}
    result = {
        "ship_intentions": ship_intentions_check(
            prediction.get("ship_intentions"),
            reference.get("ship_intentions"),
        ),
        "mooring_evidence": mooring_evidence_check(
            prediction.get("mooring_or_berthing_confidence_evidence"),
            reference.get("mooring_or_berthing_confidence_evidence"),
        ),
    }
    result["has_reference"] = True
    return result


def empty_ship_behavior_check() -> dict[str, Any]:
    return {
        "has_reference": False,
        "ship_intentions": {
            "exact_items_match": False,
            "reference_count": 0,
            "prediction_count": 0,
            "matched_exact_items": 0,
            "reference_exact_items": 0,
            "prediction_exact_items": 0,
            "matched_instance_tokens": 0,
            "reference_instance_tokens": 0,
            "prediction_instance_tokens": 0,
            "matched_instance_intentions": 0,
            "reference_instance_intentions": 0,
            "prediction_instance_intentions": 0,
            "reference_intention_label_counts": {},
            "prediction_intention_label_counts": {},
            "matched_intention_label_counts": {},
        },
        "mooring_evidence": {
            "numeric_absolute_errors": {},
            "boolean_matches": {},
        },
    }


def ship_intentions_check(prediction: Any, reference: Any) -> dict[str, Any]:
    reference_items = normalize_ship_intention_items(reference)
    prediction_items = normalize_ship_intention_items(prediction)
    reference_exact = {ship_intention_exact_key(item) for item in reference_items}
    prediction_exact = {ship_intention_exact_key(item) for item in prediction_items}
    reference_tokens = {
        str(item["instance_token"])
        for item in reference_items
        if item.get("instance_token") is not None
    }
    prediction_tokens = {
        str(item["instance_token"])
        for item in prediction_items
        if item.get("instance_token") is not None
    }
    reference_pairs = ship_intention_token_label_pairs(reference_items)
    prediction_pairs = ship_intention_token_label_pairs(prediction_items)
    matched_pairs = reference_pairs & prediction_pairs
    return {
        "exact_items_match": reference_exact == prediction_exact,
        "reference_count": len(reference_items),
        "prediction_count": len(prediction_items),
        "matched_exact_items": len(reference_exact & prediction_exact),
        "reference_exact_items": len(reference_exact),
        "prediction_exact_items": len(prediction_exact),
        "matched_instance_tokens": len(reference_tokens & prediction_tokens),
        "reference_instance_tokens": len(reference_tokens),
        "prediction_instance_tokens": len(prediction_tokens),
        "matched_instance_intentions": len(matched_pairs),
        "reference_instance_intentions": len(reference_pairs),
        "prediction_instance_intentions": len(prediction_pairs),
        "reference_intention_label_counts": dict(ship_intention_label_counts(reference_items)),
        "prediction_intention_label_counts": dict(ship_intention_label_counts(prediction_items)),
        "matched_intention_label_counts": dict(Counter(label for _, label in matched_pairs)),
    }


def normalize_ship_intention_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if not isinstance(item, dict):
            continue
        intentions = item.get("ship_intentions")
        if not isinstance(intentions, list):
            intentions = []
        items.append(
            {
                "instance_token": item.get("instance_token"),
                "category": canonical_ship_category(item.get("category")),
                "ship_intentions": [
                    str(intent)
                    for intent in intentions
                    if intent is not None
                ],
            }
        )
    return items


def ship_intention_exact_key(item: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(item.get("instance_token")),
        ship_intention_category_eval_key(item.get("category")),
        tuple(sorted(item.get("ship_intentions") or [])),
    )


def ship_intention_category_eval_key(value: Any) -> str:
    category = canonical_ship_category(value)
    if isinstance(category, str):
        return NAVLOCK_SHIP_EVAL_CATEGORY_ALIASES.get(category, category)
    return str(category)


def ship_intention_token_label_pairs(items: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs = set()
    for item in items:
        token = item.get("instance_token")
        if token is None:
            continue
        for label in item.get("ship_intentions") or []:
            pairs.add((str(token), str(label)))
    return pairs


def ship_intention_label_counts(items: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in items:
        for label in item.get("ship_intentions") or []:
            counts[str(label)] += 1
    return counts


def mooring_evidence_check(prediction: Any, reference: Any) -> dict[str, Any]:
    if not isinstance(reference, dict):
        return {"numeric_absolute_errors": {}, "boolean_matches": {}}
    if not isinstance(prediction, dict):
        prediction = {}
    numeric_keys = [
        "crew_count_2d",
        "mooring_line_count_2d",
        "ship_count_2d",
        "ship_count_3d",
    ]
    boolean_keys = ["mooring_confidence_boost_present"]
    return {
        "numeric_absolute_errors": {
            key: abs(float(prediction[key]) - float(reference[key]))
            for key in numeric_keys
            if is_number(prediction.get(key)) and is_number(reference.get(key))
        },
        "boolean_matches": {
            key: prediction.get(key) == reference.get(key)
            for key in boolean_keys
            if key in reference
        },
    }


def fusion_reasoning_check(
    prediction: Any,
    reference: Any,
) -> dict[str, Any]:
    if not isinstance(reference, dict):
        return empty_fusion_reasoning_check()
    if not isinstance(prediction, dict):
        prediction = {}
    boolean_keys = ["use_calibrated_2d_3d_fusion"]
    camera_list_keys = ["calibrated_cameras", "state_cameras_without_geometry"]
    return {
        "has_reference": True,
        "boolean_matches": {
            key: prediction.get(key) == reference.get(key)
            for key in boolean_keys
            if key in reference
        },
        "camera_list_checks": {
            key: camera_list_check(prediction.get(key), reference.get(key))
            for key in camera_list_keys
            if key in reference
        },
    }


def empty_fusion_reasoning_check() -> dict[str, Any]:
    return {
        "has_reference": False,
        "boolean_matches": {},
        "camera_list_checks": {},
    }


def camera_list_check(prediction: Any, reference: Any) -> dict[str, Any]:
    reference_values = normalize_string_list(reference)
    prediction_values = normalize_string_list(prediction)
    reference_set = set(reference_values)
    prediction_set = set(prediction_values)
    return {
        "exact_order_match": prediction_values == reference_values,
        "matched": len(reference_set & prediction_set),
        "reference": len(reference_set),
        "prediction": len(prediction_set),
    }


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def has_path(value: Any, path: str) -> bool:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_nested_schema(
    prediction: Any,
    reference: Any,
    path: str = "",
) -> dict[str, list[str]]:
    missing_paths: list[str] = []
    extra_paths: list[str] = []
    type_mismatch_paths: list[str] = []

    if isinstance(reference, dict):
        if not isinstance(prediction, dict):
            type_mismatch_paths.append(path or "$")
            missing_paths.extend(flatten_schema_paths(reference, path))
            return {
                "missing_nested_paths": sorted(missing_paths),
                "extra_nested_paths": sorted(extra_paths),
                "type_mismatch_paths": sorted(type_mismatch_paths),
            }

        reference_keys = set(reference.keys())
        prediction_keys = set(prediction.keys())
        for key in sorted(reference_keys - prediction_keys):
            child_path = join_schema_path(path, key)
            missing_paths.append(child_path)
            missing_paths.extend(flatten_schema_paths(reference[key], child_path))
        for key in sorted(prediction_keys - reference_keys):
            child_path = join_schema_path(path, key)
            extra_paths.append(child_path)
            extra_paths.extend(flatten_schema_paths(prediction[key], child_path))
        for key in sorted(reference_keys & prediction_keys):
            child = compare_nested_schema(
                prediction[key],
                reference[key],
                join_schema_path(path, key),
            )
            missing_paths.extend(child["missing_nested_paths"])
            extra_paths.extend(child["extra_nested_paths"])
            type_mismatch_paths.extend(child["type_mismatch_paths"])
    elif isinstance(reference, list):
        if not isinstance(prediction, list):
            type_mismatch_paths.append(path or "$")
        elif reference and prediction:
            child = compare_nested_schema(
                prediction[0],
                reference[0],
                f"{path}[]" if path else "[]",
            )
            missing_paths.extend(child["missing_nested_paths"])
            extra_paths.extend(child["extra_nested_paths"])
            type_mismatch_paths.extend(child["type_mismatch_paths"])
    elif type(prediction) is not type(reference):
        type_mismatch_paths.append(path or "$")

    return {
        "missing_nested_paths": sorted(set(missing_paths)),
        "extra_nested_paths": sorted(set(extra_paths)),
        "type_mismatch_paths": sorted(set(type_mismatch_paths)),
    }


def flatten_schema_paths(value: Any, path: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = join_schema_path(path, key)
            paths.append(child_path)
            paths.extend(flatten_schema_paths(child, child_path))
    elif isinstance(value, list) and value:
        child_path = f"{path}[]" if path else "[]"
        paths.append(child_path)
        paths.extend(flatten_schema_paths(value[0], child_path))
    return paths


def join_schema_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid_json = sum(1 for item in results if item["schema_check"]["valid_json"])
    exact_top_level_schema = sum(
        1
        for item in results
        if item["schema_check"]["valid_json"]
        and not item["schema_check"]["missing_top_level_keys"]
    )
    exact_nested_schema = sum(
        1
        for item in results
        if item["schema_check"]["valid_json"]
        and not item["schema_check"]["missing_top_level_keys"]
        and not item["schema_check"]["extra_top_level_keys"]
        and not item["schema_check"]["missing_nested_paths"]
        and not item["schema_check"]["extra_nested_paths"]
        and not item["schema_check"]["type_mismatch_paths"]
    )
    semantic_summary = summarize_semantics(results)
    return {
        "num_samples": len(results),
        "valid_json": valid_json,
        "exact_top_level_schema": exact_top_level_schema,
        "exact_nested_schema": exact_nested_schema,
        **semantic_summary,
    }


def invalid_json_ids(results: list[dict[str, Any]]) -> list[str]:
    invalid = []
    for index, item in enumerate(results):
        schema = item.get("schema_check")
        if isinstance(schema, dict):
            valid_json = bool(schema.get("valid_json"))
        else:
            valid_json = isinstance(item.get("prediction_json"), dict)
        if not valid_json:
            invalid.append(str(item.get("id") or f"row_{index}"))
    return invalid


def require_valid_json_results(
    results: list[dict[str, Any]],
    output_path: Optional[Path] = None,
) -> None:
    invalid = invalid_json_ids(results)
    if not invalid:
        return
    shown = ", ".join(invalid[:20])
    if len(invalid) > 20:
        shown += f", ... (+{len(invalid) - 20} more)"
    target = f"; refusing to write {output_path}" if output_path else ""
    raise SystemExit(
        f"invalid_json={len(invalid)}/{len(results)} ids=[{shown}]{target}"
    )


def summarize_semantics(results: list[dict[str, Any]]) -> dict[str, Any]:
    state_counts: dict[str, dict[str, int]] = {}
    numeric_errors: dict[str, list[float]] = {}
    ship_summary = init_ship_behavior_summary()
    fusion_summary = init_fusion_reasoning_summary()
    for item in results:
        check = item.get("semantic_check")
        if check is None:
            check = semantic_check(item.get("prediction_json"), item.get("reference", {}))
        for path, matched in check.get("state_matches", {}).items():
            counts = state_counts.setdefault(path, {"correct": 0, "total": 0})
            counts["total"] += 1
            if matched:
                counts["correct"] += 1
        for path, error in check.get("water_level_absolute_errors", {}).items():
            numeric_errors.setdefault(path, []).append(float(error))
        update_ship_behavior_summary(ship_summary, check.get("ship_behavior", {}))
        update_fusion_reasoning_summary(
            fusion_summary,
            check.get("fusion_reasoning", {}),
        )

    return {
        "state_semantic_matches": state_counts,
        "numeric_mae": {
            path: sum(errors) / len(errors)
            for path, errors in sorted(numeric_errors.items())
            if errors
        },
        "ship_behavior": finalize_ship_behavior_summary(ship_summary),
        "fusion_reasoning": finalize_fusion_reasoning_summary(fusion_summary),
    }


def init_ship_behavior_summary() -> dict[str, Any]:
    return {
        "ship_intentions_exact": {"correct": 0, "total": 0},
        "ship_intention_count_errors": [],
        "exact_items": {"matched": 0, "reference": 0, "prediction": 0},
        "instance_tokens": {"matched": 0, "reference": 0, "prediction": 0},
        "instance_intentions": {"matched": 0, "reference": 0, "prediction": 0},
        "intention_labels": {},
        "mooring_numeric_errors": {},
        "mooring_boolean_matches": {},
    }


def update_ship_behavior_summary(summary: dict[str, Any], check: dict[str, Any]) -> None:
    if not check.get("has_reference", False):
        return
    intentions = check.get("ship_intentions", {})
    exact = summary["ship_intentions_exact"]
    exact["total"] += 1
    if intentions.get("exact_items_match"):
        exact["correct"] += 1
    summary["ship_intention_count_errors"].append(
        abs(
            int(intentions.get("prediction_count", 0))
            - int(intentions.get("reference_count", 0))
        )
    )
    for source_key, summary_key in (
        ("matched_exact_items", "matched"),
        ("reference_exact_items", "reference"),
        ("prediction_exact_items", "prediction"),
    ):
        summary["exact_items"][summary_key] += int(intentions.get(source_key, 0))
    for source_key, summary_key in (
        ("matched_instance_tokens", "matched"),
        ("reference_instance_tokens", "reference"),
        ("prediction_instance_tokens", "prediction"),
    ):
        summary["instance_tokens"][summary_key] += int(intentions.get(source_key, 0))
    for source_key, summary_key in (
        ("matched_instance_intentions", "matched"),
        ("reference_instance_intentions", "reference"),
        ("prediction_instance_intentions", "prediction"),
    ):
        summary["instance_intentions"][summary_key] += int(intentions.get(source_key, 0))

    label_names = (
        set(intentions.get("reference_intention_label_counts", {}))
        | set(intentions.get("prediction_intention_label_counts", {}))
        | set(intentions.get("matched_intention_label_counts", {}))
    )
    for label in label_names:
        item = summary["intention_labels"].setdefault(
            label, {"matched": 0, "reference": 0, "prediction": 0}
        )
        item["matched"] += int(
            intentions.get("matched_intention_label_counts", {}).get(label, 0)
        )
        item["reference"] += int(
            intentions.get("reference_intention_label_counts", {}).get(label, 0)
        )
        item["prediction"] += int(
            intentions.get("prediction_intention_label_counts", {}).get(label, 0)
        )

    mooring = check.get("mooring_evidence", {})
    for key, error in mooring.get("numeric_absolute_errors", {}).items():
        summary["mooring_numeric_errors"].setdefault(key, []).append(float(error))
    for key, matched in mooring.get("boolean_matches", {}).items():
        item = summary["mooring_boolean_matches"].setdefault(
            key, {"correct": 0, "total": 0}
        )
        item["total"] += 1
        if matched:
            item["correct"] += 1


def finalize_ship_behavior_summary(summary: dict[str, Any]) -> dict[str, Any]:
    count_errors = summary["ship_intention_count_errors"]
    return {
        "ship_intentions_exact": summary["ship_intentions_exact"],
        "ship_intention_count_mae": (
            sum(count_errors) / len(count_errors) if count_errors else 0.0
        ),
        "exact_item_match": prf_from_counts(summary["exact_items"]),
        "instance_token_match": prf_from_counts(summary["instance_tokens"]),
        "instance_intention_match": prf_from_counts(summary["instance_intentions"]),
        "intention_label_match": {
            label: prf_from_counts(counts)
            for label, counts in sorted(summary["intention_labels"].items())
        },
        "mooring_evidence_numeric_mae": {
            key: sum(errors) / len(errors)
            for key, errors in sorted(summary["mooring_numeric_errors"].items())
            if errors
        },
        "mooring_evidence_boolean_matches": summary["mooring_boolean_matches"],
    }


def prf_from_counts(counts: dict[str, int]) -> dict[str, Any]:
    matched = int(counts.get("matched", 0))
    reference = int(counts.get("reference", 0))
    prediction = int(counts.get("prediction", 0))
    precision = matched / prediction if prediction else (1.0 if reference == 0 else 0.0)
    recall = matched / reference if reference else (1.0 if prediction == 0 else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "matched": matched,
        "reference": reference,
        "prediction": prediction,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def init_fusion_reasoning_summary() -> dict[str, Any]:
    return {
        "boolean_matches": {},
        "camera_lists": {},
    }


def update_fusion_reasoning_summary(
    summary: dict[str, Any],
    check: dict[str, Any],
) -> None:
    if not check.get("has_reference", False):
        return
    for key, matched in check.get("boolean_matches", {}).items():
        item = summary["boolean_matches"].setdefault(key, {"correct": 0, "total": 0})
        item["total"] += 1
        if matched:
            item["correct"] += 1
    for key, counts in check.get("camera_list_checks", {}).items():
        item = summary["camera_lists"].setdefault(
            key,
            {
                "exact_order_correct": 0,
                "exact_order_total": 0,
                "matched": 0,
                "reference": 0,
                "prediction": 0,
            },
        )
        item["exact_order_total"] += 1
        if counts.get("exact_order_match"):
            item["exact_order_correct"] += 1
        item["matched"] += int(counts.get("matched", 0))
        item["reference"] += int(counts.get("reference", 0))
        item["prediction"] += int(counts.get("prediction", 0))


def finalize_fusion_reasoning_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "boolean_matches": summary["boolean_matches"],
        "camera_list_match": {
            key: {
                "exact_order": {
                    "correct": counts["exact_order_correct"],
                    "total": counts["exact_order_total"],
                },
                **prf_from_counts(counts),
            }
            for key, counts in sorted(summary["camera_lists"].items())
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
