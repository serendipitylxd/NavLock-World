#!/usr/bin/env python3
"""Convert generic NavLock VLM semantic JSONL into Qwen3-VL chat JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
STATE_CAMERAS = ("CAM_3", "CAM_8")
CALIBRATED_CAMERAS = ("CAM_1", "CAM_2", "CAM_4", "CAM_5", "CAM_6", "CAM_7")
SHIP_2D_CLASSES = (
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Unladen_container_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
)
SHIP_3D_CLASSES = (
    "Fully_loaded_cargo_ship",
    "Fully_loaded_container_ship",
    "Unladen_cargo_ship",
    "Fully_loaded_cargo_fleet",
    "Unladen_cargo_fleet",
)
MOORING_CONFIDENCE_WEAK_RULE = (
    "Crew_member + Mooring_line + ship detection should increase confidence in "
    "berthed/moored behavior, but missing mooring lines must not rule it out "
    "because occlusion is common."
)
DEFAULT_TOP_LEVEL_KEYS = (
    "current_state",
    "future_state_10s",
    "future_water_level_delta",
    "water_surface_dynamics",
    "ship_behavior",
    "fusion_reasoning",
)
FORBIDDEN_TOP_LEVEL_KEYS = ("navlock_task", "output")
LOCK_STATE_KEYS = ("upper_gate_state", "lower_gate_state", "water_state", "water_level")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Generic VLM semantic JSONL file.")
    parser.add_argument("--output", required=True, help="Qwen3-VL chat JSONL file.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--image-policy",
        choices=("state_first", "all"),
        default="state_first",
        help=(
            "state_first keeps CAM_3/CAM_8 first, then calibrated cameras until "
            "--max-images is reached. all keeps original image order until the cap."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=12,
        help="Maximum images per sample. Keep small for RTX 4080 16GB LoRA/QLoRA.",
    )
    parser.add_argument(
        "--max-lidar-images",
        type=int,
        default=2,
        help=(
            "Maximum rendered LiDAR BEV/range-view images to reserve inside "
            "--max-images for state_first conversion."
        ),
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=65536,
        help="Per-image Qwen3-VL max_pixels. 65536 is 256x256.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to convert for smoke tests.",
    )
    parser.add_argument(
        "--include-annotation-ship-intention-context",
        action="store_true",
        help=(
            "Keep annotation-backed input ship_intentions in the user prompt. "
            "Default strips them so eval prompts cannot use dataset intent labels."
        ),
    )
    parser.add_argument(
        "--prompt-profile",
        choices=("standard", "compact_context_first"),
        default="standard",
        help=(
            "standard preserves the historical prompt. compact_context_first "
            "places non-leaky gate/water/ship contexts before schemas and omits "
            "the full input payload for frame-level evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_written = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if args.limit is not None and num_written >= args.limit:
                break
            item = json.loads(line)
            converted = convert_item(
                item,
                model=args.model,
                image_policy=args.image_policy,
                max_images=args.max_images,
                max_lidar_images=args.max_lidar_images,
                image_max_pixels=args.image_max_pixels,
                include_annotation_ship_intention_context=bool(
                    args.include_annotation_ship_intention_context
                ),
                prompt_profile=args.prompt_profile,
            )
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"wrote={output_path}")
    print(f"model={args.model}")
    print(f"num_items={num_written}")
    print(f"image_policy={args.image_policy}")
    print(f"max_images={args.max_images}")
    print(f"max_lidar_images={args.max_lidar_images}")
    print(f"image_max_pixels={args.image_max_pixels}")
    print(f"prompt_profile={args.prompt_profile}")


def convert_item(
    item: dict[str, Any],
    model: str,
    image_policy: str,
    max_images: int,
    image_max_pixels: int,
    max_lidar_images: int = 2,
    include_annotation_ship_intention_context: bool = False,
    prompt_profile: str = "standard",
) -> dict[str, Any]:
    item = prepare_prompt_item(
        item,
        include_annotation_ship_intention_context=include_annotation_ship_intention_context,
    )
    visual_inputs = select_visual_inputs(
        item,
        image_policy=image_policy,
        max_images=max_images,
        max_lidar_images=max_lidar_images,
    )
    user_content = [
        {
            "type": "image",
            "image": to_file_uri(visual["path"]),
            "max_pixels": image_max_pixels,
        }
        for visual in visual_inputs
    ]
    user_content.append(
        {
            "type": "text",
            "text": build_user_prompt(
                item,
                visual_inputs,
                prompt_profile=prompt_profile,
            ),
        }
    )

    return {
        "id": item["id"],
        "model": model,
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": json.dumps(item["answer"], ensure_ascii=False),
            },
        ],
        "metadata": {
            "source_task": item["task"],
            "split": item["split"],
            "scene_token": item["scene_token"],
            "scene_name": item["scene_name"],
            "sample_token": item.get("sample_token"),
            "timestamp": item.get("timestamp"),
            "timestamp_str": item.get("timestamp_str"),
            "current_frame_index": item.get("current_frame_index"),
            "selected_num_images": len(visual_inputs),
            "selected_num_lidar_images": sum(
                1 for visual in visual_inputs if visual["kind"].startswith("lidar")
            ),
            "image_policy": image_policy,
            "max_lidar_images": max_lidar_images,
            "image_max_pixels": image_max_pixels,
            "prompt_profile": prompt_profile,
            "source_has_lidar_paths": True,
            "source_has_lidar_rendered_views": bool(item.get("lidar_images")),
        },
    }


def prepare_prompt_item(
    item: dict[str, Any],
    *,
    include_annotation_ship_intention_context: bool = False,
) -> dict[str, Any]:
    if include_annotation_ship_intention_context:
        return item
    sanitized = copy.deepcopy(item)
    input_payload = sanitized.get("input")
    if isinstance(input_payload, dict):
        strip_annotation_ship_intentions_from_frames(input_payload.get("frames"))
        # Existing gate contexts were derived from annotation-backed ship
        # intentions in the generic builder. Drop them so the converter rebuilds
        # label-free gate transition context from observed lock states.
        input_payload.pop("gate_transition_context", None)
    return sanitized


def strip_annotation_ship_intentions_from_frames(frames: Any) -> None:
    if not isinstance(frames, list):
        return
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        instances = frame.get("ship_instances")
        if not isinstance(instances, list):
            continue
        for instance in instances:
            if isinstance(instance, dict):
                instance.pop("ship_intentions", None)


def select_visual_inputs(
    item: dict[str, Any],
    image_policy: str,
    max_images: int,
    max_lidar_images: int = 2,
) -> list[dict[str, Any]]:
    if max_images <= 0:
        return []
    if image_policy == "all":
        return _all_visual_inputs(item)[:max_images]

    frames = item["input"]["frames"]
    lidar_budget = min(max(0, int(max_lidar_images)), max(0, max_images - 2))
    lidar_inputs = _latest_lidar_visual_inputs(frames, lidar_budget)
    camera_budget = max_images - len(lidar_inputs)
    camera_inputs = _state_first_camera_visual_inputs(frames, camera_budget)
    return (camera_inputs[:2] + lidar_inputs + camera_inputs[2:])[:max_images]


def select_images(
    item: dict[str, Any],
    image_policy: str,
    max_images: int,
    max_lidar_images: int = 2,
) -> list[str]:
    return [
        visual["path"]
        for visual in select_visual_inputs(
            item,
            image_policy=image_policy,
            max_images=max_images,
            max_lidar_images=max_lidar_images,
        )
    ]


def _all_visual_inputs(item: dict[str, Any]) -> list[dict[str, Any]]:
    visual_inputs: list[dict[str, Any]] = []
    for index, path in enumerate(item.get("images", [])):
        visual_inputs.append({"path": path, "kind": "camera", "order_hint": index})
    for index, path in enumerate(item.get("lidar_images", [])):
        view_type = "range_view" if path.endswith("_range.png") else "bev"
        visual_inputs.append(
            {
                "path": path,
                "kind": f"lidar_{view_type}",
                "view_type": view_type,
                "order_hint": index,
            }
        )
    return visual_inputs


def _state_first_camera_visual_inputs(
    frames: list[dict[str, Any]],
    max_images: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_set: set[str] = set()

    def add_frame_channels(
        channels: tuple[str, ...],
        frame_order: list[dict[str, Any]],
    ) -> None:
        for frame in frame_order:
            for channel in channels:
                image = frame["images"].get(channel)
                if not image:
                    continue
                path = image["path"]
                if path in selected_set:
                    continue
                selected.append(
                    {
                        "path": path,
                        "kind": "camera",
                        "channel": channel,
                        "frame_index": frame.get("frame_index"),
                        "relative_time_sec": frame.get("relative_time_sec"),
                        "role": image.get("state_camera_role")
                        or image.get("camera_role"),
                    }
                )
                selected_set.add(path)
                if len(selected) >= max_images:
                    return

    latest_first = list(reversed(frames))
    add_frame_channels(STATE_CAMERAS, latest_first)
    if len(selected) >= max_images:
        return selected
    add_frame_channels(CALIBRATED_CAMERAS, latest_first)
    return selected


def _latest_lidar_visual_inputs(
    frames: list[dict[str, Any]],
    max_images: int,
) -> list[dict[str, Any]]:
    if max_images <= 0:
        return []
    selected: list[dict[str, Any]] = []
    for frame in reversed(frames):
        rendered = frame.get("lidar", {}).get("rendered_views") or {}
        frame_views = []
        for view_name in ("bev", "range_view"):
            path = rendered.get(view_name)
            if path:
                frame_views.append(
                    {
                        "path": path,
                        "kind": f"lidar_{view_name}",
                        "view_type": view_name,
                        "frame_index": frame.get("frame_index"),
                        "relative_time_sec": frame.get("relative_time_sec"),
                        "channel": frame.get("lidar", {}).get("channel", "LIDAR_TOP"),
                    }
                )
        if not frame_views:
            continue
        selected.extend(frame_views[: max_images - len(selected)])
        if len(selected) >= max_images:
            break
    return selected


def build_user_prompt(
    item: dict[str, Any],
    visual_inputs: list[dict[str, Any]],
    prompt_profile: str = "standard",
) -> str:
    frames = item.get("input", {}).get("frames", [])
    compact_input_summary = build_compact_input_summary(item)
    if prompt_profile == "compact_context_first":
        compact_input_summary = compact_context_first_input_summary(
            compact_input_summary
        )
        has_future_state = "future_state_10s" in item["answer"]
        payload = {
            "navlock_task": item["task"],
            "instruction": "Return the required NavLock JSON object.",
            "response_contract": build_response_contract(
                item["answer"],
                compact=True,
            ),
            "compact_response_template": build_compact_response_template(
                item["answer"]
            ),
            "ship_behavior_context": build_ship_behavior_context(frames),
            "water_level_context": build_water_level_context(compact_input_summary),
            "gate_state_context": build_gate_state_context(compact_input_summary),
            "fusion_reasoning_context": build_fusion_reasoning_context(item),
            "compact_input_summary": compact_input_summary,
            "selected_visual_inputs": _compact_prompt_visual_inputs(visual_inputs),
            "output_format": {
                "type": "json",
                "must_match_schema_of_training_answer": True,
                "required_top_level_keys": list(item["answer"].keys()),
                "forbidden_top_level_keys": list(FORBIDDEN_TOP_LEVEL_KEYS),
                "do_not_wrap_answer": True,
            },
        }
        if has_future_state:
            payload["gate_transition_context"] = compact_gate_transition_context(
                build_gate_transition_context(compact_input_summary)
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if prompt_profile != "standard":
        raise ValueError(f"unknown prompt_profile={prompt_profile}")
    payload = {
        "response_contract": build_response_contract(item["answer"]),
        "compact_response_template": build_compact_response_template(item["answer"]),
        "water_level_context": build_water_level_context(compact_input_summary),
        "gate_state_context": build_gate_state_context(compact_input_summary),
        "gate_transition_context": build_gate_transition_context(
            compact_input_summary
        ),
        "fusion_reasoning_context": build_fusion_reasoning_context(item),
        "ship_behavior_context": build_ship_behavior_context(frames),
        "compact_input_summary": compact_input_summary,
        "selected_visual_inputs": _prompt_visual_inputs(visual_inputs),
        "output_format": {
            "type": "json",
            "must_match_schema_of_training_answer": True,
            "required_top_level_keys": list(item["answer"].keys()),
            "forbidden_top_level_keys": list(FORBIDDEN_TOP_LEVEL_KEYS),
            "do_not_wrap_answer": True,
        },
        "navlock_task": item["task"],
        "instruction": item["instruction"],
        "input": item["input"],
        "response_schema_details": {
            "required_json_paths": build_required_json_paths(item["answer"]),
            "required_nested_schema": build_schema_outline(item["answer"]),
        },
        "image_usage_note": (
            f"The preceding {len(visual_inputs)} images follow selected_visual_inputs "
            "order. CAM_3/CAM_8 are prioritized for gate and water-surface evidence. "
            "Rendered LiDAR BEV/range-view images are visual summaries of LIDAR_TOP "
            "for geometric and ship-position context when included."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _prompt_visual_inputs(
    visual_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact = []
    for index, visual in enumerate(visual_inputs):
        item = {
            "image_index": index,
            "kind": visual.get("kind"),
        }
        for key in ("channel", "view_type", "frame_index", "relative_time_sec", "role"):
            if key in visual:
                item[key] = visual[key]
        compact.append(item)
    return compact


def _compact_prompt_visual_inputs(
    visual_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact = []
    for index, visual in enumerate(visual_inputs):
        item = {
            "i": index,
            "kind": visual.get("kind"),
        }
        for src_key, dst_key in (
            ("channel", "cam"),
            ("view_type", "view"),
            ("frame_index", "frame"),
            ("role", "role"),
        ):
            if src_key in visual:
                item[dst_key] = visual[src_key]
        compact.append(item)
    return compact


def build_response_contract(
    answer: dict[str, Any],
    *,
    compact: bool = False,
) -> dict[str, Any]:
    answer_keys = tuple(answer.keys())
    required_keys = answer_keys or DEFAULT_TOP_LEVEL_KEYS
    if compact:
        return {
            "must_output_raw_json_object": True,
            "required_top_level_keys_in_order": list(required_keys),
            "schema_critical_rules": [
                "Return only JSON; no wrapper keys.",
                "Keep every required top-level key in order.",
                "Use JSON null for null fields and [] for empty arrays.",
                "Copy current_state and water deltas from water_level_context and gate_state_context.",
                "Use gate_transition_context only for future gate checks.",
                "Copy fusion_reasoning_context camera lists exactly.",
                "For ship_behavior.ship_intentions, copy only non-empty labels from ship_behavior_context.latest_ship_instances.",
                "For mooring evidence, copy ship_behavior_context.input_mooring_evidence_counts exactly.",
                "Do not output ship_behavior_context as a top-level key.",
            ],
            "forbidden_top_level_keys": list(FORBIDDEN_TOP_LEVEL_KEYS),
            "must_keep_objects": [
                "water_surface_dynamics",
                "ship_behavior",
                "ship_behavior.mooring_or_berthing_confidence_evidence",
                "fusion_reasoning",
            ],
        }
    return {
        "must_output_raw_json_object": True,
        "required_top_level_keys_in_order": list(required_keys),
        "schema_critical_rules": [
            "Return only JSON; no wrapper keys.",
            "Keep every required top-level key in order, including scalar keys.",
            "Use JSON null for null fields.",
            "water_surface_dynamics.wave_annotation_source is a string; use \"none\", never null.",
            "Weak wave rule: filling uses source derived_from_water_state_target_region_rule, wave_expected true, camera CAM_3, region upper_gate_left_in_chamber.",
            "Weak wave rule: emptying uses source derived_from_water_state_target_region_rule, wave_expected true, camera CAM_8, region lower_gate_right_outside_chamber.",
            "Weak wave rule: idle uses source none, wave_expected false, and null target wave camera/region fields.",
            "Copy current_state.water_level from water_level_context.current_water_level; never use image/lidar/ship/time numbers for water_level.",
            "Copy current_state.upper_gate_state, current_state.lower_gate_state, and current_state.water_state from gate_state_context.current_state exactly.",
            "future_state_10s gate states are predictions and may differ from current_state; do not assume all gates are closed or blindly copy current gates.",
            "Use gate_transition_context.candidate_future_gate_checks for future_state_10s gate labels; open vs closing and closed vs opening are distinct labels.",
            "An open gate may transition to closing only when gate_transition_context.ship_berthing_status.all_labeled_ship_instances_berthed is true.",
            "If input shows opening_to_open completed, keep that gate open in future_state_10s unless gate_transition_context.ship_berthing_status.all_labeled_ship_instances_berthed is true.",
            "If gate_transition_context.future_gate_domain_rules contains a forced_future_label for a gate, copy that label into future_state_10s for that gate.",
            "For future_state_10s.upper_gate_state, inspect CAM_3 evidence and input upper-gate sequence before choosing open, closed, opening, or closing.",
            "For future_state_10s.lower_gate_state, inspect CAM_8 evidence and input lower-gate sequence before choosing open, closed, opening, or closing.",
            "If gate_state_context.input_gate_state_sequence is stable, water_state is idle, and no opening/closing transition cue is present, future_state_10s gate states usually retain gate_state_context.current_state.",
            "If water_level_context.stable_input_water_level is true and water_state stays idle, use future_water_level_delta 0.0 and keep future_state_10s.water_level equal to current_state.water_level.",
            "For fusion_reasoning, copy fusion_reasoning_context.use_calibrated_2d_3d_fusion, calibrated_cameras, and state_cameras_without_geometry exactly.",
            "For ship_behavior.ship_intentions, copy ship_behavior_context.latest_ship_instances instance_token, category, and ship_intentions only when non-empty ship_intentions are present in that context; otherwise use []. Do not use dataset annotation-backed ship_intentions from input frames.",
            "For ship_behavior.mooring_or_berthing_confidence_evidence, copy ship_behavior_context.input_mooring_evidence_counts exactly, including weak_rule.",
            "Do not output ship_behavior_context as a top-level key.",
            "Use [] for empty arrays.",
            "Do not create path-like keys.",
        ],
        "forbidden_top_level_keys": list(FORBIDDEN_TOP_LEVEL_KEYS),
        "must_keep_objects": [
            "water_surface_dynamics",
            "ship_behavior",
            "ship_behavior.mooring_or_berthing_confidence_evidence",
            "fusion_reasoning",
        ],
    }


def build_compact_response_template(answer: dict[str, Any]) -> dict[str, Any]:
    template = {
        "current_state": _pick_schema(answer.get("current_state", {})),
        "future_state_10s": _pick_schema(answer.get("future_state_10s", {})),
        "future_water_level_delta": build_schema_outline(
            answer.get("future_water_level_delta")
        ),
        "water_surface_dynamics": _pick_schema(
            answer.get("water_surface_dynamics", {})
        ),
        "ship_behavior": {
            "ship_intentions": [
                {
                    "instance_token": "string",
                    "category": "string",
                    "ship_intentions": ["string"],
                }
            ],
            "mooring_or_berthing_confidence_evidence": _pick_schema(
                answer.get("ship_behavior", {}).get(
                    "mooring_or_berthing_confidence_evidence", {}
                )
            ),
        },
        "fusion_reasoning": _pick_schema(answer.get("fusion_reasoning", {})),
    }
    return {
        key: template.get(key, build_schema_outline(answer[key]))
        for key in answer.keys()
    }


def _pick_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "string; use none when no source"
                if key == "wave_annotation_source"
                else build_schema_outline(child)
            )
            for key, child in value.items()
        }
    return build_schema_outline(value)


def build_compact_input_summary(item: dict[str, Any]) -> dict[str, Any]:
    input_payload = item.get("input", {})
    frames = input_payload.get("frames", [])
    lock_states = [
        {
            "frame_index": frame.get("frame_index"),
            "relative_time_sec": frame.get("relative_time_sec"),
            **frame.get("lock_state", {}),
        }
        for frame in frames
    ]
    last_frame = frames[-1] if frames else {}
    last_features = last_frame.get("flat_perception_features", {})
    water_levels = [
        state.get("water_level")
        for state in lock_states
        if isinstance(state.get("water_level"), (int, float))
    ]
    water_level_delta = (
        water_levels[-1] - water_levels[0] if len(water_levels) >= 2 else None
    )
    current_state = input_payload.get("current_state_from_last_input_frame")
    if not current_state and lock_states:
        current_state = {
            key: lock_states[-1][key]
            for key in LOCK_STATE_KEYS
            if key in lock_states[-1]
        }
    return {
        "temporal_setup": input_payload.get("temporal_setup", {}),
        "current_state_from_last_input_frame": current_state or {},
        "input_lock_state_sequence": lock_states,
        "gate_transition_context": input_payload.get("gate_transition_context", {}),
        "input_water_level_delta": water_level_delta,
        "last_frame_detection_counts": _compact_detection_counts(last_features),
        "detector_sources": input_payload.get("detector_sources", {}),
    }


def compact_context_first_input_summary(
    compact_input_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = copy.deepcopy(compact_input_summary)
    sequence = summary.get("input_lock_state_sequence")
    if isinstance(sequence, list):
        summary["input_lock_state_sequence"] = compact_lock_state_sequence(sequence)
    summary.pop("gate_transition_context", None)
    detector_sources = summary.get("detector_sources")
    if isinstance(detector_sources, dict):
        summary["detector_sources"] = {
            key: detector_sources[key]
            for key in ("2d", "3d")
            if key in detector_sources
        }
    return summary


def compact_lock_state_sequence(sequence: list[Any], max_items: int = 12) -> list[Any]:
    states = [state for state in sequence if isinstance(state, dict)]
    if len(states) <= max_items:
        return copy.deepcopy(states)
    selected_indices: list[int] = [0]
    previous = state_signature(states[0])
    for index, state in enumerate(states[1:], start=1):
        current = state_signature(state)
        if current != previous:
            if index - 1 not in selected_indices:
                selected_indices.append(index - 1)
            selected_indices.append(index)
            previous = current
    if len(states) - 1 not in selected_indices:
        selected_indices.append(len(states) - 1)
    while len(selected_indices) > max_items:
        selected_indices.pop(1)
    return [copy.deepcopy(states[index]) for index in selected_indices]


def state_signature(state: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        state.get("upper_gate_state"),
        state.get("lower_gate_state"),
        state.get("water_state"),
    )


def build_water_level_context(
    compact_input_summary: dict[str, Any],
) -> dict[str, Any]:
    current_state = compact_input_summary.get("current_state_from_last_input_frame")
    if not isinstance(current_state, dict):
        current_state = {}
    sequence = compact_input_summary.get("input_lock_state_sequence")
    if not isinstance(sequence, list):
        sequence = []
    levels = [
        state.get("water_level")
        for state in sequence
        if isinstance(state, dict)
        and isinstance(state.get("water_level"), (int, float))
    ]
    water_level_range = [min(levels), max(levels)] if levels else None
    current_water_level = current_state.get("water_level")
    return {
        "current_water_level": current_water_level,
        "current_water_state": current_state.get("water_state"),
        "input_water_level_range": water_level_range,
        "input_water_level_delta": compact_input_summary.get("input_water_level_delta"),
        "stable_input_water_level": (
            bool(levels) and max(levels) - min(levels) <= 1e-6
        ),
    }


def build_gate_state_context(
    compact_input_summary: dict[str, Any],
) -> dict[str, Any]:
    current_state = compact_input_summary.get("current_state_from_last_input_frame")
    if not isinstance(current_state, dict):
        current_state = {}
    sequence = compact_input_summary.get("input_lock_state_sequence")
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


def build_gate_transition_context(
    compact_input_summary: dict[str, Any],
) -> dict[str, Any]:
    existing = compact_input_summary.get("gate_transition_context")
    if isinstance(existing, dict) and existing:
        return compact_gate_transition_context(existing)

    current_state = compact_input_summary.get("current_state_from_last_input_frame")
    if not isinstance(current_state, dict):
        current_state = {}
    sequence = compact_input_summary.get("input_lock_state_sequence")
    if not isinstance(sequence, list):
        sequence = []
    gate_sequence = [
        {
            "frame_index": state.get("frame_index"),
            "relative_time_sec": state.get("relative_time_sec"),
            "upper_gate_state": state.get("upper_gate_state"),
            "lower_gate_state": state.get("lower_gate_state"),
            "water_state": state.get("water_state"),
            "water_level": state.get("water_level"),
        }
        for state in sequence
        if isinstance(state, dict)
    ]
    observed_transitions = observed_gate_transitions(gate_sequence)
    ship_status = {
        "ship_berthing_labels_available": False,
        "all_labeled_ship_instances_berthed": False,
        "gate_closing_precondition": (
            "An open gate may transition to closing only when all ships are berthed."
        ),
    }
    opening_hold_rules = opening_completed_hold_rules(
        current_state,
        observed_transitions,
        ship_status,
    )
    return {
        "state_camera_mapping": {
            "upper_gate_state": "CAM_3",
            "lower_gate_state": "CAM_8",
        },
        "observed_input_gate_transitions": observed_transitions,
        "ship_berthing_status": ship_status,
        "candidate_future_gate_checks": candidate_future_gate_checks(current_state),
        "future_gate_domain_rules": opening_hold_rules,
        "opening_completed_hold_rules": opening_hold_rules,
        "critical_label_pairs": [
            ["open", "closing"],
            ["closed", "opening"],
            ["opening", "open"],
            ["closing", "closed"],
        ],
        "retention_rule": (
            "Retain current gate labels only when the state camera shows no motion "
            "cue; after an input opening_to_open transition, keep open in the short "
            "future unless all labeled ships are ship_berthed."
        ),
    }


def compact_gate_transition_context(context: dict[str, Any]) -> dict[str, Any]:
    observed = context.get("observed_input_gate_transitions")
    candidates = context.get("candidate_future_gate_checks")
    return {
        "state_camera_mapping": context.get(
            "state_camera_mapping",
            {
                "upper_gate_state": "CAM_3",
                "lower_gate_state": "CAM_8",
            },
        ),
        "observed_input_gate_transitions": (
            observed[-4:] if isinstance(observed, list) else []
        ),
        "ship_berthing_status": context.get("ship_berthing_status", {}),
        "candidate_future_gate_checks": compact_candidate_future_gate_checks(
            candidates if isinstance(candidates, list) else []
        ),
        "future_gate_domain_rules": context.get("future_gate_domain_rules", []),
        "opening_completed_hold_rules": context.get(
            "opening_completed_hold_rules", []
        ),
        "critical_label_pairs": [
            ["open", "closing"],
            ["closed", "opening"],
            ["opening", "open"],
            ["closing", "closed"],
        ],
        "retention_rule": (
            "Retain current gate labels only when the state camera shows no motion "
            "cue; after an input opening_to_open transition, keep open in the short "
            "future unless all labeled ships are ship_berthed."
        ),
    }


def compact_candidate_future_gate_checks(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                key: item[key]
                for key in (
                    "gate",
                    "state_camera",
                    "current_label",
                    "competing_future_label",
                    "confusing_pair",
                    "open_to_closing_requires_all_ships_berthed",
                    "all_labeled_ship_instances_berthed",
                )
                if key in item
            }
        )
    return compact


def observed_gate_transitions(
    sequence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    transitions = []
    previous: Optional[dict[str, Any]] = None
    for state in sequence:
        if previous is None:
            previous = state
            continue
        for gate_key in ("upper_gate_state", "lower_gate_state"):
            before = previous.get(gate_key)
            after = state.get(gate_key)
            if before == after:
                continue
            transitions.append(
                {
                    "gate": gate_key,
                    "from": before,
                    "to": after,
                    "from_frame_index": previous.get("frame_index"),
                    "to_frame_index": state.get("frame_index"),
                    "to_relative_time_sec": state.get("relative_time_sec"),
                }
            )
        previous = state
    return transitions


def opening_completed_hold_rules(
    current_state: dict[str, Any],
    observed_transitions: list[dict[str, Any]],
    ship_status: dict[str, Any],
) -> list[dict[str, Any]]:
    if ship_status.get("all_labeled_ship_instances_berthed", False):
        return []
    rules = []
    for gate_key in ("upper_gate_state", "lower_gate_state"):
        if current_state.get(gate_key) != "open":
            continue
        gate_transitions = [
            transition
            for transition in observed_transitions
            if transition.get("gate") == gate_key
        ]
        if not gate_transitions:
            continue
        last_transition = gate_transitions[-1]
        if (
            last_transition.get("from") != "opening"
            or last_transition.get("to") != "open"
        ):
            continue
        rules.append(
            {
                "gate": gate_key,
                "forced_future_label": "open",
                "condition": (
                    "input shows opening_to_open completed; short horizon remains "
                    "open unless all labeled ships are ship_berthed"
                ),
                "exception": (
                    "If all labeled ships are ship_berthed, the open gate may start "
                    "closing."
                ),
            }
        )
    return rules


def candidate_future_gate_checks(current_state: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for gate_key, camera in (
        ("upper_gate_state", "CAM_3"),
        ("lower_gate_state", "CAM_8"),
    ):
        current_label = current_state.get(gate_key)
        competing_label = {
            "open": "closing",
            "closed": "opening",
            "opening": "open",
            "closing": "closed",
        }.get(current_label)
        if competing_label is None:
            continue
        checks.append(
            {
                "gate": gate_key,
                "state_camera": camera,
                "current_label": current_label,
                "competing_future_label": competing_label,
                "confusing_pair": [current_label, competing_label],
                "open_to_closing_requires_all_ships_berthed": bool(
                    current_label == "open" and competing_label == "closing"
                ),
                "all_labeled_ship_instances_berthed": False,
            }
        )
    return checks


def build_fusion_reasoning_context(item: dict[str, Any]) -> dict[str, Any]:
    camera_layout = item.get("input", {}).get("camera_layout")
    if not isinstance(camera_layout, dict):
        camera_layout = {}
    calibrated_cameras = camera_layout.get("calibrated_geometry_cameras")
    state_cameras = camera_layout.get("uncalibrated_state_cameras")
    if not isinstance(calibrated_cameras, list):
        calibrated_cameras = list(CALIBRATED_CAMERAS)
    if not isinstance(state_cameras, list):
        state_cameras = list(STATE_CAMERAS)
    return {
        "use_calibrated_2d_3d_fusion": bool(calibrated_cameras),
        "calibrated_cameras": [
            str(camera) for camera in calibrated_cameras if camera is not None
        ],
        "state_cameras_without_geometry": [
            str(camera) for camera in state_cameras if camera is not None
        ],
    }


def build_ship_behavior_context(frames: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "latest_ship_instances": latest_ship_instances(frames),
        "input_ship_intention_observation_counts": dict(
            sorted(ship_intention_observation_counts(frames).items())
        ),
        "input_mooring_evidence_counts": input_mooring_evidence_counts(frames),
    }


def latest_ship_instances(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for frame in reversed(frames):
        instances = [
            compact_ship_instance(instance)
            for instance in frame.get("ship_instances", [])
            if is_ship_instance(instance)
        ]
        if instances:
            return instances[:8]
    return []


def compact_ship_instance(instance: dict[str, Any]) -> dict[str, Any]:
    item = {
        "instance_token": instance.get("instance_token"),
        "category": instance.get("category"),
        "ship_intentions": [
            str(label)
            for label in (instance.get("ship_intentions") or [])
            if label is not None
        ],
    }
    for key in ("translation_xy", "velocity_xy", "num_lidar_points"):
        if key in instance:
            item[key] = instance[key]
    return {key: value for key, value in item.items() if value is not None}


def ship_intention_observation_counts(frames: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for frame in frames:
        for instance in frame.get("ship_instances", []):
            if not is_ship_instance(instance):
                continue
            for label in instance.get("ship_intentions") or []:
                counts[str(label)] += 1
    return counts


def input_mooring_evidence_counts(frames: list[dict[str, Any]]) -> dict[str, Any]:
    crew_count = 0
    mooring_line_count = 0
    ship_2d_count = 0
    ship_3d_count = 0
    saw_counts = False
    for frame in frames:
        for image in frame.get("images", {}).values():
            counts = (
                (image.get("perception_2d_summary") or {}).get("counts_by_class")
                or {}
            )
            if counts:
                saw_counts = True
            crew_count += int(counts.get("Crew_member", 0))
            mooring_line_count += int(counts.get("Mooring_line", 0))
            ship_2d_count += sum(int(counts.get(name, 0)) for name in SHIP_2D_CLASSES)

        counts_3d = (
            (frame.get("lidar", {}).get("perception_3d_summary") or {}).get(
                "counts_by_class"
            )
            or {}
        )
        if counts_3d:
            saw_counts = True
        ship_3d_count += sum(int(counts_3d.get(name, 0)) for name in SHIP_3D_CLASSES)

    if not saw_counts:
        return {}
    return {
        "crew_count_2d": crew_count,
        "mooring_line_count_2d": mooring_line_count,
        "ship_count_2d": ship_2d_count,
        "ship_count_3d": ship_3d_count,
        "mooring_confidence_boost_present": bool(
            crew_count > 0
            and mooring_line_count > 0
            and (ship_2d_count > 0 or ship_3d_count > 0)
        ),
        "weak_rule": MOORING_CONFIDENCE_WEAK_RULE,
    }


def is_ship_instance(instance: dict[str, Any]) -> bool:
    category = instance.get("category")
    return (
        bool(instance.get("ship_intentions"))
        or category in SHIP_2D_CLASSES
        or category in SHIP_3D_CLASSES
    )


def _compact_detection_counts(flat_features: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in flat_features.items()
        if key.endswith("_num_detections") or key.endswith("_num_ship_detections")
    }


def build_schema_outline(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: build_schema_outline(child) for key, child in value.items()}
    if isinstance(value, list):
        if value:
            return [build_schema_outline(value[0])]
        return []
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if value is None:
        return "null"
    return "string"


def build_schema_critical_templates(answer: dict[str, Any]) -> dict[str, Any]:
    templates: dict[str, Any] = {}
    evidence = (
        answer.get("ship_behavior", {})
        .get("mooring_or_berthing_confidence_evidence")
    )
    if isinstance(evidence, dict):
        templates[
            "ship_behavior.mooring_or_berthing_confidence_evidence"
        ] = {
            key: build_schema_outline(evidence.get(key))
            for key in (
                "crew_count_2d",
                "mooring_line_count_2d",
                "ship_count_2d",
                "ship_count_3d",
                "mooring_confidence_boost_present",
                "weak_rule",
            )
            if key in evidence
        }
    return templates


def build_required_json_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            paths.extend(build_required_json_paths(child, child_prefix))
        return paths
    if isinstance(value, list):
        return [prefix]
    return [prefix]


def to_file_uri(path: str) -> str:
    raw_path = Path(path)
    if raw_path.is_absolute():
        resolved = raw_path
    else:
        resolved = (Path.cwd() / raw_path).resolve()
    return resolved.as_uri()


if __name__ == "__main__":
    main()
