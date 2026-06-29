#!/usr/bin/env python3
"""Create controlled Qwen3-VL VLM semantic input ablations without retraining."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable, Iterable


ABLATION_CHOICES = (
    "full",
    "text_only",
    "state_cameras_only",
    "lidar_views_only",
    "no_lidar_images",
    "no_state_camera_images",
    "no_lock_operation_transition_prior",
    "no_operational_telemetry",
    "no_ship_behavior_context",
    "no_mooring_evidence",
    "no_wave_evidence",
    "no_perception_summaries",
    "deployable_perception_only",
)

STATE_CAMERA_CHANNELS = {"CAM_3", "CAM_8"}
MOORING_CLASSES = {"Crew_member", "Mooring_line"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Qwen3-VL JSONL input.")
    parser.add_argument("--output", type=Path, required=True, help="Ablated JSONL output.")
    parser.add_argument(
        "--ablation",
        action="append",
        choices=ABLATION_CHOICES,
        required=True,
        help="Ablation to apply. Repeat to compose independent removals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    ablations = normalize_ablations(args.ablation)
    out = [ablate_item(row, ablations) for row in rows]
    write_jsonl(args.output, out)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "num_rows": len(out),
        "ablations": ablations,
        "selected_num_images": sorted(
            {image_count_from_item(row) for row in out}
        ),
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_ablations(raw: list[str]) -> list[str]:
    out: list[str] = []
    for item in raw:
        if item == "full":
            continue
        if item == "deployable_perception_only":
            for expanded in (
                "deployable_perception_only",
                "no_ship_behavior_context",
                "no_lock_operation_transition_prior",
            ):
                if expanded not in out:
                    out.append(expanded)
            continue
        if item not in out:
            out.append(item)
    return out


def ablate_item(item: dict[str, Any], ablations: list[str]) -> dict[str, Any]:
    ablated = copy.deepcopy(item)
    user_message = first_user_message(ablated)
    content = user_message.get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
        user_message["content"] = content
    if not isinstance(content, list):
        return ablated

    text_index = first_text_part_index(content)
    if text_index is None:
        return ablated

    payload = json.loads(str(content[text_index].get("text", "{}")))
    image_parts = [part for part in content if isinstance(part, dict) and part.get("type") == "image"]
    visual_inputs = payload.get("selected_visual_inputs")
    if not isinstance(visual_inputs, list):
        visual_inputs = [{} for _ in image_parts]

    kept_indices = list(range(len(image_parts)))
    kept_indices = apply_image_ablations(kept_indices, visual_inputs, image_parts, ablations)
    payload = apply_text_ablations(payload, ablations)
    payload["selected_visual_inputs"] = reindex_visual_inputs(visual_inputs, kept_indices)
    payload["image_usage_note"] = image_usage_note(payload["selected_visual_inputs"])
    visible_ablations = public_ablations(ablations)
    if ablations:
        payload["input_ablation"] = visible_ablations

    new_content: list[dict[str, Any]] = []
    for index, part in enumerate(image_parts):
        if index in kept_indices:
            new_content.append(part)
    new_content.append(
        {
            "type": "text",
            "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        }
    )
    user_message["content"] = new_content

    metadata = ablated.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["input_ablation"] = visible_ablations or ["full"]
        metadata["selected_num_images"] = len(kept_indices)
        metadata["selected_num_lidar_images"] = sum(
            1 for item in payload["selected_visual_inputs"] if is_lidar_visual(item)
        )
    return ablated


def public_ablations(ablations: list[str]) -> list[str]:
    if "deployable_perception_only" not in ablations:
        return list(ablations)
    internal = {"no_ship_behavior_context", "no_lock_operation_transition_prior"}
    return [item for item in ablations if item not in internal]


def first_user_message(item: dict[str, Any]) -> dict[str, Any]:
    for message in item.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    raise ValueError("item has no user message")


def first_text_part_index(content: list[Any]) -> int | None:
    for index, part in enumerate(content):
        if isinstance(part, dict) and part.get("type") == "text":
            return index
    return None


def apply_image_ablations(
    kept_indices: list[int],
    visual_inputs: list[Any],
    image_parts: list[dict[str, Any]],
    ablations: list[str],
) -> list[int]:
    predicates: list[Callable[[int], bool]] = []
    if "text_only" in ablations:
        return []
    if "state_cameras_only" in ablations:
        predicates.append(lambda idx: is_state_camera_visual(visual_at(visual_inputs, idx), image_parts[idx]))
    if "lidar_views_only" in ablations:
        predicates.append(lambda idx: is_lidar_visual(visual_at(visual_inputs, idx), image_parts[idx]))
    if "no_lidar_images" in ablations:
        predicates.append(lambda idx: not is_lidar_visual(visual_at(visual_inputs, idx), image_parts[idx]))
    if "no_state_camera_images" in ablations:
        predicates.append(
            lambda idx: not is_state_camera_visual(visual_at(visual_inputs, idx), image_parts[idx])
        )
    for predicate in predicates:
        kept_indices = [idx for idx in kept_indices if predicate(idx)]
    return kept_indices


def apply_text_ablations(payload: dict[str, Any], ablations: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    if "no_lock_operation_transition_prior" in ablations:
        remove_keys_recursive(out, {"gate_transition_context"})
        filter_schema_rules(
            out,
            (
                "gate_transition_context",
                "forced_future_label",
                "opening_to_open",
                "open gate may transition",
                "candidate_future_gate_checks",
            ),
        )
        append_instruction_note(out, "Do not use the lock-operation transition prior.")

    if "no_operational_telemetry" in ablations:
        remove_operational_telemetry(out)
        filter_schema_rules(out, ("water_level_context", "gate_state_context", "copy current_state"))
        append_instruction_note(out, "Operational lock telemetry is hidden for this ablation.")

    if "no_ship_behavior_context" in ablations:
        remove_keys_recursive(out, {"ship_behavior_context", "ship_instances"})
        filter_schema_rules(out, ("ship_behavior_context",))
        append_instruction_note(out, "Do not use the ship-behavior copy prior.")

    if "no_mooring_evidence" in ablations:
        remove_key_recursive(out, "input_mooring_evidence_counts")
        scrub_mooring_detection_counts(out)
        filter_schema_rules(out, ("mooring", "Crew_member", "Mooring_line"))
        append_instruction_note(out, "Mooring and berthing-confidence evidence is hidden.")

    if "no_wave_evidence" in ablations:
        filter_schema_rules(out, ("Weak wave rule",))
        strip_instruction_phrases(
            out,
            (
                "whether filling or emptying may cause waves or surface disturbance in the target water-surface region; ",
                "water-surface ",
            ),
        )
        append_instruction_note(out, "Water-surface wave evidence rules are hidden.")

    if "no_perception_summaries" in ablations:
        remove_keys_recursive(
            out,
            {
                "perception_2d_summary",
                "perception_3d_summary",
                "flat_perception_features",
                "detector_sources",
                "last_frame_detection_counts",
                "lidar_visualization",
            },
        )
        append_instruction_note(out, "Structured perception summaries are hidden.")

    if "deployable_perception_only" in ablations:
        append_instruction_note(
            out,
            (
                "Deployable perception-only input: use operational telemetry, "
                "camera/LiDAR images, and RTMDet/Hydro3DNet perception summaries; "
                "do not use annotation-backed ship instances, previous "
                "ship-intention labels, or label-derived ship berthing priors."
            ),
        )
    return out


def visual_at(visual_inputs: list[Any], index: int) -> dict[str, Any]:
    if index < len(visual_inputs) and isinstance(visual_inputs[index], dict):
        return visual_inputs[index]
    return {}


def is_state_camera_visual(visual: dict[str, Any], image_part: dict[str, Any] | None = None) -> bool:
    channel = str(visual.get("channel") or "")
    if channel in STATE_CAMERA_CHANNELS:
        return True
    image = str((image_part or {}).get("image") or "")
    return any(f"/{channel}/" in image for channel in STATE_CAMERA_CHANNELS)


def is_lidar_visual(visual: dict[str, Any], image_part: dict[str, Any] | None = None) -> bool:
    kind = str(visual.get("kind") or "")
    view_type = str(visual.get("view_type") or "")
    image = str((image_part or {}).get("image") or "")
    return (
        kind.startswith("lidar")
        or view_type in {"bev", "range_view"}
        or "lidar_views" in image
        or "_bev" in image
        or "_range" in image
    )


def reindex_visual_inputs(visual_inputs: list[Any], kept_indices: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for new_index, old_index in enumerate(kept_indices):
        visual = copy.deepcopy(visual_at(visual_inputs, old_index))
        visual["image_index"] = new_index
        out.append(visual)
    return out


def image_usage_note(visual_inputs: list[dict[str, Any]]) -> str:
    if not visual_inputs:
        return "No image parts are included for this input ablation; use only the remaining structured text."
    kinds = ", ".join(str(item.get("kind", "image")) for item in visual_inputs)
    return f"The preceding {len(visual_inputs)} images follow selected_visual_inputs order after ablation: {kinds}."


def image_count_from_item(item: dict[str, Any]) -> int:
    message = first_user_message(item)
    content = message.get("content", [])
    if not isinstance(content, list):
        return 0
    return sum(
        1 for part in content if isinstance(part, dict) and part.get("type") == "image"
    )


def remove_keys_recursive(value: Any, keys: set[str]) -> None:
    if isinstance(value, dict):
        for key in list(value):
            if key in keys:
                del value[key]
            else:
                remove_keys_recursive(value[key], keys)
    elif isinstance(value, list):
        for item in value:
            remove_keys_recursive(item, keys)


def remove_key_recursive(value: Any, key_to_remove: str) -> None:
    remove_keys_recursive(value, {key_to_remove})


def remove_operational_telemetry(payload: dict[str, Any]) -> None:
    for key in ("water_level_context", "gate_state_context"):
        payload.pop(key, None)
    compact = payload.get("compact_input_summary")
    if isinstance(compact, dict):
        for key in (
            "current_state_from_last_input_frame",
            "input_lock_state_sequence",
            "input_water_level_delta",
        ):
            compact.pop(key, None)
    input_payload = payload.get("input")
    if isinstance(input_payload, dict):
        input_payload.pop("current_state_from_last_input_frame", None)
        for frame in input_payload.get("frames") or []:
            if isinstance(frame, dict):
                frame.pop("lock_state", None)


def filter_schema_rules(payload: dict[str, Any], blocked_substrings: tuple[str, ...]) -> None:
    rules = (payload.get("response_contract") or {}).get("schema_critical_rules")
    if not isinstance(rules, list):
        return
    blocked_lower = tuple(item.lower() for item in blocked_substrings)
    payload["response_contract"]["schema_critical_rules"] = [
        rule
        for rule in rules
        if not any(fragment in str(rule).lower() for fragment in blocked_lower)
    ]


def append_instruction_note(payload: dict[str, Any], note: str) -> None:
    instruction = payload.get("instruction")
    if isinstance(instruction, str) and note not in instruction:
        payload["instruction"] = instruction.rstrip() + " " + note


def strip_instruction_phrases(payload: dict[str, Any], phrases: tuple[str, ...]) -> None:
    instruction = payload.get("instruction")
    if not isinstance(instruction, str):
        return
    for phrase in phrases:
        instruction = instruction.replace(phrase, "")
    payload["instruction"] = instruction


def scrub_mooring_detection_counts(value: Any) -> None:
    if isinstance(value, dict):
        if all(key in value for key in MOORING_CLASSES) and all(
            isinstance(value.get(key), (int, float)) for key in MOORING_CLASSES
        ):
            for key in MOORING_CLASSES:
                value[key] = 0
        for key in list(value):
            if key in MOORING_CLASSES:
                value[key] = 0
            else:
                scrub_mooring_detection_counts(value[key])
    elif isinstance(value, list):
        for item in value:
            scrub_mooring_detection_counts(item)


if __name__ == "__main__":
    main()
