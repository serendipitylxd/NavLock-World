#!/usr/bin/env python3
"""Rebuild VLM semantic Qwen input with baseline-derived ship-intention context.

The VLM semantic branch was trained with ship-intention context as input and
``ship_behavior.ship_intentions`` as output. This tool keeps that input/output
contract, but replaces annotation-backed ship-intention context in the prompt
with ship intentions produced by the current perception/geometric baseline.
The assistant reference is left unchanged so evaluation still scores against the
original target.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any, Iterable


SHIP_INTENTION_LABELS = {
    "ship_berthed",
    "ship_entering_lock",
    "ship_leaving_lock",
    "ship_static",
    "ship_moving",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Existing Qwen VLM semantic JSONL input.",
    )
    parser.add_argument(
        "--ship-source",
        type=Path,
        required=True,
        help=(
            "JSONL containing baseline-derived ship_behavior.ship_intentions, "
            "for example the current fused baseline prediction JSONL."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Qwen VLM semantic JSONL with rebuilt ship-intention context.",
    )
    parser.add_argument(
        "--source-name",
        default="hydro3dnet_rtmdet_geometry",
        help="Metadata label for the replacement source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_index = load_ship_intention_index(args.ship_source)
    rows = load_jsonl(args.input)
    rebuilt = []
    missing_source = 0
    for row in rows:
        source_items, source_found = source_items_for_row(row, source_index)
        if not source_found:
            missing_source += 1
            source_items = []
        rebuilt.append(
            rebuild_item_ship_context(
                row,
                source_items,
                source_name=args.source_name,
                source_found=source_found,
            )
        )
    write_jsonl(args.output, rebuilt)
    report = {
        "input": str(args.input),
        "ship_source": str(args.ship_source),
        "output": str(args.output),
        "rows": len(rebuilt),
        "source_scenes": len(source_index["by_scene"]),
        "source_frames": len(source_index["by_scene_sample"]),
        "missing_source_rows": missing_source,
        "source_name": args.source_name,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"expected JSON object row in {path}")
            rows.append(parsed)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_ship_intention_index(path: Path) -> dict[str, dict[Any, list[dict[str, Any]]]]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    by_scene_sample: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        scene_token = scene_token_from_row(row)
        if not scene_token:
            continue
        items = extract_ship_intentions(row)
        normalized = normalize_ship_intention_items(items)
        sample_token = sample_token_from_row(row)
        if sample_token:
            by_scene_sample[(scene_token, sample_token)] = normalized
        else:
            by_scene[scene_token] = normalized
    return {"by_scene": by_scene, "by_scene_sample": by_scene_sample}


def load_ship_intentions_by_scene(path: Path) -> dict[str, list[dict[str, Any]]]:
    return load_ship_intention_index(path)["by_scene"]


def source_items_for_row(
    row: dict[str, Any],
    source_index: dict[str, dict[Any, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], bool]:
    scene_token = scene_token_from_row(row)
    sample_token = sample_token_from_row(row)
    by_scene_sample = source_index.get("by_scene_sample", {})
    if scene_token and sample_token:
        key = (scene_token, sample_token)
        if key in by_scene_sample:
            return by_scene_sample[key], True
    by_scene = source_index.get("by_scene", {})
    if scene_token in by_scene:
        return by_scene[scene_token], True
    return [], False


def scene_token_from_row(row: dict[str, Any]) -> str:
    scene_token = row.get("scene_token")
    if isinstance(scene_token, str) and scene_token:
        return scene_token
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        scene_token = metadata.get("scene_token")
        if isinstance(scene_token, str) and scene_token:
            return scene_token
    row_id = row.get("id")
    if isinstance(row_id, str) and ":prediction:" in row_id:
        return row_id.split(":prediction:", 1)[1]
    if isinstance(row_id, str) and ":recognition_frame:" in row_id:
        remainder = row_id.split(":recognition_frame:", 1)[1]
        return remainder.rsplit(":", 1)[0]
    if isinstance(row_id, str) and ":recognition:" in row_id:
        return row_id.split(":recognition:", 1)[1]
    return ""


def sample_token_from_row(row: dict[str, Any]) -> str:
    sample_token = row.get("sample_token")
    if isinstance(sample_token, str) and sample_token:
        return sample_token
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        sample_token = metadata.get("sample_token")
        if isinstance(sample_token, str) and sample_token:
            return sample_token
    row_id = row.get("id")
    if isinstance(row_id, str) and ":recognition_frame:" in row_id:
        return row_id.rsplit(":", 1)[-1]
    return ""


def extract_ship_intentions(row: dict[str, Any]) -> Any:
    for candidate in (
        row.get("prediction_json"),
        row.get("prediction"),
        row,
    ):
        if not isinstance(candidate, dict):
            continue
        behavior = candidate.get("ship_behavior")
        if isinstance(behavior, dict):
            return behavior.get("ship_intentions")
    return []


def normalize_ship_intention_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        token = item.get("instance_token")
        if token is None:
            continue
        token = str(token)
        if not token or token in seen:
            continue
        labels = [
            str(label)
            for label in (item.get("ship_intentions") or [])
            if label is not None
        ]
        labels = [label for label in labels if label in SHIP_INTENTION_LABELS]
        if not labels:
            continue
        normalized = {
            "instance_token": token,
            "category": str(item.get("category") or "Unknown_vessel"),
            "ship_intentions": labels,
        }
        out.append(normalized)
        seen.add(token)
    return out


def rebuild_item_ship_context(
    item: dict[str, Any],
    source_items: list[dict[str, Any]],
    source_name: str = "hydro3dnet_rtmdet_geometry",
    source_found: bool = True,
) -> dict[str, Any]:
    rebuilt = copy.deepcopy(item)
    source_items = normalize_ship_intention_items(source_items)
    payload = prompt_payload(rebuilt)
    replace_payload_ship_context(payload, source_items)
    replace_prompt_payload(rebuilt, payload)
    metadata = rebuilt.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["ship_intention_context_source"] = source_name
        metadata["ship_intention_context_source_found"] = bool(source_found)
        metadata["ship_intention_context_count"] = len(source_items)
    return rebuilt


def prompt_payload(item: dict[str, Any]) -> dict[str, Any]:
    user_message = first_user_message(item)
    content = user_message.get("content", [])
    if isinstance(content, str):
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("prompt text must be a JSON object")
        return payload
    if not isinstance(content, list):
        raise ValueError("user content must be a list or string")
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            payload = json.loads(str(part.get("text", "{}")))
            if not isinstance(payload, dict):
                raise ValueError("prompt text must be a JSON object")
            return payload
    raise ValueError("item has no user text payload")


def replace_prompt_payload(item: dict[str, Any], payload: dict[str, Any]) -> None:
    user_message = first_user_message(item)
    content = user_message.get("content", [])
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if isinstance(content, str):
        user_message["content"] = text
        return
    if not isinstance(content, list):
        raise ValueError("user content must be a list or string")
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            part["text"] = text
            return
    raise ValueError("item has no user text payload")


def first_user_message(item: dict[str, Any]) -> dict[str, Any]:
    for message in item.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    raise ValueError("item has no user message")


def replace_payload_ship_context(
    payload: dict[str, Any],
    source_items: list[dict[str, Any]],
) -> None:
    source_items = normalize_ship_intention_items(source_items)
    replace_ship_behavior_context(payload, source_items)
    replace_frame_ship_instances(payload, source_items)
    frame_count = input_frame_count(payload)
    update_all_gate_transition_contexts(payload, source_items, frame_count)


def replace_ship_behavior_context(
    payload: dict[str, Any],
    source_items: list[dict[str, Any]],
) -> None:
    context = payload.get("ship_behavior_context")
    if not isinstance(context, dict):
        context = {}
        payload["ship_behavior_context"] = context
    context["latest_ship_instances"] = copy.deepcopy(source_items)
    context["input_ship_intention_observation_counts"] = dict(
        ship_intention_counts(source_items, max(1, input_frame_count(payload)))
    )


def replace_frame_ship_instances(
    payload: dict[str, Any],
    source_items: list[dict[str, Any]],
) -> None:
    input_payload = payload.get("input")
    if not isinstance(input_payload, dict):
        return
    frames = input_payload.get("frames")
    if not isinstance(frames, list):
        return
    for frame in frames:
        if isinstance(frame, dict):
            frame["ship_instances"] = copy.deepcopy(source_items)


def update_all_gate_transition_contexts(
    payload: dict[str, Any],
    source_items: list[dict[str, Any]],
    frame_count: int,
) -> None:
    contexts: list[dict[str, Any]] = []
    for context in (
        payload.get("gate_transition_context"),
        nested_get(payload, ("compact_input_summary", "gate_transition_context")),
        nested_get(payload, ("input", "gate_transition_context")),
    ):
        if isinstance(context, dict):
            contexts.append(context)
    for context in contexts:
        update_gate_transition_context(context, source_items, frame_count)


def update_gate_transition_context(
    context: dict[str, Any],
    source_items: list[dict[str, Any]],
    frame_count: int,
) -> None:
    status = context.get("ship_berthing_status")
    if not isinstance(status, dict):
        status = {}
        context["ship_berthing_status"] = status
    num_instances = len(source_items)
    num_berthed = sum(
        1 for item in source_items if "ship_berthed" in item.get("ship_intentions", [])
    )
    all_berthed = bool(num_instances and num_berthed == num_instances)
    status.update(
        {
            "berth_label": status.get("berth_label", "ship_berthed"),
            "num_labeled_ship_instances": num_instances,
            "num_labeled_berthed_ship_instances": num_berthed,
            "ship_berthing_labels_available": bool(num_instances),
            "all_labeled_ship_instances_berthed": all_berthed,
            "gate_closing_precondition": status.get(
                "gate_closing_precondition",
                "An open gate may transition to closing only when all ships are berthed.",
            ),
        }
    )
    for check in context.get("candidate_future_gate_checks", []):
        if isinstance(check, dict):
            check["all_labeled_ship_instances_berthed"] = all_berthed
    current_state = current_gate_state_from_context(context)
    observed = context.get("observed_input_gate_transitions")
    if not isinstance(observed, list):
        observed = []
    opening_hold = opening_completed_hold_rules(current_state, observed, all_berthed)
    domain_rules = future_gate_domain_rules(current_state, observed, all_berthed)
    domain_rules.extend(opening_hold)
    context["future_gate_domain_rules"] = domain_rules
    context["opening_completed_hold_rules"] = opening_hold
    context["ship_intention_observation_counts"] = dict(
        ship_intention_counts(source_items, max(1, frame_count))
    )


def current_gate_state_from_context(context: dict[str, Any]) -> dict[str, Any]:
    current: dict[str, Any] = {}
    for check in context.get("candidate_future_gate_checks", []):
        if isinstance(check, dict) and check.get("gate"):
            current[str(check["gate"])] = check.get("current_label")
    return current


def future_gate_domain_rules(
    current_state: dict[str, Any],
    observed_transitions: list[Any],
    all_berthed: bool,
) -> list[dict[str, Any]]:
    if not all_berthed:
        return []
    rules: list[dict[str, Any]] = []
    for gate_key in ("upper_gate_state", "lower_gate_state"):
        if current_state.get(gate_key) == "closing" or observed_open_to_closing(
            observed_transitions, gate_key
        ):
            rules.append(
                {
                    "gate": gate_key,
                    "forced_future_label": "closing",
                    "condition": (
                        "input already shows open_to_closing and all labeled ships "
                        "are ship_berthed"
                    ),
                }
            )
    return rules


def opening_completed_hold_rules(
    current_state: dict[str, Any],
    observed_transitions: list[Any],
    all_berthed: bool,
) -> list[dict[str, Any]]:
    if all_berthed:
        return []
    rules: list[dict[str, Any]] = []
    for gate_key in ("upper_gate_state", "lower_gate_state"):
        if current_state.get(gate_key) != "open":
            continue
        if not observed_opening_to_open(observed_transitions, gate_key):
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


def observed_open_to_closing(transitions: list[Any], gate_key: str) -> bool:
    return any(
        isinstance(item, dict)
        and item.get("gate") == gate_key
        and item.get("from") == "open"
        and item.get("to") == "closing"
        for item in transitions
    )


def observed_opening_to_open(transitions: list[Any], gate_key: str) -> bool:
    gate_transitions = [
        item
        for item in transitions
        if isinstance(item, dict) and item.get("gate") == gate_key
    ]
    if not gate_transitions:
        return False
    last = gate_transitions[-1]
    return last.get("from") == "opening" and last.get("to") == "open"


def ship_intention_counts(items: list[dict[str, Any]], multiplier: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in items:
        for label in item.get("ship_intentions", []):
            counts[str(label)] += multiplier
    return counts


def input_frame_count(payload: dict[str, Any]) -> int:
    frames = nested_get(payload, ("input", "frames"))
    return len(frames) if isinstance(frames, list) else 0


def nested_get(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


if __name__ == "__main__":
    main()
