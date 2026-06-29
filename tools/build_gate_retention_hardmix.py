#!/usr/bin/env python3
"""Build VLM semantic hardmix data for future gate-state transitions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="VLM semantic prediction JSONL.")
    parser.add_argument("--output", required=True, help="Output hardmix JSONL.")
    parser.add_argument(
        "--stable-repeat",
        type=int,
        default=2,
        help="Repeat stable gate-retention samples this many times.",
    )
    parser.add_argument(
        "--transition-repeat",
        type=int,
        default=4,
        help="Repeat ordinary transition samples this many times.",
    )
    parser.add_argument(
        "--focus-transition-repeat",
        type=int,
        default=8,
        help=(
            "Repeat confusing open/closing and closed/opening transition samples "
            "this many times."
        ),
    )
    parser.add_argument(
        "--active-repeat",
        type=int,
        default=3,
        help="Repeat stable opening/closing motion-label samples this many times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    write_counts: Counter[str] = Counter()
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        answer = item.get("answer", {})
        current = answer.get("current_state", {})
        future = answer.get("future_state_10s", {})
        if not isinstance(current, dict) or not isinstance(future, dict):
            continue
        transition_labels = gate_transition_labels(current, future)
        bucket = gate_bucket(current, future, transition_labels)
        source_counts[bucket] += 1
        repeat = repeat_for_bucket(
            bucket=bucket,
            stable_repeat=args.stable_repeat,
            transition_repeat=args.transition_repeat,
            focus_transition_repeat=args.focus_transition_repeat,
            active_repeat=args.active_repeat,
        )
        for copy_index in range(repeat):
            clone = dict(item)
            clone["id"] = f"{item.get('id')}:gatehard{copy_index}"
            metadata = dict(clone.get("metadata", {}))
            metadata["gate_hardmix_source_id"] = item.get("id")
            metadata["gate_hardmix_bucket"] = bucket
            metadata["gate_hardmix_transition_labels"] = transition_labels
            clone["metadata"] = metadata
            selected.append(clone)
            write_counts[bucket] += 1

    with output_path.open("w", encoding="utf-8") as dst:
        for item in selected:
            dst.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"wrote={output_path}")
    print(f"num_items={len(selected)}")
    print("source_counts=" + json.dumps(source_counts, sort_keys=True))
    print("write_counts=" + json.dumps(write_counts, sort_keys=True))


def gate_transition_labels(
    current: dict[str, Any],
    future: dict[str, Any],
) -> list[str]:
    labels = []
    for gate_key, prefix in (
        ("upper_gate_state", "upper"),
        ("lower_gate_state", "lower"),
    ):
        current_label = current.get(gate_key)
        future_label = future.get(gate_key)
        if current_label == future_label:
            continue
        labels.append(
            f"{prefix}_{normalize_label(current_label)}_to_{normalize_label(future_label)}"
        )
    return labels


def gate_bucket(
    current: dict[str, Any],
    future: dict[str, Any],
    transition_labels: list[str],
) -> str:
    if any(is_focus_transition(label) for label in transition_labels):
        return "focus_transition_" + "+".join(sorted(transition_labels))
    if transition_labels:
        return "transition_" + "+".join(sorted(transition_labels))

    current_tuple = gate_tuple(current)
    future_tuple = gate_tuple(future)
    upper, lower, water = current_tuple
    if upper in {"opening", "closing"} or lower in {"opening", "closing"}:
        return "stable_motion_label"
    if water == "idle" and upper == "closed" and lower == "open":
        return "stable_lower_open_idle"
    if water == "idle" and upper == "open" and lower == "closed":
        return "stable_upper_open_idle"
    if water == "idle":
        return "stable_other_idle"
    return "stable_non_idle"


def gate_tuple(state: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        state.get("upper_gate_state"),
        state.get("lower_gate_state"),
        state.get("water_state"),
    )


def normalize_label(value: Any) -> str:
    if value is None:
        return "none"
    return str(value).replace(" ", "_")


def is_focus_transition(label: str) -> bool:
    return label in {
        "upper_open_to_closing",
        "upper_closed_to_opening",
        "lower_open_to_closing",
        "lower_closed_to_opening",
    }


def repeat_for_bucket(
    bucket: str,
    stable_repeat: int,
    transition_repeat: int,
    focus_transition_repeat: int,
    active_repeat: int,
) -> int:
    if bucket.startswith("focus_transition_"):
        return max(1, focus_transition_repeat)
    if bucket.startswith("transition_"):
        return max(1, transition_repeat)
    if bucket == "stable_motion_label":
        return max(1, active_repeat)
    if bucket in {"stable_lower_open_idle", "stable_upper_open_idle"}:
        return max(1, stable_repeat)
    return 1


if __name__ == "__main__":
    main()
