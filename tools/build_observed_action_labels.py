#!/usr/bin/env python3
"""Derive weak observed-action labels from gate and water-state transitions."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Optional


DEFAULT_MAX_GAP_SEC = 120.0
ACTION_FIELDS = (
    "observed_action",
    "action_start_time",
    "action_end_time",
    "action_target",
    "action_source",
    "action_confidence",
)
GATE_WATER_ACTIONS = (
    "hold",
    "open_upper_gate",
    "close_upper_gate",
    "open_lower_gate",
    "close_lower_gate",
    "start_filling",
    "start_emptying",
    "stop_filling_emptying",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--max-gap-sec", type=float, default=DEFAULT_MAX_GAP_SEC)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/observed_action_labels/summary.json"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("outputs/observed_action_labels/observed_actions.jsonl"),
    )
    parser.add_argument(
        "--no-update-sample",
        action="store_true",
        help="Only write summary/jsonl, do not update sample.json.",
    )
    parser.add_argument(
        "--no-update-pkl",
        action="store_true",
        help="Do not synchronize huaiyin_infos_*.pkl files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_path = args.data_root / "v1.0-trainval" / "sample.json"
    rows = json.loads(sample_path.read_text(encoding="utf-8"))
    labels = build_action_labels(rows, max_gap_sec=args.max_gap_sec)

    if not args.no_update_sample:
        update_sample_rows(rows, labels)
        sample_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    pkl_report = []
    if not args.no_update_pkl:
        pkl_report = update_info_pkls(args.data_root, labels)

    write_action_jsonl(args.jsonl_output, rows, labels)
    summary = build_summary(
        rows,
        labels=labels,
        max_gap_sec=args.max_gap_sec,
        sample_path=sample_path,
        pkl_report=pkl_report,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote={args.summary_output}")
    print(f"wrote={args.jsonl_output}")
    print(f"num_samples={len(rows)}")
    print(f"action_counts={summary['action_counts']}")
    print(f"updated_sample={not args.no_update_sample}")
    print(f"updated_pkl={not args.no_update_pkl}")


def build_action_labels(
    rows: list[dict[str, Any]], *, max_gap_sec: float = DEFAULT_MAX_GAP_SEC
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for timeline in build_timelines(rows, max_gap_sec=max_gap_sec):
        raw_labels = [
            classify_gate_water_action(row, next_row(timeline, index))
            for index, row in enumerate(timeline)
        ]
        for start, end in action_episode_ranges(timeline, raw_labels):
            start_time = timeline[start]["timestamp"]
            end_time = timeline[end]["timestamp"]
            for index in range(start, end + 1):
                token = timeline[index]["token"]
                labels[token] = {
                    **raw_labels[index],
                    "action_start_time": start_time,
                    "action_end_time": end_time,
                }
    return labels


def build_timelines(
    rows: list[dict[str, Any]], *, max_gap_sec: float
) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: int(row["timestamp"]))
    timelines: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    max_gap_us = int(max_gap_sec * 1_000_000)
    for row in ordered:
        if current:
            prev = current[-1]
            must_split = (
                date_of(prev) != date_of(row)
                or int(row["timestamp"]) - int(prev["timestamp"]) > max_gap_us
            )
            if must_split:
                timelines.append(current)
                current = []
        current.append(row)
    if current:
        timelines.append(current)
    return timelines


def classify_gate_water_action(
    row: dict[str, Any], future: Optional[dict[str, Any]]
) -> dict[str, Any]:
    stop_water_action = classify_water_stop_action(
        row.get("lock_water_state"),
        future.get("lock_water_state") if future else None,
    )
    if stop_water_action:
        return stop_water_action

    upper_action = classify_gate_action(
        row.get("upper_gate_state"),
        future.get("upper_gate_state") if future else None,
        side="upper",
    )
    if upper_action:
        return upper_action

    lower_action = classify_gate_action(
        row.get("lower_gate_state"),
        future.get("lower_gate_state") if future else None,
        side="lower",
    )
    if lower_action:
        return lower_action

    water_action = classify_water_action(
        row.get("lock_water_state"),
        future.get("lock_water_state") if future else None,
    )
    if water_action:
        return water_action

    return {
        "observed_action": "hold",
        "action_target": "none",
        "action_source": "no_gate_water_transition",
        "action_confidence": 0.90,
    }


def classify_gate_action(
    current: Any, future: Any, *, side: str
) -> Optional[dict[str, Any]]:
    if current == "opening" or (
        current == "closed" and future in {"opening", "open"}
    ):
        return gate_action(f"open_{side}_gate", f"{side}_gate", current, future)
    if current == "closing" or (
        current == "open" and future in {"closing", "closed"}
    ):
        return gate_action(f"close_{side}_gate", f"{side}_gate", current, future)
    return None


def gate_action(
    action: str, target: str, current: Any, future: Any
) -> dict[str, Any]:
    return {
        "observed_action": action,
        "action_target": target,
        "action_source": "gate_state_transition_weak",
        "action_confidence": 1.00 if current in {"opening", "closing"} else 0.90,
    }


def classify_water_action(current: Any, future: Any) -> Optional[dict[str, Any]]:
    stop_action = classify_water_stop_action(current, future)
    if stop_action:
        return stop_action
    if current == "idle" and future == "filling":
        return water_action("start_filling", transition=True)
    if current == "idle" and future == "emptying":
        return water_action("start_emptying", transition=True)
    if current == "filling":
        return water_action("start_filling", transition=False)
    if current == "emptying":
        return water_action("start_emptying", transition=False)
    return None


def classify_water_stop_action(current: Any, future: Any) -> Optional[dict[str, Any]]:
    if current in {"filling", "emptying"} and future == "idle":
        return water_action("stop_filling_emptying", transition=True)
    return None


def water_action(action: str, *, transition: bool) -> dict[str, Any]:
    return {
        "observed_action": action,
        "action_target": "water_system",
        "action_source": "water_state_transition_weak",
        "action_confidence": 1.00 if transition else 0.95,
    }


def action_episode_ranges(
    timeline: list[dict[str, Any]], labels: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    if not timeline:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(timeline)):
        if action_key(labels[index]) != action_key(labels[index - 1]):
            ranges.append((start, index - 1))
            start = index
    ranges.append((start, len(timeline) - 1))
    return ranges


def action_key(label: dict[str, Any]) -> tuple[Any, Any]:
    return label.get("observed_action"), label.get("action_target")


def next_row(timeline: list[dict[str, Any]], index: int) -> Optional[dict[str, Any]]:
    if index + 1 >= len(timeline):
        return None
    return timeline[index + 1]


def update_sample_rows(
    rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> None:
    for row in rows:
        clear_action_fields(row)
        label = labels.get(row.get("token"))
        if label:
            row.update(label)


def update_info_pkls(
    data_root: Path, labels: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    report = []
    seen: set[Path] = set()
    for pattern_root in (data_root, data_root / "infos"):
        for path in sorted(pattern_root.glob("huaiyin_infos_*.pkl")):
            if path in seen:
                continue
            seen.add(path)
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            data_list = payload.get("data_list") if isinstance(payload, dict) else payload
            if not isinstance(data_list, list):
                continue
            changed = 0
            matched_rows = 0
            for item in data_list:
                if not isinstance(item, dict):
                    continue
                clear_action_fields(item)
                label = labels.get(item.get("sample_token"))
                if not label:
                    continue
                matched_rows += 1
                before = {key: item.get(key) for key in label}
                item.update(label)
                if any(before[key] != item.get(key) for key in label):
                    changed += 1
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            report.append(
                {
                    "path": str(path),
                    "matched_rows": matched_rows,
                    "changed_rows": changed,
                }
            )
    return report


def write_action_jsonl(
    path: Path, rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: int(item["timestamp"])):
            label = labels[row["token"]]
            out = {
                "sample_token": row["token"],
                "sample_idx": row.get("timestamp_str")
                or row.get("token", "").replace("sample_", ""),
                "timestamp": row.get("timestamp"),
                "scene_token": row.get("scene_token"),
                "upper_gate_state": row.get("upper_gate_state"),
                "lower_gate_state": row.get("lower_gate_state"),
                "lock_water_state": row.get("lock_water_state"),
                **label,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def build_summary(
    rows: list[dict[str, Any]],
    *,
    labels: dict[str, dict[str, Any]],
    max_gap_sec: float,
    sample_path: Path,
    pkl_report: list[dict[str, Any]],
) -> dict[str, Any]:
    timelines = build_timelines(rows, max_gap_sec=max_gap_sec)
    action_counts = Counter(label["observed_action"] for label in labels.values())
    target_counts = Counter(label["action_target"] for label in labels.values())
    return {
        "settings": {
            "sample_path": str(sample_path),
            "max_gap_sec": max_gap_sec,
            "source": "gate/water state transition weak labels",
        },
        "num_samples": len(rows),
        "num_timelines": len(timelines),
        "timeline_lengths": {
            "min": min((len(timeline) for timeline in timelines), default=0),
            "max": max((len(timeline) for timeline in timelines), default=0),
        },
        "action_counts": dict(sorted(action_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "episode_counts": episode_counts(labels),
        "pkl_update_report": pkl_report,
    }


def episode_counts(labels: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    seen: set[tuple[Any, Any, Any]] = set()
    for label in labels.values():
        key = (
            label.get("observed_action"),
            label.get("action_start_time"),
            label.get("action_end_time"),
        )
        if key in seen:
            continue
        seen.add(key)
        counts[str(label.get("observed_action"))] += 1
    return dict(sorted(counts.items()))


def clear_action_fields(row: dict[str, Any]) -> None:
    for key in ACTION_FIELDS:
        row.pop(key, None)


def date_of(row: dict[str, Any]) -> str:
    timestamp_str = str(row.get("timestamp_str") or row.get("sample_idx") or "")
    if len(timestamp_str) >= 10:
        return timestamp_str[:10]
    token = str(row.get("token") or "")
    return token.replace("sample_", "")[:10]


if __name__ == "__main__":
    main()
