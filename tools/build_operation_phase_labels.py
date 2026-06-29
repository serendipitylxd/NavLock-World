#!/usr/bin/env python3
"""Derive weak operation-phase labels from gate, water, and observed-action states."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MAX_GAP_SEC = 120.0
PHASE_FIELDS = ("operation_phase", "phase_start_time", "phase_end_time")
OPERATION_PHASES = (
    "all_gates_closed_idle",
    "upper_gate_open_idle",
    "lower_gate_open_idle",
    "gate_opening",
    "gate_closing",
    "filling",
    "emptying",
    "hold_uncertain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--max-gap-sec", type=float, default=DEFAULT_MAX_GAP_SEC)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/operation_phase_labels/summary.json"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("outputs/operation_phase_labels/operation_phases.jsonl"),
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
    labels = build_phase_labels(rows, max_gap_sec=args.max_gap_sec)

    if not args.no_update_sample:
        update_sample_rows(rows, labels)
        sample_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    pkl_report = []
    if not args.no_update_pkl:
        pkl_report = update_info_pkls(args.data_root, labels)

    write_phase_jsonl(args.jsonl_output, rows, labels)
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
    print(f"phase_counts={summary['phase_counts']}")
    print(f"updated_sample={not args.no_update_sample}")
    print(f"updated_pkl={not args.no_update_pkl}")


def build_phase_labels(
    rows: list[dict[str, Any]], *, max_gap_sec: float = DEFAULT_MAX_GAP_SEC
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for timeline in build_timelines(rows, max_gap_sec=max_gap_sec):
        phases = [classify_operation_phase(row) for row in timeline]
        for start, end in phase_episode_ranges(phases):
            start_time = timeline[start]["timestamp"]
            end_time = timeline[end]["timestamp"]
            for index in range(start, end + 1):
                labels[timeline[index]["token"]] = {
                    "operation_phase": phases[index],
                    "phase_start_time": start_time,
                    "phase_end_time": end_time,
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


def classify_operation_phase(row: dict[str, Any]) -> str:
    action = row.get("observed_action")
    upper = row.get("upper_gate_state")
    lower = row.get("lower_gate_state")
    water = row.get("lock_water_state")

    if water == "filling" or action == "start_filling":
        return "filling"
    if water == "emptying" or action == "start_emptying":
        return "emptying"
    if action in {"open_upper_gate", "open_lower_gate"} or "opening" in {upper, lower}:
        return "gate_opening"
    if action in {"close_upper_gate", "close_lower_gate"} or "closing" in {upper, lower}:
        return "gate_closing"
    if water == "idle" and upper == "open" and lower == "closed":
        return "upper_gate_open_idle"
    if water == "idle" and upper == "closed" and lower == "open":
        return "lower_gate_open_idle"
    if water == "idle" and upper == "closed" and lower == "closed":
        return "all_gates_closed_idle"
    return "hold_uncertain"


def phase_episode_ranges(phases: list[str]) -> list[tuple[int, int]]:
    if not phases:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(phases)):
        if phases[index] != phases[index - 1]:
            ranges.append((start, index - 1))
            start = index
    ranges.append((start, len(phases) - 1))
    return ranges


def update_sample_rows(
    rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> None:
    for row in rows:
        clear_phase_fields(row)
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
                clear_phase_fields(item)
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


def write_phase_jsonl(
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
                "observed_action": row.get("observed_action"),
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
    phase_counts = Counter(label["operation_phase"] for label in labels.values())
    return {
        "settings": {
            "sample_path": str(sample_path),
            "max_gap_sec": max_gap_sec,
            "source": "gate/water/action weak phase labels",
        },
        "num_samples": len(rows),
        "num_timelines": len(timelines),
        "timeline_lengths": {
            "min": min((len(timeline) for timeline in timelines), default=0),
            "max": max((len(timeline) for timeline in timelines), default=0),
        },
        "phase_counts": dict(sorted(phase_counts.items())),
        "episode_counts": episode_counts(labels),
        "pkl_update_report": pkl_report,
    }


def episode_counts(labels: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    seen: set[tuple[Any, Any, Any]] = set()
    for label in labels.values():
        key = (
            label.get("operation_phase"),
            label.get("phase_start_time"),
            label.get("phase_end_time"),
        )
        if key in seen:
            continue
        seen.add(key)
        counts[str(label.get("operation_phase"))] += 1
    return dict(sorted(counts.items()))


def clear_phase_fields(row: dict[str, Any]) -> None:
    for key in PHASE_FIELDS:
        row.pop(key, None)


def date_of(row: dict[str, Any]) -> str:
    timestamp_str = str(row.get("timestamp_str") or row.get("sample_idx") or "")
    if len(timestamp_str) >= 10:
        return timestamp_str[:10]
    token = str(row.get("token") or "")
    return token.replace("sample_", "")[:10]


if __name__ == "__main__":
    main()
