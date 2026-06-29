#!/usr/bin/env python3
"""Derive per-frame upstream/downstream water-level proxies.

The dataset currently stores the lock-chamber water level only. For planner
preconditions we derive side-level proxies from stable open-gate plateaus:
after a gate has been open for a configurable delay, the chamber level is used
as a proxy for the outside water level on that side.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_DELAY_SEC = 30.0
DEFAULT_MATCH_THRESHOLD_M = 0.20
DEFAULT_MAX_EPISODE_GAP_SEC = 120.0
UPSTREAM_PREFIX = "upstream"
DOWNSTREAM_PREFIX = "downstream"
OLD_SIDE_WATER_FIELDS = (
    "upstream_water_level_proxy",
    "downstream_water_level_proxy",
    "upper_level_matched_proxy",
    "lower_level_matched_proxy",
    "upstream_water_level_proxy_source",
    "downstream_water_level_proxy_source",
    "upstream_water_level_proxy_confidence",
    "downstream_water_level_proxy_confidence",
    "side_water_level_proxy_delay_sec",
    "side_water_level_proxy_match_threshold_m",
)


@dataclass(frozen=True)
class OpenEpisode:
    side: str
    gate_field: str
    date: str
    start_ts: int
    end_ts: int
    start_timestamp_str: str
    end_timestamp_str: str
    num_frames: int
    num_delayed_frames: int
    median_level: float
    high_confidence: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--delay-sec", type=float, default=DEFAULT_DELAY_SEC)
    parser.add_argument(
        "--match-threshold-m", type=float, default=DEFAULT_MATCH_THRESHOLD_M
    )
    parser.add_argument(
        "--max-episode-gap-sec", type=float, default=DEFAULT_MAX_EPISODE_GAP_SEC
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/side_water_level_proxy/summary.json"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("outputs/side_water_level_proxy/side_water_level_proxy.jsonl"),
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
    rows_sorted = sorted(rows, key=lambda row: int(row["timestamp"]))

    episodes = {
        UPSTREAM_PREFIX: build_open_episodes(
            rows_sorted,
            side=UPSTREAM_PREFIX,
            gate_field="upper_gate_state",
            delay_sec=args.delay_sec,
            max_gap_sec=args.max_episode_gap_sec,
        ),
        DOWNSTREAM_PREFIX: build_open_episodes(
            rows_sorted,
            side=DOWNSTREAM_PREFIX,
            gate_field="lower_gate_state",
            delay_sec=args.delay_sec,
            max_gap_sec=args.max_episode_gap_sec,
        ),
    }

    proxies = build_proxies(
        rows,
        episodes=episodes,
        delay_sec=args.delay_sec,
        match_threshold_m=args.match_threshold_m,
    )

    if not args.no_update_sample:
        update_sample_rows(rows, proxies)
        sample_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    pkl_report = []
    if not args.no_update_pkl:
        pkl_report = update_info_pkls(args.data_root, proxies)

    write_proxy_jsonl(args.jsonl_output, rows, proxies)
    summary = build_summary(
        rows,
        proxies=proxies,
        episodes=episodes,
        delay_sec=args.delay_sec,
        match_threshold_m=args.match_threshold_m,
        max_episode_gap_sec=args.max_episode_gap_sec,
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
    print(f"upstream_high_conf_episodes={summary['episodes']['upstream']['high_confidence']}")
    print(
        f"downstream_high_conf_episodes={summary['episodes']['downstream']['high_confidence']}"
    )
    print(f"updated_sample={not args.no_update_sample}")
    print(f"updated_pkl={not args.no_update_pkl}")


def build_open_episodes(
    rows_sorted: list[dict[str, Any]],
    *,
    side: str,
    gate_field: str,
    delay_sec: float,
    max_gap_sec: float,
) -> list[OpenEpisode]:
    episodes: list[OpenEpisode] = []
    current: list[dict[str, Any]] = []
    current_date: Optional[str] = None
    last_ts: Optional[int] = None

    def close_current() -> None:
        nonlocal current, current_date, last_ts
        if current:
            episodes.append(
                make_episode(
                    current,
                    side=side,
                    gate_field=gate_field,
                    delay_sec=delay_sec,
                )
            )
        current = []
        current_date = None
        last_ts = None

    max_gap_us = int(max_gap_sec * 1_000_000)
    for row in rows_sorted:
        ts = int(row["timestamp"])
        row_date = date_of(row)
        is_open_idle = (
            row.get(gate_field) == "open"
            and row.get("lock_water_state") == "idle"
            and is_number(row.get("water_level"))
        )
        must_break = bool(
            current
            and (
                not is_open_idle
                or row_date != current_date
                or (last_ts is not None and ts - last_ts > max_gap_us)
            )
        )
        if must_break:
            close_current()
        if is_open_idle:
            if not current:
                current_date = row_date
            current.append(row)
            last_ts = ts
        elif current:
            close_current()
    close_current()
    return episodes


def make_episode(
    rows: list[dict[str, Any]],
    *,
    side: str,
    gate_field: str,
    delay_sec: float,
) -> OpenEpisode:
    start_ts = int(rows[0]["timestamp"])
    delayed_rows = [
        row for row in rows if (int(row["timestamp"]) - start_ts) / 1_000_000.0 >= delay_sec
    ]
    platform_rows = delayed_rows or rows
    levels = [float(row["water_level"]) for row in platform_rows]
    return OpenEpisode(
        side=side,
        gate_field=gate_field,
        date=date_of(rows[0]),
        start_ts=start_ts,
        end_ts=int(rows[-1]["timestamp"]),
        start_timestamp_str=rows[0].get("timestamp_str", ""),
        end_timestamp_str=rows[-1].get("timestamp_str", ""),
        num_frames=len(rows),
        num_delayed_frames=len(delayed_rows),
        median_level=round(float(statistics.median(levels)), 3),
        high_confidence=bool(delayed_rows),
    )


def build_proxies(
    rows: list[dict[str, Any]],
    *,
    episodes: dict[str, list[OpenEpisode]],
    delay_sec: float,
    match_threshold_m: float,
) -> dict[str, dict[str, Any]]:
    by_side_date = {
        side: group_episodes_by_date(side_episodes)
        for side, side_episodes in episodes.items()
    }
    proxies: dict[str, dict[str, Any]] = {}
    for row in rows:
        token = row["token"]
        upstream = choose_side_proxy(
            row,
            side=UPSTREAM_PREFIX,
            gate_field="upper_gate_state",
            date_episodes=by_side_date[UPSTREAM_PREFIX],
            delay_sec=delay_sec,
            match_threshold_m=match_threshold_m,
        )
        downstream = choose_side_proxy(
            row,
            side=DOWNSTREAM_PREFIX,
            gate_field="lower_gate_state",
            date_episodes=by_side_date[DOWNSTREAM_PREFIX],
            delay_sec=delay_sec,
            match_threshold_m=match_threshold_m,
        )
        proxies[token] = {
            "upstream_water_level": upstream["value"],
            "downstream_water_level": downstream["value"],
        }
    return proxies


def choose_side_proxy(
    row: dict[str, Any],
    *,
    side: str,
    gate_field: str,
    date_episodes: dict[str, list[OpenEpisode]],
    delay_sec: float,
    match_threshold_m: float,
) -> dict[str, Any]:
    row_date = date_of(row)
    row_ts = int(row["timestamp"])
    side_episodes = date_episodes.get(row_date, [])
    containing = containing_episode(side_episodes, row_ts)
    if (
        containing
        and row.get(gate_field) == "open"
        and row.get("lock_water_state") == "idle"
    ):
        if is_number(row.get("water_level")):
            current_level = round(float(row["water_level"]), 3)
            if not containing.high_confidence:
                return {
                    "value": current_level,
                    "source": f"short_{side}_gate_open_current_level",
                    "confidence": 0.80,
                }
            if abs(current_level - containing.median_level) > match_threshold_m:
                return {
                    "value": current_level,
                    "source": f"{side}_gate_open_current_level_outlier_correction",
                    "confidence": 1.0,
                }
        if containing.high_confidence:
            return {
                "value": containing.median_level,
                "source": f"{side}_gate_future_after_{int(delay_sec)}s_episode_median",
                "confidence": 0.85,
            }
        nearest_high = nearest_episode(side_episodes, row_ts, high_confidence_only=True)
        if nearest_high:
            return {
                "value": nearest_high.median_level,
                "source": f"nearest_same_day_{side}_open_after_{int(delay_sec)}s_episode_median",
                "confidence": 0.55,
            }
        return {
            "value": containing.median_level,
            "source": f"short_{side}_open_episode_median",
            "confidence": 0.40,
        }

    prefer_future = (
        side == UPSTREAM_PREFIX and row.get("lock_water_state") == "filling"
    ) or (
        side == DOWNSTREAM_PREFIX and row.get("lock_water_state") == "emptying"
    )
    nearest_high = nearest_episode(
        side_episodes,
        row_ts,
        high_confidence_only=True,
        prefer_future=prefer_future,
    )
    if nearest_high:
        source_direction = "future" if prefer_future and nearest_high.start_ts >= row_ts else "nearest"
        return {
            "value": nearest_high.median_level,
            "source": f"{source_direction}_same_day_{side}_open_after_{int(delay_sec)}s_episode_median",
            "confidence": 0.65 if source_direction == "nearest" else 0.75,
        }
    nearest_any = nearest_episode(side_episodes, row_ts, high_confidence_only=False)
    if nearest_any:
        return {
            "value": nearest_any.median_level,
            "source": f"nearest_same_day_short_{side}_open_episode_median",
            "confidence": 0.40,
        }
    return {"value": None, "source": "no_same_day_open_episode", "confidence": 0.0}


def update_sample_rows(
    rows: list[dict[str, Any]], proxies: dict[str, dict[str, Any]]
) -> None:
    for row in rows:
        remove_old_side_water_fields(row)
        proxy = proxies.get(row.get("token"))
        if proxy:
            row.update(proxy)


def update_info_pkls(
    data_root: Path, proxies: dict[str, dict[str, Any]]
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
                proxy = proxies.get(item.get("sample_token"))
                remove_old_side_water_fields(item)
                if not proxy:
                    continue
                matched_rows += 1
                before = {key: item.get(key) for key in proxy}
                item.update(proxy)
                if any(before[key] != item.get(key) for key in proxy):
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


def write_proxy_jsonl(
    path: Path, rows: list[dict[str, Any]], proxies: dict[str, dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: int(item["timestamp"])):
            proxy = proxies[row["token"]]
            out = {
                "sample_token": row["token"],
                "sample_idx": row.get("timestamp_str") or row.get("token", "").replace("sample_", ""),
                "timestamp": row.get("timestamp"),
                "scene_token": row.get("scene_token"),
                "upper_gate_state": row.get("upper_gate_state"),
                "lower_gate_state": row.get("lower_gate_state"),
                "lock_water_state": row.get("lock_water_state"),
                "water_level": row.get("water_level"),
                **proxy,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def build_summary(
    rows: list[dict[str, Any]],
    *,
    proxies: dict[str, dict[str, Any]],
    episodes: dict[str, list[OpenEpisode]],
    delay_sec: float,
    match_threshold_m: float,
    max_episode_gap_sec: float,
    sample_path: Path,
    pkl_report: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "settings": {
            "sample_path": str(sample_path),
            "delay_sec": delay_sec,
            "match_threshold_m": match_threshold_m,
            "max_episode_gap_sec": max_episode_gap_sec,
        },
        "num_samples": len(rows),
        "episodes": {
            side: summarize_episodes(side_episodes)
            for side, side_episodes in episodes.items()
        },
        "proxy_coverage": summarize_proxy_coverage(proxies),
        "daily_proxy_stats": summarize_daily_proxy_stats(
            rows, proxies, match_threshold_m
        ),
        "pkl_update_report": pkl_report,
    }


def summarize_episodes(episodes: list[OpenEpisode]) -> dict[str, Any]:
    by_date: dict[str, list[OpenEpisode]] = {}
    for episode in episodes:
        by_date.setdefault(episode.date, []).append(episode)
    return {
        "total": len(episodes),
        "high_confidence": sum(1 for episode in episodes if episode.high_confidence),
        "by_date": {
            date: {
                "total": len(items),
                "high_confidence": sum(1 for item in items if item.high_confidence),
                "median_levels_high_confidence": [
                    item.median_level for item in items if item.high_confidence
                ],
                "episodes": [episode_to_dict(item) for item in items],
            }
            for date, items in sorted(by_date.items())
        },
    }


def summarize_proxy_coverage(proxies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for side, value_key in (
        (UPSTREAM_PREFIX, "upstream_water_level"),
        (DOWNSTREAM_PREFIX, "downstream_water_level"),
    ):
        values = list(proxies.values())
        out[side] = {
            "non_null": sum(1 for proxy in values if proxy.get(value_key) is not None),
        }
    return out


def summarize_daily_proxy_stats(
    rows: list[dict[str, Any]],
    proxies: dict[str, dict[str, Any]],
    match_threshold_m: float,
) -> dict[str, Any]:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_date.setdefault(date_of(row), []).append(row)
    out = {}
    for date, date_rows in sorted(by_date.items()):
        out[date] = {}
        for side, key in (
            (UPSTREAM_PREFIX, "upstream_water_level"),
            (DOWNSTREAM_PREFIX, "downstream_water_level"),
        ):
            vals = [
                proxies[row["token"]].get(key)
                for row in date_rows
                if is_number(proxies[row["token"]].get(key))
            ]
            matched_vals = [
                matched(
                    row.get("water_level"),
                    proxies[row["token"]].get(key),
                    match_threshold_m,
                )
                for row in date_rows
            ]
            matched_vals = [val for val in matched_vals if val is not None]
            out[date][side] = {
                "proxy_level": numeric_stats(vals),
                "matched_true": sum(1 for val in matched_vals if val is True),
                "matched_total": len(matched_vals),
            }
    return out


def remove_old_side_water_fields(row: dict[str, Any]) -> None:
    for key in OLD_SIDE_WATER_FIELDS:
        row.pop(key, None)


def episode_to_dict(episode: OpenEpisode) -> dict[str, Any]:
    return {
        "start": episode.start_timestamp_str,
        "end": episode.end_timestamp_str,
        "num_frames": episode.num_frames,
        "num_delayed_frames": episode.num_delayed_frames,
        "median_level": episode.median_level,
        "high_confidence": episode.high_confidence,
    }


def group_episodes_by_date(
    episodes: list[OpenEpisode],
) -> dict[str, list[OpenEpisode]]:
    out: dict[str, list[OpenEpisode]] = {}
    for episode in episodes:
        out.setdefault(episode.date, []).append(episode)
    for items in out.values():
        items.sort(key=lambda episode: episode.start_ts)
    return out


def containing_episode(
    episodes: list[OpenEpisode], timestamp: int
) -> Optional[OpenEpisode]:
    for episode in episodes:
        if episode.start_ts <= timestamp <= episode.end_ts:
            return episode
    return None


def nearest_episode(
    episodes: list[OpenEpisode],
    timestamp: int,
    *,
    high_confidence_only: bool,
    prefer_future: bool = False,
) -> Optional[OpenEpisode]:
    candidates = [
        episode
        for episode in episodes
        if (episode.high_confidence or not high_confidence_only)
    ]
    if prefer_future:
        future = [episode for episode in candidates if episode.start_ts >= timestamp]
        if future:
            return min(future, key=lambda episode: episode.start_ts - timestamp)
    if not candidates:
        return None
    return min(candidates, key=lambda episode: episode_distance_us(episode, timestamp))


def episode_distance_us(episode: OpenEpisode, timestamp: int) -> int:
    if episode.start_ts <= timestamp <= episode.end_ts:
        return 0
    if timestamp < episode.start_ts:
        return episode.start_ts - timestamp
    return timestamp - episode.end_ts


def matched(
    water_level: Any, proxy_level: Any, match_threshold_m: float
) -> Optional[bool]:
    if not is_number(water_level) or not is_number(proxy_level):
        return None
    return abs(float(water_level) - float(proxy_level)) <= match_threshold_m


def date_of(row: dict[str, Any]) -> str:
    timestamp_str = str(row.get("timestamp_str") or row.get("sample_idx") or "")
    if len(timestamp_str) >= 10:
        return timestamp_str[:10]
    token = str(row.get("token") or "")
    return token.replace("sample_", "")[:10]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def numeric_stats(values: list[Any]) -> dict[str, Any]:
    numbers = [float(value) for value in values if is_number(value)]
    if not numbers:
        return {"n": 0}
    return {
        "n": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
    }


def count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
