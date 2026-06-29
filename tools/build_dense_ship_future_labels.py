#!/usr/bin/env python3
"""Build dense per-frame ship future labels and simple rollout baselines."""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from navlock_world.berth_ship_intentions import _inside_box
from navlock_world.lock_world_state import load_lock_chamber_bounds
from tools.build_action_future_labels import load_scene_berth_slots


HORIZONS = (10, 20, 30)
MOTION_LABELS = {
    "ship_berthed",
    "ship_entering_lock",
    "ship_leaving_lock",
    "ship_static",
    "ship_moving",
}
COARSE_REGIONS = (
    "upper_gate_zone",
    "lower_gate_zone",
    "outside_lock_width",
    "between_berths",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--horizons-sec", default="10,20,30")
    parser.add_argument("--max-time-delta-sec", type=float, default=2.0)
    parser.add_argument(
        "--lock-boundary-map",
        type=Path,
        default=Path("data/maps/huaiyin_lock_boundary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/dense_ship_future_labels_valtest.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/summary_valtest_dense_ship_future_labels.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    horizons = [int(part.strip()) for part in args.horizons_sec.split(",") if part.strip()]
    sequences = load_sequences(args.data_root, splits)
    berths_by_scene = load_scene_berth_slots(args.data_root / "v1.0-trainval" / "scene.json")
    chamber = load_lock_chamber_bounds(args.lock_boundary_map)
    rows = build_dense_ship_future_labels(
        sequences,
        berths_by_scene=berths_by_scene,
        chamber=chamber,
        horizons_sec=horizons,
        max_time_delta_sec=args.max_time_delta_sec,
    )
    summary = build_summary(
        rows,
        splits=splits,
        horizons_sec=horizons,
        max_time_delta_sec=args.max_time_delta_sec,
        output=args.output,
    )
    write_jsonl(args.output, rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote={args.output}")
    print(f"wrote_summary={args.summary_output}")
    print(f"num_frames={summary['num_frames']}")
    for horizon, metrics in summary["baseline_metrics"]["constant_velocity"].items():
        print(
            f"{horizon}: berth_f1={metrics['berth_occupied_f1']:.3f} "
            f"region_acc={metrics['future_region_accuracy']:.3f} "
            f"motion_acc={metrics['motion_accuracy']:.3f}"
        )


def load_sequences(data_root: Path, splits: list[str]) -> list[dict[str, Any]]:
    sequences = []
    for split in splits:
        path = data_root / "navlock_sequences" / f"scene_sequences_{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sequence in payload.get("sequences") or []:
            item = dict(sequence)
            item["split"] = split
            sequences.append(item)
    return sequences


def build_dense_ship_future_labels(
    sequences: list[dict[str, Any]],
    *,
    berths_by_scene: dict[str, list[dict[str, Any]]],
    chamber: Optional[dict[str, float]],
    horizons_sec: list[int],
    max_time_delta_sec: float,
) -> list[dict[str, Any]]:
    rows = []
    max_delta_us = int(max_time_delta_sec * 1_000_000)
    for sequence in sequences:
        frames = sorted(
            sequence.get("frames") or [],
            key=lambda frame: (int(frame.get("timestamp") or 0), str(frame.get("sample_token"))),
        )
        timestamps = [int(frame.get("timestamp") or 0) for frame in frames]
        berths = berths_by_scene.get(sequence.get("scene_token"), [])
        previous_by_token: dict[str, dict[str, Any]] = {}
        for index, frame in enumerate(frames):
            current_ships = ships_by_token(frame, berths, chamber)
            previous_ships = ships_by_token(frames[index - 1], berths, chamber) if index > 0 else {}
            horizons = {}
            for horizon_sec in horizons_sec:
                future = find_nearest_future_frame(
                    frames,
                    timestamps,
                    int(frame.get("timestamp") or 0) + horizon_sec * 1_000_000,
                    max_delta_us=max_delta_us,
                )
                if future is None:
                    horizons[horizon_name(horizon_sec)] = None
                    continue
                horizons[horizon_name(horizon_sec)] = build_horizon_label(
                    frame,
                    future,
                    current_ships=current_ships,
                    previous_ships=previous_ships,
                    berths=berths,
                    chamber=chamber,
                    horizon_sec=horizon_sec,
                )
            rows.append(
                {
                    "sample_token": frame.get("sample_token"),
                    "split": sequence.get("split"),
                    "scene_token": sequence.get("scene_token"),
                    "timestamp": frame.get("timestamp"),
                    "timestamp_str": frame.get("timestamp_str"),
                    "direction": sequence.get("direction"),
                    "current_ship_count": len(current_ships),
                    "current_occupancy": occupancy_from_ships(current_ships, berths),
                    "dense_ship_future_targets": {"horizons": horizons},
                }
            )
            previous_by_token = current_ships
        del previous_by_token
    return rows


def find_nearest_future_frame(
    frames: list[dict[str, Any]],
    timestamps: list[int],
    target_timestamp: int,
    *,
    max_delta_us: int,
) -> Optional[dict[str, Any]]:
    index = bisect_left(timestamps, target_timestamp)
    candidates = []
    if index < len(frames):
        candidates.append(frames[index])
    if index > 0:
        candidates.append(frames[index - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda frame: abs(int(frame["timestamp"]) - target_timestamp))
    if abs(int(best["timestamp"]) - target_timestamp) > max_delta_us:
        return None
    return best


def build_horizon_label(
    current_frame: dict[str, Any],
    future_frame: dict[str, Any],
    *,
    current_ships: dict[str, dict[str, Any]],
    previous_ships: dict[str, dict[str, Any]],
    berths: list[dict[str, Any]],
    chamber: Optional[dict[str, float]],
    horizon_sec: int,
) -> dict[str, Any]:
    future_ships = ships_by_token(future_frame, berths, chamber)
    matched = []
    for token, current in current_ships.items():
        future = future_ships.get(token)
        if future is None:
            continue
        dt = max(
            1e-6,
            (int(future_frame.get("timestamp") or 0) - int(current_frame.get("timestamp") or 0))
            / 1_000_000.0,
        )
        dx = future["x"] - current["x"]
        dy = future["y"] - current["y"]
        matched.append(
            {
                "instance_token": token,
                "category": current.get("category"),
                "current_xy": [round(current["x"], 4), round(current["y"], 4)],
                "future_xy": [round(future["x"], 4), round(future["y"], 4)],
                "delta_xy": [round(dx, 4), round(dy, 4)],
                "speed_mps": round(math.hypot(dx, dy) / dt, 4),
                "current_region": current.get("region"),
                "future_region": future.get("region"),
                "current_berth_slot": current.get("berth_slot"),
                "future_berth_slot": future.get("berth_slot"),
                "target_motion_state": motion_label_from_instance(future),
                "current_motion_state": motion_label_from_instance(current),
                "visibility_level": future.get("visibility_level"),
                "occlusion_state": future.get("occlusion_state"),
            }
        )
    return {
        "horizon_sec": horizon_sec,
        "sample_token": future_frame.get("sample_token"),
        "timestamp": future_frame.get("timestamp"),
        "time_delta_sec": round(
            (
                int(future_frame.get("timestamp") or 0)
                - int(current_frame.get("timestamp") or 0)
                - horizon_sec * 1_000_000
            )
            / 1_000_000.0,
            4,
        ),
        "future_occupancy": occupancy_from_ships(future_ships, berths),
        "matched_ships": matched,
        "persistence_prediction": persistence_prediction(current_ships, berths),
        "constant_velocity_prediction": constant_velocity_prediction(
            current_ships,
            previous_ships,
            current_frame=current_frame,
            berths=berths,
            chamber=chamber,
            horizon_sec=horizon_sec,
        ),
    }


def ships_by_token(
    frame: dict[str, Any],
    berths: list[dict[str, Any]],
    chamber: Optional[dict[str, float]],
) -> dict[str, dict[str, Any]]:
    out = {}
    for inst in frame.get("instances_3d") or []:
        if not is_vessel_instance(inst):
            continue
        token = inst.get("instance_token")
        translation = inst.get("translation")
        if token is None or not isinstance(translation, list) or len(translation) < 2:
            continue
        x = float(translation[0])
        y = float(translation[1])
        out[str(token)] = {
            "instance_token": str(token),
            "category": inst.get("category"),
            "timestamp": frame.get("timestamp"),
            "x": x,
            "y": y,
            "region": coarse_region(x, y, chamber),
            "berth_slot": berth_slot_of_xy(x, y, berths) or inst.get("assigned_berth_slot"),
            "ship_intentions": list(inst.get("ship_intentions") or []),
            "attribute_names": list(inst.get("attribute_names") or []),
            "visibility_level": inst.get("visibility_level"),
            "occlusion_state": inst.get("occlusion_state"),
        }
    return out


def is_vessel_instance(item: dict[str, Any]) -> bool:
    category = str(item.get("category") or "").lower()
    return bool(item.get("ship_intentions")) or any(
        marker in category for marker in ("ship", "fleet", "vessel", "tugboat")
    )


def coarse_region(x: float, y: float, chamber: Optional[dict[str, float]]) -> str:
    if chamber is None:
        return "outside_lock_width"
    if x < chamber["x_min"] or x > chamber["x_max"]:
        return "outside_lock_width"
    if y > chamber["y_max"]:
        return "upper_gate_zone"
    if y < chamber["y_min"]:
        return "lower_gate_zone"
    return "between_berths"


def berth_slot_of_xy(x: float, y: float, berths: list[dict[str, Any]]) -> Optional[str]:
    for index, slot in enumerate(berths, start=1):
        if _inside_box(x, y, slot):
            return str(slot.get("slot_id") or f"berth_slot_{index:02d}")
    return None


def motion_label_from_instance(item: dict[str, Any]) -> str:
    for label in item.get("ship_intentions") or []:
        if label in MOTION_LABELS:
            return str(label)
    attrs = {str(value) for value in item.get("attribute_names") or []}
    if "ship.berthed" in attrs:
        return "ship_berthed"
    return "ship_static"


def occupancy_from_ships(
    ships: dict[str, dict[str, Any]],
    berths: list[dict[str, Any]],
) -> dict[str, Any]:
    berth_slots = []
    for index, slot in enumerate(berths, start=1):
        slot_id = str(slot.get("slot_id") or f"berth_slot_{index:02d}")
        tokens = [
            token for token, ship in ships.items()
            if ship.get("berth_slot") == slot_id
        ]
        berth_slots.append(
            {
                "region_id": slot_id,
                "occupied": bool(tokens),
                "ship_count": len(tokens),
                "ship_tokens": sorted(tokens),
            }
        )
    coarse = {region: [] for region in COARSE_REGIONS}
    for token, ship in ships.items():
        coarse.setdefault(str(ship.get("region")), []).append(token)
    return {
        "berth_slots": berth_slots,
        "coarse_regions": [
            {
                "region_id": region,
                "ship_count": len(tokens),
                "ship_tokens": sorted(tokens),
            }
            for region, tokens in coarse.items()
        ],
        "num_occupied_berths": sum(1 for slot in berth_slots if slot["occupied"]),
        "num_ships": len(ships),
    }


def persistence_prediction(
    current_ships: dict[str, dict[str, Any]],
    berths: list[dict[str, Any]],
) -> dict[str, Any]:
    ships = {
        token: dict(ship, predicted_motion_state=motion_label_from_instance(ship))
        for token, ship in current_ships.items()
    }
    return {
        "future_occupancy": occupancy_from_ships(ships, berths),
        "ships": {
            token: {
                "future_region": ship.get("region"),
                "future_berth_slot": ship.get("berth_slot"),
                "motion_state": ship.get("predicted_motion_state"),
            }
            for token, ship in ships.items()
        },
    }


def constant_velocity_prediction(
    current_ships: dict[str, dict[str, Any]],
    previous_ships: dict[str, dict[str, Any]],
    *,
    current_frame: dict[str, Any],
    berths: list[dict[str, Any]],
    chamber: Optional[dict[str, float]],
    horizon_sec: int,
) -> dict[str, Any]:
    predicted_ships = {}
    current_time = int(current_frame.get("timestamp") or 0)
    for token, ship in current_ships.items():
        prev = previous_ships.get(token)
        if prev is None:
            px, py = ship["x"], ship["y"]
            speed = 0.0
        else:
            prev_time = int(prev.get("timestamp") or current_time)
            dt = max(1e-6, (current_time - prev_time) / 1_000_000.0)
            vx = (ship["x"] - prev["x"]) / dt
            vy = (ship["y"] - prev["y"]) / dt
            px = ship["x"] + vx * horizon_sec
            py = ship["y"] + vy * horizon_sec
            speed = math.hypot(vx, vy)
        region = coarse_region(px, py, chamber)
        berth_slot = berth_slot_of_xy(px, py, berths)
        predicted_ships[token] = {
            **ship,
            "x": px,
            "y": py,
            "region": region,
            "berth_slot": berth_slot,
            "predicted_motion_state": predicted_motion_label(ship, berth_slot, speed),
        }
    return {
        "future_occupancy": occupancy_from_ships(predicted_ships, berths),
        "ships": {
            token: {
                "future_region": ship.get("region"),
                "future_berth_slot": ship.get("berth_slot"),
                "motion_state": ship.get("predicted_motion_state"),
            }
            for token, ship in predicted_ships.items()
        },
    }


def predicted_motion_label(
    current_ship: dict[str, Any], berth_slot: Optional[str], speed_proxy: float
) -> str:
    current_label = motion_label_from_instance(current_ship)
    if berth_slot and speed_proxy < 0.5:
        return "ship_berthed"
    if current_label in {"ship_entering_lock", "ship_leaving_lock"}:
        return current_label
    if speed_proxy >= 3.0:
        return "ship_moving"
    return current_label


def build_summary(
    rows: list[dict[str, Any]],
    *,
    splits: list[str],
    horizons_sec: list[int],
    max_time_delta_sec: float,
    output: Path,
) -> dict[str, Any]:
    return {
        "output": str(output),
        "splits": splits,
        "horizons_sec": horizons_sec,
        "max_time_delta_sec": max_time_delta_sec,
        "num_frames": len(rows),
        "horizon_coverage": horizon_coverage(rows, horizons_sec),
        "target_motion_counts": target_motion_counts(rows, horizons_sec),
        "baseline_metrics": {
            "persistence": baseline_metrics(rows, horizons_sec, "persistence_prediction"),
            "constant_velocity": baseline_metrics(
                rows, horizons_sec, "constant_velocity_prediction"
            ),
        },
        "notes": [
            "Dense ship future labels use observed future frames and annotation instance_token matching.",
            "Baselines are rollout diagnostics; they do not create counterfactual ship GT.",
            "constant_velocity uses a simple previous/current displacement proxy and should be treated as a lower-bound geometry baseline.",
        ],
    }


def horizon_coverage(rows: list[dict[str, Any]], horizons_sec: list[int]) -> dict[str, int]:
    out = {}
    for horizon in horizons_sec:
        key = horizon_name(horizon)
        out[key] = sum(
            1
            for row in rows
            if isinstance(
                row.get("dense_ship_future_targets", {}).get("horizons", {}).get(key),
                dict,
            )
        )
    return out


def target_motion_counts(
    rows: list[dict[str, Any]], horizons_sec: list[int]
) -> dict[str, dict[str, int]]:
    out = {}
    for horizon in horizons_sec:
        key = horizon_name(horizon)
        counts = Counter()
        for label in iter_horizon_labels(rows, key):
            for ship in label.get("matched_ships") or []:
                counts[str(ship.get("target_motion_state"))] += 1
        out[key] = dict(counts)
    return out


def baseline_metrics(
    rows: list[dict[str, Any]], horizons_sec: list[int], prediction_key: str
) -> dict[str, Any]:
    out = {}
    for horizon in horizons_sec:
        key = horizon_name(horizon)
        tp = fp = fn = 0
        region_hits = region_total = 0
        berth_hits = berth_total = 0
        motion_hits = motion_total = 0
        for label in iter_horizon_labels(rows, key):
            target_occ = label.get("future_occupancy") or {}
            pred = label.get(prediction_key) or {}
            pred_occ = pred.get("future_occupancy") or {}
            target_slots = occupied_slot_set(target_occ)
            pred_slots = occupied_slot_set(pred_occ)
            tp += len(target_slots & pred_slots)
            fp += len(pred_slots - target_slots)
            fn += len(target_slots - pred_slots)
            pred_ships = pred.get("ships") or {}
            for ship in label.get("matched_ships") or []:
                token = str(ship.get("instance_token"))
                pred_ship = pred_ships.get(token) or {}
                region_total += 1
                if pred_ship.get("future_region") == ship.get("future_region"):
                    region_hits += 1
                berth_total += 1
                if pred_ship.get("future_berth_slot") == ship.get("future_berth_slot"):
                    berth_hits += 1
                motion_total += 1
                if pred_ship.get("motion_state") == ship.get("target_motion_state"):
                    motion_hits += 1
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        out[key] = {
            "berth_occupied_tp": tp,
            "berth_occupied_fp": fp,
            "berth_occupied_fn": fn,
            "berth_occupied_precision": precision,
            "berth_occupied_recall": recall,
            "berth_occupied_f1": f1,
            "future_region_accuracy": safe_div(region_hits, region_total),
            "future_region_count": region_total,
            "future_berth_slot_accuracy": safe_div(berth_hits, berth_total),
            "future_berth_slot_count": berth_total,
            "motion_accuracy": safe_div(motion_hits, motion_total),
            "motion_count": motion_total,
        }
    return out


def iter_horizon_labels(rows: list[dict[str, Any]], horizon_key: str):
    for row in rows:
        label = row.get("dense_ship_future_targets", {}).get("horizons", {}).get(horizon_key)
        if isinstance(label, dict):
            yield label


def occupied_slot_set(occupancy: dict[str, Any]) -> set[str]:
    return {
        str(slot.get("region_id"))
        for slot in occupancy.get("berth_slots") or []
        if slot.get("occupied")
    }


def horizon_name(horizon_sec: int) -> str:
    return f"t_plus_{horizon_sec}s"


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
