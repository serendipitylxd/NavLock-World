#!/usr/bin/env python3
"""Evaluate dense ship future rollout from deployable full-frame world state."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


HORIZONS = (10, 20, 30)
COARSE_REGIONS = (
    "upper_gate_zone",
    "lower_gate_zone",
    "outside_lock_width",
    "between_berths",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployable-world-state",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/deployable_world_state_valtest_full_deployable.jsonl"
        ),
    )
    parser.add_argument(
        "--dense-labels",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/dense_ship_future_labels_valtest.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/summary_valtest_deployable_dense_ship_rollout.json"
        ),
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/predictions_valtest_deployable_dense_ship_rollout.jsonl"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_by_sample = {
        str(row.get("sample_token")): row for row in read_jsonl(args.deployable_world_state)
    }
    label_rows = read_jsonl(args.dense_labels)
    predictions = build_predictions(label_rows, world_by_sample)
    summary = build_summary(
        label_rows,
        predictions,
        deployable_world_state=args.deployable_world_state,
        dense_labels=args.dense_labels,
        prediction_output=args.prediction_output,
    )
    write_jsonl(args.prediction_output, predictions)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote={args.prediction_output}")
    print(f"wrote_summary={args.summary_output}")
    print(f"matched_frames={summary['matched_deployable_frames']}/{summary['num_label_frames']}")
    for mode, mode_metrics in summary["rollout_metrics"].items():
        print(mode)
        for horizon, metrics in mode_metrics.items():
            print(
                f"  {horizon}: berth_f1={metrics['berth_occupied_f1']:.3f} "
                f"coarse_f1={metrics['coarse_region_count_f1']:.3f} "
                f"motion_f1={metrics['motion_count_f1']:.3f}"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_predictions(
    label_rows: list[dict[str, Any]],
    world_by_sample: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for label in label_rows:
        sample_token = str(label.get("sample_token"))
        world = world_by_sample.get(sample_token)
        if world is None:
            continue
        modes = {
            "deployable_persistence": deployable_persistence(world),
            "dispatch_aware_rollout": dispatch_aware_rollout(world),
        }
        output.append(
            {
                "sample_token": sample_token,
                "split": label.get("split"),
                "scene_token": label.get("scene_token"),
                "timestamp": label.get("timestamp"),
                "timestamp_str": label.get("timestamp_str"),
                "rollout_modes": {
                    mode_name: {
                        horizon_name(h): mode_prediction
                        for h in HORIZONS
                    }
                    for mode_name, mode_prediction in modes.items()
                },
            }
        )
    return output


def deployable_persistence(world: dict[str, Any]) -> dict[str, Any]:
    occupancy = (world.get("lock_occupancy") or {}).get("current") or {}
    motion = (world.get("vessel_motion_flow") or {}).get("input_window") or []
    return {
        "future_occupancy": occupancy,
        "motion_counts": motion_counts(motion),
        "num_ships": int(occupancy.get("num_ships") or len(motion)),
    }


def dispatch_aware_rollout(world: dict[str, Any]) -> dict[str, Any]:
    pred = deployable_persistence(world)
    stitch = world.get("planner_feature_stitch") or {}
    phase = stitch.get("ship_operation_phase")
    motion = (world.get("vessel_motion_flow") or {}).get("input_window") or []
    counts = motion_counts(motion)
    if phase == "ship_entering":
        counts = Counter(counts)
        counts["ship_entering_lock"] = max(counts.get("ship_entering_lock", 0), 1)
        counts.pop("ship_static", None)
        counts.pop("ship_berthed", None)
        pred["motion_counts"] = dict(counts)
    elif phase == "ship_leaving":
        counts = Counter(counts)
        counts["ship_leaving_lock"] = max(counts.get("ship_leaving_lock", 0), 1)
        counts.pop("ship_static", None)
        counts.pop("ship_berthed", None)
        pred["motion_counts"] = dict(counts)
    return pred


def motion_counts(motion_items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for item in motion_items:
        label = str(item.get("motion_state") or "unknown")
        counts[label] += 1
    return dict(counts)


def build_summary(
    label_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    deployable_world_state: Path,
    dense_labels: Path,
    prediction_output: Path,
) -> dict[str, Any]:
    pred_by_sample = {str(row.get("sample_token")): row for row in predictions}
    modes = ["deployable_persistence", "dispatch_aware_rollout"]
    return {
        "deployable_world_state": str(deployable_world_state),
        "dense_labels": str(dense_labels),
        "prediction_output": str(prediction_output),
        "num_label_frames": len(label_rows),
        "matched_deployable_frames": len(predictions),
        "rollout_metrics": {
            mode: metrics_for_mode(label_rows, pred_by_sample, mode) for mode in modes
        },
        "notes": [
            "Deployable world-state uses hydro_track IDs, while dense labels use annotation instance_token IDs.",
            "This evaluation therefore scores spatial state: berth occupancy, coarse-region count, and motion-state count; it does not score token-level ship identity.",
            "dispatch_aware_rollout only adjusts motion-state counts from temporal dispatch stitch; future berth/coarse occupancy is persistence in this first bridge.",
        ],
    }


def metrics_for_mode(
    label_rows: list[dict[str, Any]],
    pred_by_sample: dict[str, dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    out = {}
    for horizon in HORIZONS:
        key = horizon_name(horizon)
        berth_tp = berth_fp = berth_fn = 0
        coarse_tp = coarse_fp = coarse_fn = 0
        motion_tp = motion_fp = motion_fn = 0
        count_abs_error = 0
        count_total = 0
        covered = 0
        for label in label_rows:
            target = (
                label.get("dense_ship_future_targets", {})
                .get("horizons", {})
                .get(key)
            )
            pred_row = pred_by_sample.get(str(label.get("sample_token")))
            if not isinstance(target, dict) or pred_row is None:
                continue
            pred = (
                pred_row.get("rollout_modes", {})
                .get(mode, {})
                .get(key, {})
            )
            if not isinstance(pred, dict):
                continue
            covered += 1
            target_occ = target.get("future_occupancy") or {}
            pred_occ = pred.get("future_occupancy") or {}
            btp, bfp, bfn = set_counts(
                occupied_slot_set(pred_occ),
                occupied_slot_set(target_occ),
            )
            berth_tp += btp
            berth_fp += bfp
            berth_fn += bfn
            ctp, cfp, cfn = multiset_counts(
                coarse_count_counter(pred_occ),
                coarse_count_counter(target_occ),
            )
            coarse_tp += ctp
            coarse_fp += cfp
            coarse_fn += cfn
            mtp, mfp, mfn = multiset_counts(
                Counter(pred.get("motion_counts") or {}),
                target_motion_counter(target),
            )
            motion_tp += mtp
            motion_fp += mfp
            motion_fn += mfn
            count_abs_error += abs(
                int(pred_occ.get("num_ships") or pred.get("num_ships") or 0)
                - int(target_occ.get("num_ships") or 0)
            )
            count_total += 1
        out[key] = {
            "num_targets": covered,
            **prf("berth_occupied", berth_tp, berth_fp, berth_fn),
            **prf("coarse_region_count", coarse_tp, coarse_fp, coarse_fn),
            **prf("motion_count", motion_tp, motion_fp, motion_fn),
            "ship_count_mae": (
                round(count_abs_error / count_total, 4) if count_total else None
            ),
            "ship_count_eval_count": count_total,
        }
    return out


def occupied_slot_set(occupancy: dict[str, Any]) -> set[str]:
    return {
        str(slot.get("region_id"))
        for slot in occupancy.get("berth_slots") or []
        if slot.get("occupied")
    }


def coarse_count_counter(occupancy: dict[str, Any]) -> Counter[str]:
    counts = Counter()
    for region in occupancy.get("coarse_regions") or []:
        region_id = str(region.get("region_id") or "unknown")
        counts[region_id] += int(region.get("ship_count") or 0)
    return counts


def target_motion_counter(target: dict[str, Any]) -> Counter[str]:
    counts = Counter()
    for ship in target.get("matched_ships") or []:
        counts[str(ship.get("target_motion_state") or "unknown")] += 1
    return counts


def set_counts(pred: set[str], target: set[str]) -> tuple[int, int, int]:
    return len(pred & target), len(pred - target), len(target - pred)


def multiset_counts(pred: Counter[str], target: Counter[str]) -> tuple[int, int, int]:
    labels = set(pred) | set(target)
    tp = fp = fn = 0
    for label in labels:
        p = int(pred.get(label, 0))
        t = int(target.get(label, 0))
        tp += min(p, t)
        if p > t:
            fp += p - t
        elif t > p:
            fn += t - p
    return tp, fp, fn


def prf(prefix: str, tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        f"{prefix}_tp": tp,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
    }


def horizon_name(horizon_sec: int) -> str:
    return f"t_plus_{horizon_sec}s"


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


if __name__ == "__main__":
    main()
