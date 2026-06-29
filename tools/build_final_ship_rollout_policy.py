#!/usr/bin/env python3
"""Build the final deployable ship-rollout policy from aggregate and per-track heads.

The final policy keeps the strongest deployable role split:

* berth occupancy and total ship count come from deployable persistence;
* motion counts come from the aggregate learned count head;
* coarse-region counts use the aggregate head at 10s and the calibrated
  per-track head at 20/30s, where track-aware rollout is stronger.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from tools.evaluate_deployable_dense_ship_rollout import (
    HORIZONS,
    metrics_for_mode,
    read_jsonl,
    write_jsonl,
)


AGGREGATE_HYBRID_MODE = "deployable_berth_learned_motion_rollout"
PER_TRACK_HYBRID_MODE = "per_track_berth_hybrid_rollout"
FINAL_MODE = "final_deployable_ship_rollout_policy"
TRACK_AWARE_COARSE_HORIZONS = {"t_plus_20s", "t_plus_30s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-predictions",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/"
            "predictions_valtest_deployable_ship_rollout_head.jsonl"
        ),
    )
    parser.add_argument(
        "--per-track-predictions",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/"
            "predictions_valtest_per_track_ship_rollout_head.jsonl"
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
        "--prediction-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/"
            "predictions_valtest_final_ship_rollout_policy.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "outputs/action_conditioned_world_model/"
            "summary_valtest_final_ship_rollout_policy.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_rows = read_jsonl(args.aggregate_predictions)
    per_track_rows = read_jsonl(args.per_track_predictions)
    dense_labels = read_jsonl(args.dense_labels)

    predictions = build_final_predictions(aggregate_rows, per_track_rows)
    summary = build_summary(
        dense_labels,
        predictions,
        aggregate_predictions=args.aggregate_predictions,
        per_track_predictions=args.per_track_predictions,
        dense_labels=args.dense_labels,
        prediction_output=args.prediction_output,
    )
    write_jsonl(args.prediction_output, predictions)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote_predictions={args.prediction_output}")
    print(f"wrote_summary={args.summary_output}")
    for horizon, metrics in summary["rollout_metrics"][FINAL_MODE].items():
        print(
            f"{horizon}: berth_f1={metrics['berth_occupied_f1']:.3f} "
            f"coarse_f1={metrics['coarse_region_count_f1']:.3f} "
            f"motion_f1={metrics['motion_count_f1']:.3f}"
        )


def build_final_predictions(
    aggregate_rows: list[dict[str, Any]],
    per_track_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    per_track_by_sample = {str(row.get("sample_token")): row for row in per_track_rows}
    output = []
    for aggregate_row in aggregate_rows:
        sample_token = str(aggregate_row.get("sample_token"))
        per_track_row = per_track_by_sample.get(sample_token)
        if per_track_row is None:
            continue
        merged = copy.deepcopy(aggregate_row)
        merged_modes = merged.setdefault("rollout_modes", {})
        merged_modes.update(copy.deepcopy(per_track_row.get("rollout_modes") or {}))
        final_horizons = {}
        for horizon in HORIZONS:
            horizon_key = horizon_name(horizon)
            final_horizons[horizon_key] = build_final_horizon_prediction(
                aggregate_row,
                per_track_row,
                horizon_key,
            )
        merged_modes[FINAL_MODE] = final_horizons
        output.append(merged)
    return output


def build_final_horizon_prediction(
    aggregate_row: dict[str, Any],
    per_track_row: dict[str, Any],
    horizon_key: str,
) -> dict[str, Any]:
    aggregate_pred = (
        aggregate_row.get("rollout_modes", {})
        .get(AGGREGATE_HYBRID_MODE, {})
        .get(horizon_key, {})
    )
    per_track_pred = (
        per_track_row.get("rollout_modes", {})
        .get(PER_TRACK_HYBRID_MODE, {})
        .get(horizon_key, {})
    )
    final = copy.deepcopy(aggregate_pred)
    final_occ = copy.deepcopy(final.get("future_occupancy") or {})
    aggregate_occ = aggregate_pred.get("future_occupancy") or {}
    per_track_occ = per_track_pred.get("future_occupancy") or {}

    final_occ["berth_slots"] = copy.deepcopy(aggregate_occ.get("berth_slots") or [])
    final_occ["num_occupied_berths"] = aggregate_occ.get("num_occupied_berths")
    final_occ["num_ships"] = aggregate_occ.get("num_ships", final.get("num_ships", 0))
    if horizon_key in TRACK_AWARE_COARSE_HORIZONS:
        final_occ["coarse_regions"] = copy.deepcopy(
            per_track_occ.get("coarse_regions") or final_occ.get("coarse_regions") or []
        )
        coarse_source = PER_TRACK_HYBRID_MODE
    else:
        coarse_source = AGGREGATE_HYBRID_MODE

    final["future_occupancy"] = final_occ
    final["motion_counts"] = copy.deepcopy(aggregate_pred.get("motion_counts") or {})
    final["num_ships"] = final_occ.get("num_ships", final.get("num_ships", 0))
    final["policy_sources"] = {
        "berth_slots": "deployable_persistence",
        "num_ships": "deployable_persistence",
        "coarse_regions": coarse_source,
        "motion_counts": AGGREGATE_HYBRID_MODE,
    }
    return final


def build_summary(
    dense_label_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    aggregate_predictions: Path,
    per_track_predictions: Path,
    dense_labels: Path,
    prediction_output: Path,
) -> dict[str, Any]:
    pred_by_sample = {str(row.get("sample_token")): row for row in predictions}
    modes = [
        "deployable_persistence",
        "dispatch_aware_rollout",
        AGGREGATE_HYBRID_MODE,
        PER_TRACK_HYBRID_MODE,
        FINAL_MODE,
    ]
    return {
        "aggregate_predictions": str(aggregate_predictions),
        "per_track_predictions": str(per_track_predictions),
        "dense_labels": str(dense_labels),
        "prediction_output": str(prediction_output),
        "num_label_frames": len(dense_label_rows),
        "matched_prediction_frames": len(predictions),
        "rollout_metrics": {
            mode: metrics_for_mode(dense_label_rows, pred_by_sample, mode)
            for mode in modes
        },
        "policy": {
            "berth_slots": "deployable persistence",
            "num_ships": "deployable persistence",
            "coarse_regions": {
                "t_plus_10s": AGGREGATE_HYBRID_MODE,
                "t_plus_20s": PER_TRACK_HYBRID_MODE,
                "t_plus_30s": PER_TRACK_HYBRID_MODE,
            },
            "motion_counts": AGGREGATE_HYBRID_MODE,
        },
        "notes": [
            "This tool does not train a new model; it composes already deployable aggregate and per-track rollout heads.",
            "Berth occupancy stays on deployable persistence because learned berth rollout is still weaker than persistence.",
            "Long-horizon coarse-region rollout uses the calibrated per-track branch to keep track-aware spatial consistency.",
            "Motion-count rollout uses the aggregate learned count head, which remains the most stable motion-count branch overall.",
        ],
    }


def horizon_name(horizon_sec: int) -> str:
    return f"t_plus_{horizon_sec}s"


if __name__ == "__main__":
    main()
