#!/usr/bin/env python3
"""Fuse detection-derived lock world-state priors into VLM semantic predictions.

The VLM semantic branch remains responsible for gate/water/ship semantic prediction.
``lock_occupancy`` and ``vessel_motion_flow`` are better produced by the
detection-derived geometric pipeline, then attached to the same
``prediction_json`` object for final downstream use.

Typical flow:

    python tools/derive_world_state_from_detections.py --data-root data --split test \
      --output outputs/lock_world_state/derived_test_from_detections.jsonl
    python tools/apply_lock_world_state_prior.py \
      --predictions outputs/qwen3vl_4b_eval/predictions_test24.jsonl \
      --world-state outputs/lock_world_state/derived_test_from_detections.jsonl \
      --output outputs/qwen3vl_4b_eval/predictions_test24_with_lock_world_state.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Optional

from scripts.evaluate_qwen3vl_lora_adapter import schema_check, semantic_check, write_jsonl

WORLD_STATE_KEYS = ("lock_occupancy", "vessel_motion_flow")
RECOMPUTE_CHOICES = ("if-reference-has-fields", "always", "never")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--world-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("replace", "fill"),
        default="replace",
        help="replace overwrites existing world-state fields; fill only adds missing fields.",
    )
    parser.add_argument(
        "--recompute-checks",
        choices=RECOMPUTE_CHOICES,
        default="if-reference-has-fields",
        help=(
            "When to recompute schema_check/semantic_check after injecting fields. "
            "The default avoids penalizing legacy VLM semantic references that do not "
            "contain lock_occupancy/vessel_motion_flow."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.predictions)
    world_state_by_scene = load_world_state(args.world_state)
    report = apply_lock_world_state_prior(
        rows,
        world_state_by_scene,
        mode=args.mode,
        recompute_checks=args.recompute_checks,
    )
    write_jsonl(args.output, rows)
    report.update(
        {
            "predictions": str(args.predictions),
            "world_state": str(args.world_state),
            "output": str(args.output),
        }
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_world_state(path: Path) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        state = json.loads(line)
        scene_token = state.get("scene_token")
        if scene_token:
            states[scene_token] = state
    return states


def apply_lock_world_state_prior(
    rows: list[dict[str, Any]],
    world_state_by_scene: dict[str, dict[str, Any]],
    *,
    mode: str = "replace",
    recompute_checks: str = "if-reference-has-fields",
) -> dict[str, Any]:
    if mode not in {"replace", "fill"}:
        raise ValueError(f"unsupported mode: {mode}")
    if recompute_checks not in RECOMPUTE_CHOICES:
        raise ValueError(f"unsupported recompute_checks: {recompute_checks}")

    matched_rows = 0
    prior_applied_rows = 0
    missing_world_state = 0
    recomputed_rows = 0
    changed_fields = {key: 0 for key in WORLD_STATE_KEYS}

    for row in rows:
        scene_token = scene_token_of(row)
        state = world_state_by_scene.get(scene_token or "")
        if state is None:
            missing_world_state += 1
            continue
        matched_rows += 1

        prediction = prediction_object(row)
        if prediction is None:
            prediction = {}
            row["prediction_json"] = prediction

        original_prediction = copy.deepcopy(prediction)
        changed = False
        for key in WORLD_STATE_KEYS:
            if key not in state:
                continue
            if mode == "fill" and key in prediction:
                continue
            prediction[key] = copy.deepcopy(state[key])
            changed_fields[key] += 1
            changed = True

        if not changed:
            continue

        row.setdefault("prediction_json_raw", original_prediction)
        row["lock_world_state_prior"] = {
            "mode": mode,
            "scene_token": scene_token,
            "fields": [key for key in WORLD_STATE_KEYS if key in state],
        }
        prior_applied_rows += 1

        reference = row.get("reference")
        if isinstance(reference, dict) and should_recompute(reference, recompute_checks):
            row["schema_check"] = schema_check(prediction, reference)
            row["semantic_check"] = semantic_check(prediction, reference)
            recomputed_rows += 1

    return {
        "num_rows": len(rows),
        "matched_rows": matched_rows,
        "prior_applied_rows": prior_applied_rows,
        "missing_world_state": missing_world_state,
        "recomputed_rows": recomputed_rows,
        "mode": mode,
        "recompute_checks": recompute_checks,
        "changed_fields": changed_fields,
    }


def scene_token_of(row: dict[str, Any]) -> Optional[str]:
    for source in (row.get("prediction_json"), row):
        if isinstance(source, dict) and source.get("scene_token"):
            return source["scene_token"]
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("scene_token"):
        return metadata["scene_token"]
    item_id = row.get("id")
    if isinstance(item_id, str):
        return item_id.rsplit(":", 1)[-1]
    return None


def prediction_object(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    prediction = row.get("prediction_json")
    if isinstance(prediction, dict):
        return prediction
    if isinstance(prediction, str):
        try:
            parsed = json.loads(prediction)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, dict):
            row["prediction_json"] = parsed
            return parsed
    return None


def should_recompute(reference: dict[str, Any], recompute_checks: str) -> bool:
    if recompute_checks == "always":
        return True
    if recompute_checks == "never":
        return False
    return any(key in reference for key in WORLD_STATE_KEYS)


if __name__ == "__main__":
    main()
