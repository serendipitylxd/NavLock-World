#!/usr/bin/env python3
"""Evaluate predicted lock occupancy + vessel motion flow against GT labels.

Compares a predictions JSONL to the ground-truth ``lock_world_state_<split>.jsonl``
(see :mod:`navlock_world.lock_world_state`). A prediction row may carry the
predicted object under ``prediction`` / ``parsed_prediction`` / ``pred`` /
``output`` / ``prediction_json``, or the row itself may be the prediction object.
``--section current`` evaluates the current occupancy + input-window flow;
``--section future_10s`` evaluates the future occupancy + target-window flow.

Run from the repository root:

    python tools/evaluate_lock_world_state_from_predictions.py \
      --gt outputs/lock_world_state/lock_world_state_test.jsonl \
      --pred outputs/eval/qwen_test_with_occ_flow_predictions.jsonl \
      --section future_10s
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

_PRED_KEYS = ("prediction", "parsed_prediction", "pred", "output", "prediction_json")
_SECTION_TO_WINDOW = {"current": "input_window", "future_10s": "target_window"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--section", default="future_10s", choices=("current", "future_10s"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt = _load_gt(args.gt)
    preds = _load_preds(args.pred)
    window = _SECTION_TO_WINDOW[args.section]

    evaluated = 0
    slot_correct = slot_total = 0
    tp = fp = fn = 0
    motion_correct = motion_total = 0

    for scene_token, gt_state in gt.items():
        pred_obj = preds.get(scene_token)
        if pred_obj is None:
            continue
        gt_lock = gt_state.get("lock_occupancy") or {}
        gt_motion = gt_state.get("vessel_motion_flow") or {}
        if args.section not in gt_lock and window not in gt_motion:
            continue
        evaluated += 1

        gt_occ = gt_lock.get(args.section) or {}
        pred_occ = (pred_obj.get("lock_occupancy") or {}).get(args.section) or {}
        gt_slots = _slot_occupied_map(gt_occ)
        pred_slots = _slot_occupied_map(pred_occ)
        for region_id, gt_occupied in gt_slots.items():
            pred_occupied = pred_slots.get(region_id, False)
            slot_total += 1
            if pred_occupied == gt_occupied:
                slot_correct += 1
            if gt_occupied and pred_occupied:
                tp += 1
            elif pred_occupied and not gt_occupied:
                fp += 1
            elif gt_occupied and not pred_occupied:
                fn += 1

        gt_flow = gt_motion.get(window) or []
        pred_flow = (pred_obj.get("vessel_motion_flow") or {}).get(window) or []
        pred_motion = {
            item.get("instance_token"): item.get("motion_state")
            for item in pred_flow
            if isinstance(item, dict)
        }
        for item in gt_flow:
            if not isinstance(item, dict):
                continue
            motion_total += 1
            if pred_motion.get(item.get("instance_token")) == item.get("motion_state"):
                motion_correct += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    report = {
        "section": args.section,
        "evaluated_scenes": evaluated,
        "slot_occupancy_accuracy": (slot_correct / slot_total) if slot_total else 0.0,
        "occupied_slot_prf": {"precision": precision, "recall": recall, "f1": f1},
        "vessel_motion_state_accuracy": (motion_correct / motion_total) if motion_total else 0.0,
        "counts": {
            "gt_scenes": len(gt),
            "pred_scenes": len(preds),
            "slot_total": slot_total,
            "slot_correct": slot_correct,
            "occupied_tp": tp,
            "occupied_fp": fp,
            "occupied_fn": fn,
            "motion_total": motion_total,
            "motion_correct": motion_correct,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _load_gt(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            state = json.loads(line)
            out[state.get("scene_token")] = state
    return out


def _load_preds(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        obj = _extract_pred_obj(row)
        scene_token = _scene_token_of(row, obj)
        if scene_token is not None and obj is not None:
            out[scene_token] = obj
    return out


def _extract_pred_obj(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    for key in _PRED_KEYS:
        value = row.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                value = None
        if isinstance(value, dict) and ("lock_occupancy" in value or "vessel_motion_flow" in value):
            return value
    if "lock_occupancy" in row or "vessel_motion_flow" in row:
        return row
    return None


def _scene_token_of(row: dict[str, Any], obj: Optional[dict[str, Any]]) -> Optional[str]:
    for source in (obj or {}, row):
        if source.get("scene_token"):
            return source["scene_token"]
    item_id = row.get("id") or (obj or {}).get("id")
    if isinstance(item_id, str):
        return item_id.rsplit(":", 1)[-1]
    return None


def _slot_occupied_map(occupancy: dict[str, Any]) -> dict[str, bool]:
    out = {}
    for slot in occupancy.get("berth_slots") or []:
        if isinstance(slot, dict) and slot.get("region_id") is not None:
            out[slot["region_id"]] = bool(slot.get("occupied"))
    return out


if __name__ == "__main__":
    main()
