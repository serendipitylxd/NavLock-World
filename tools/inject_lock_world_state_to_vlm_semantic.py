#!/usr/bin/env python3
"""Inject lock-occupancy + vessel-motion-flow world state into VLM semantic JSONL.

Adds ``answer.lock_occupancy`` and ``answer.vessel_motion_flow`` to every VLM semantic
item (matched by ``scene_token``). With ``--add-input-context`` it also adds the
NON-LEAKY input context only -- ``input.lock_occupancy_context`` from the current
(last input frame) occupancy and ``input.vessel_motion_flow_context`` from the
input-window flow. The future (``future_10s`` / ``target_window``) state is never
written into ``input``. This is a ship-lock berth-aware world state, not a generic
3D voxel occupancy.

Run from the repository root:

    python tools/inject_lock_world_state_to_vlm_semantic.py \
      --input outputs/vlm_semantic/navlock_vlm_semantic_prediction_train.jsonl \
      --world-state outputs/lock_world_state/lock_world_state_train.jsonl \
      --output outputs/vlm_semantic/navlock_vlm_semantic_prediction_train_with_occ_flow.jsonl \
      --add-input-context
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--world-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--add-input-context", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_state = _load_world_state(args.world_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    num = 0
    matched = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            scene_token = _scene_token_of(item)
            state = world_state.get(scene_token)
            if state is not None:
                matched += 1
                _inject(item, state, add_input_context=args.add_input_context)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            num += 1

    print(f"input={args.input} world_state={args.world_state}")
    print(f"output={args.output} num={num} matched={matched}")


def _load_world_state(path: Path) -> dict[str, dict[str, Any]]:
    states = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        state = json.loads(line)
        states[state.get("scene_token")] = state
    return states


def _scene_token_of(item: dict[str, Any]) -> Optional[str]:
    if item.get("scene_token"):
        return item["scene_token"]
    item_id = item.get("id")
    if isinstance(item_id, str):
        return item_id.rsplit(":", 1)[-1]
    return None


def _inject(item: dict[str, Any], state: dict[str, Any], add_input_context: bool) -> None:
    answer = item.get("answer")
    if isinstance(answer, dict):
        answer["lock_occupancy"] = state["lock_occupancy"]
        answer["vessel_motion_flow"] = state["vessel_motion_flow"]

    if add_input_context:
        input_payload = item.get("input")
        if isinstance(input_payload, dict):
            # NON-LEAKY: current occupancy (last input frame) + input-window flow only.
            input_payload["lock_occupancy_context"] = state["lock_occupancy"]["current"]
            input_payload["vessel_motion_flow_context"] = state["vessel_motion_flow"]["input_window"]


if __name__ == "__main__":
    main()
