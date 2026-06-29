#!/usr/bin/env python3
"""Build a VLM semantic ablation comparison between Qwen3-VL-4B and InternVL3.5-4B.

Both evaluators store per-sample ``schema_check``/``semantic_check`` blocks with
the same shape, so this loads two predictions JSONL files, reuses the shared
``summarize_results`` helper for identical metric definitions, and emits a
focused side-by-side ablation (valid JSON, schema, current/future gate and water
states, water-level MAE, ship intentions). It also reports how many InternVL rows
needed trailing-JSON repair so the comparison stays transparent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_qwen3vl_lora_adapter import summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-file",
        default=(
            "outputs/qwen3vl_4b_lora_twostage_template_fixwave4_noamp_eval/"
            "predictions_test24_768_prompt_context_20260606.jsonl"
        ),
        help="Selected Qwen3-VL-4B LoRA predictions JSONL.",
    )
    parser.add_argument(
        "--internvl-file",
        default=(
            "outputs/internvl3_5_4b_eval/"
            "predictions_test24_fp16_768_repair_guarded.jsonl"
        ),
        help="InternVL3.5-4B zero-shot predictions JSONL.",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/internvl3_5_4b_eval/"
            "comparison_qwen3vl4b_vs_internvl3_5_4b_test24.json"
        ),
        help="Output comparison JSON.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no predictions loaded from {path}")
    return rows


def state_metric(summary: dict[str, Any], path: str) -> dict[str, int]:
    return summary.get("state_semantic_matches", {}).get(path, {"correct": 0, "total": 0})


def model_block(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_results(rows)
    ship = summary.get("ship_behavior", {})
    block = {
        "file": str(path),
        "num_samples": summary["num_samples"],
        "valid_json": summary["valid_json"],
        "exact_nested_schema": summary["exact_nested_schema"],
        "json_repaired": sum(1 for row in rows if row.get("json_repaired")),
        "current_upper_gate": state_metric(summary, "current_state.upper_gate_state"),
        "current_lower_gate": state_metric(summary, "current_state.lower_gate_state"),
        "current_water_state": state_metric(summary, "current_state.water_state"),
        "future_upper_gate": state_metric(summary, "future_state_10s.upper_gate_state"),
        "future_lower_gate": state_metric(summary, "future_state_10s.lower_gate_state"),
        "future_water_state": state_metric(summary, "future_state_10s.water_state"),
        "target_water_state": state_metric(
            summary, "water_surface_dynamics.target_water_state"
        ),
        "future_water_level_mae": summary.get("numeric_mae", {}).get(
            "future_state_10s.water_level"
        ),
        "ship_intentions_exact": ship.get(
            "ship_intentions_exact", {"correct": 0, "total": 0}
        ),
    }
    return block


def fmt(metric: dict[str, int]) -> str:
    return f"{metric['correct']}/{metric['total']}"


def print_table(qwen: dict[str, Any], internvl: dict[str, Any]) -> None:
    rows = [
        ("valid JSON", f"{qwen['valid_json']}/{qwen['num_samples']}", f"{internvl['valid_json']}/{internvl['num_samples']}"),
        ("exact nested schema", f"{qwen['exact_nested_schema']}/{qwen['num_samples']}", f"{internvl['exact_nested_schema']}/{internvl['num_samples']}"),
        ("json repaired", str(qwen["json_repaired"]), str(internvl["json_repaired"])),
        ("current upper gate", fmt(qwen["current_upper_gate"]), fmt(internvl["current_upper_gate"])),
        ("current lower gate", fmt(qwen["current_lower_gate"]), fmt(internvl["current_lower_gate"])),
        ("current water state", fmt(qwen["current_water_state"]), fmt(internvl["current_water_state"])),
        ("future upper gate", fmt(qwen["future_upper_gate"]), fmt(internvl["future_upper_gate"])),
        ("future lower gate", fmt(qwen["future_lower_gate"]), fmt(internvl["future_lower_gate"])),
        ("future water state", fmt(qwen["future_water_state"]), fmt(internvl["future_water_state"])),
        ("target water state", fmt(qwen["target_water_state"]), fmt(internvl["target_water_state"])),
        ("ship intentions exact", fmt(qwen["ship_intentions_exact"]), fmt(internvl["ship_intentions_exact"])),
    ]
    width = max(len(name) for name, _, _ in rows)
    print(f"{'metric'.ljust(width)} | {'Qwen3-VL-4B LoRA':>18} | {'InternVL3.5-4B':>16}")
    print("-" * (width + 40))
    for name, q, i in rows:
        print(f"{name.ljust(width)} | {q:>18} | {i:>16}")


def main() -> None:
    args = parse_args()
    qwen_rows = load_rows(Path(args.qwen_file))
    internvl_rows = load_rows(Path(args.internvl_file))
    qwen_block = model_block(Path(args.qwen_file), qwen_rows)
    internvl_block = model_block(Path(args.internvl_file), internvl_rows)
    comparison = {
        "qwen3vl_4b_lora_selected": qwen_block,
        "internvl3_5_4b_zero_shot": internvl_block,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(comparison, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print_table(qwen_block, internvl_block)
    print(f"\nsaved={args.output}")


if __name__ == "__main__":
    main()
