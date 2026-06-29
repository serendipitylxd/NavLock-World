#!/usr/bin/env python3
"""Train a perception-fused temporal baseline for NavLock future prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navlock_world.datasets import NavLockSceneDataset
from navlock_world.models import NavLockPerceptionTemporalBaseline
from navlock_world.training import (
    PerceptionFeatureStore,
    PerceptionTemporalTensorizer,
    perception_temporal_collate,
)


class TensorizedPredictionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: NavLockSceneDataset,
        tensorizer: PerceptionTemporalTensorizer,
    ) -> None:
        self.base_dataset = base_dataset
        self.tensorizer = tensorizer

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        return self.tensorizer.tensorize_sample(self.base_dataset[index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-dir", default="outputs/perception_features")
    parser.add_argument("--output-dir", default="outputs/perception_temporal_baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--water-level-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cache = Path(args.cache_dir) / "perception_features_train.json"
    val_cache = Path(args.cache_dir) / "perception_features_val.json"
    _require_cache(train_cache)
    _require_cache(val_cache)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensorizer = PerceptionTemporalTensorizer(PerceptionFeatureStore(train_cache))
    val_tensorizer = PerceptionTemporalTensorizer(PerceptionFeatureStore(val_cache))

    train_dataset = TensorizedPredictionDataset(
        NavLockSceneDataset(data_root=args.data_root, split="train", mode="prediction"),
        train_tensorizer,
    )
    val_dataset = TensorizedPredictionDataset(
        NavLockSceneDataset(data_root=args.data_root, split="val", mode="prediction"),
        val_tensorizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=perception_temporal_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=perception_temporal_collate,
    )

    model = NavLockPerceptionTemporalBaseline(
        input_dim=train_tensorizer.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            water_level_weight=args.water_level_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            water_level_weight=args.water_level_weight,
        )
        record = {
            "epoch": epoch,
            "device": str(device),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=True))

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "feature_names": train_tensorizer.feature_names,
                    "best_val_loss": best_val,
                    "train_size": len(train_dataset),
                    "val_size": len(val_dataset),
                },
                output_dir / "best.pt",
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)
        f.write("\n")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    water_level_weight: float,
) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    num_batches = 0
    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch["features"], batch["frame_mask"])
            loss, metrics = perception_future_loss(
                outputs, batch, water_level_weight=water_level_weight
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        num_batches += 1
        iterator.set_postfix(loss=f"{metrics['loss']:.4f}")

    return {key: value / max(num_batches, 1) for key, value in totals.items()}


def perception_future_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    water_level_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    upper_loss = ce(outputs["upper_gate_logits"], batch["upper_gate_targets"])
    lower_loss = ce(outputs["lower_gate_logits"], batch["lower_gate_targets"])
    water_state_loss = ce(outputs["water_state_logits"], batch["water_state_targets"])
    water_level_loss = mse(outputs["water_level"], batch["water_level_targets"])
    ship_loss = bce(outputs["ship_intention_logits"], batch["ship_intention_targets"])
    total = (
        upper_loss
        + lower_loss
        + water_state_loss
        + water_level_weight * water_level_loss
        + ship_loss
    )

    water_level_mae = (
        outputs["water_level"].detach() - batch["water_level_targets"]
    ).abs().mean()
    metrics = {
        "loss": float(total.detach().cpu()),
        "upper_gate_loss": float(upper_loss.detach().cpu()),
        "lower_gate_loss": float(lower_loss.detach().cpu()),
        "water_state_loss": float(water_state_loss.detach().cpu()),
        "water_level_loss": float(water_level_loss.detach().cpu()),
        "water_level_mae": float(water_level_mae.detach().cpu()),
        "ship_intention_loss": float(ship_loss.detach().cpu()),
    }
    return total, metrics


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _require_cache(path: Path) -> None:
    if path.exists():
        return
    raise FileNotFoundError(
        f"missing perception cache: {path}. Build it after exporting detector "
        "predictions, for example: python3 tools/build_perception_feature_cache.py "
        f"--split {path.stem.rsplit('_', 1)[-1]}"
    )


if __name__ == "__main__":
    main()
