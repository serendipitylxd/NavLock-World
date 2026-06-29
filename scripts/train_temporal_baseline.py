#!/usr/bin/env python3
"""Train a lightweight structured temporal baseline for NavLock-World."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navlock_world.datasets import NavLockSceneDataset
from navlock_world.models import NavLockTemporalBaseline
from navlock_world.training.losses import masked_sequence_loss
from navlock_world.training import NavLockTensorizer, navlock_tensor_collate


class TensorizedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: NavLockSceneDataset, tensorizer: NavLockTensorizer) -> None:
        self.base_dataset = base_dataset
        self.tensorizer = tensorizer

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        return self.tensorizer.tensorize_sample(self.base_dataset[index])


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_epoch(model, loader, optimizer, device, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {}
    num_batches = 0
    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch["features"])
            loss, metrics = masked_sequence_loss(outputs, batch)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--mode", default="all", choices=["recognition", "prediction", "all"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/temporal_baseline")
    parser.add_argument("--no-observed-lock-state", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensorizer = NavLockTensorizer(
        include_observed_lock_state=not args.no_observed_lock_state
    )

    train_dataset = TensorizedDataset(
        NavLockSceneDataset(data_root=args.data_root, split="train", mode=args.mode),
        tensorizer,
    )
    val_dataset = TensorizedDataset(
        NavLockSceneDataset(data_root=args.data_root, split="val", mode=args.mode),
        tensorizer,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=navlock_tensor_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=navlock_tensor_collate,
    )

    model = NavLockTemporalBaseline(
        input_dim=tensorizer.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
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
                    "feature_names": tensorizer.feature_names,
                    "best_val_loss": best_val,
                },
                output_dir / "best.pt",
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
