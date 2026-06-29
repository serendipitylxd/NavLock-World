"""Loss helpers for NavLock baseline training."""

from __future__ import annotations

import torch
from torch import nn


def masked_sequence_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute masked gate, water, and ship-intention losses."""

    mask = batch["frame_mask"]
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    upper_loss = ce(
        outputs["upper_gate_logits"].reshape(-1, outputs["upper_gate_logits"].shape[-1]),
        batch["upper_gate_targets"].reshape(-1),
    )
    lower_loss = ce(
        outputs["lower_gate_logits"].reshape(-1, outputs["lower_gate_logits"].shape[-1]),
        batch["lower_gate_targets"].reshape(-1),
    )
    water_loss = ce(
        outputs["water_logits"].reshape(-1, outputs["water_logits"].shape[-1]),
        batch["water_targets"].reshape(-1),
    )

    ship_loss_raw = bce(
        outputs["ship_intention_logits"],
        batch["ship_intention_targets"],
    )
    ship_mask = mask.unsqueeze(-1).expand_as(ship_loss_raw)
    ship_loss = ship_loss_raw[ship_mask].mean()

    total = upper_loss + lower_loss + water_loss + ship_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "upper_gate_loss": float(upper_loss.detach().cpu()),
        "lower_gate_loss": float(lower_loss.detach().cpu()),
        "water_loss": float(water_loss.detach().cpu()),
        "ship_intention_loss": float(ship_loss.detach().cpu()),
    }
    return total, metrics

