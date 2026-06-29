"""A lightweight temporal baseline for NavLock structured scene features."""

from __future__ import annotations

import torch
from torch import nn


class NavLockTemporalBaseline(nn.Module):
    """GRU baseline with gate, water, and ship-intention heads.

    This model is intentionally small. It validates the temporal learning
    pipeline before adding image encoders, point-cloud encoders, or local VLMs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_gate_classes: int = 4,
        num_water_classes: int = 3,
        num_ship_intention_classes: int = 3,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.upper_gate_head = nn.Linear(hidden_dim, num_gate_classes)
        self.lower_gate_head = nn.Linear(hidden_dim, num_gate_classes)
        self.water_head = nn.Linear(hidden_dim, num_water_classes)
        self.ship_intention_head = nn.Linear(hidden_dim, num_ship_intention_classes)

    def forward(self, frame_features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded, _ = self.encoder(frame_features)
        encoded = self.norm(encoded)
        return {
            "upper_gate_logits": self.upper_gate_head(encoded),
            "lower_gate_logits": self.lower_gate_head(encoded),
            "water_logits": self.water_head(encoded),
            "ship_intention_logits": self.ship_intention_head(encoded),
        }

