"""Perception-fused temporal future prediction baseline."""

from __future__ import annotations

import torch
from torch import nn


class NavLockPerceptionTemporalBaseline(nn.Module):
    """GRU baseline for future state prediction from detector features."""

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
        self.water_state_head = nn.Linear(hidden_dim, num_water_classes)
        self.water_level_head = nn.Linear(hidden_dim, 1)
        self.ship_intention_head = nn.Linear(hidden_dim, num_ship_intention_classes)

    def forward(
        self, frame_features: torch.Tensor, frame_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        encoded, _ = self.encoder(frame_features)
        lengths = frame_mask.long().sum(dim=1).clamp(min=1)
        batch_index = torch.arange(encoded.shape[0], device=encoded.device)
        final_encoded = encoded[batch_index, lengths - 1]
        final_encoded = self.norm(final_encoded)
        return {
            "upper_gate_logits": self.upper_gate_head(final_encoded),
            "lower_gate_logits": self.lower_gate_head(final_encoded),
            "water_state_logits": self.water_state_head(final_encoded),
            "water_level": self.water_level_head(final_encoded).squeeze(-1),
            "ship_intention_logits": self.ship_intention_head(final_encoded),
        }
