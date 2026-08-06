"""MLP decoder: syndrome bits → logit for the logical observable flip."""

from __future__ import annotations

import torch
from torch import nn


class MLPDecoder(nn.Module):
    def __init__(
        self,
        num_detectors: int,
        hidden: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        width = num_detectors
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
