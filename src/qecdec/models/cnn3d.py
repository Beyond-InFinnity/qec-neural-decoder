"""3D CNN over the surface code's (C, T, H, W) syndrome volume."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SurfaceConvDecoder(nn.Module):
    """Scatters the flat detector vector into a space-time volume, then Conv3d.

    Locality prior in all three dimensions; X/Z check families as input
    channels. Receptive field per axis = 1 + 2*depth (kernel 3, no dilation) —
    keep >= max(rounds+1, d+1).
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int, int],
        flat_index: np.ndarray,
        channels: int = 48,
        depth: int = 4,
        head: int = 128,
    ):
        super().__init__()
        self.volume_shape = volume_shape
        self.register_buffer(
            "flat_index", torch.from_numpy(flat_index.astype(np.int64)), persistent=False
        )
        # GroupNorm keeps depth-5 stacks trainable (plain Conv+ReLU at depth 5
        # collapsed to constant output — frozen loss ~ln 2; see phase2 v2 log).
        convs: list[nn.Module] = []
        in_ch = volume_shape[0]
        for _ in range(depth):
            convs += [
                nn.Conv3d(in_ch, channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, channels),
                nn.ReLU(),
            ]
            in_ch = channels
        self.convs = nn.Sequential(*convs)
        c, t, h, w = volume_shape
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * t * h * w, head),
            nn.ReLU(),
            nn.Linear(head, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        volume = x.new_zeros((b, int(np.prod(self.volume_shape))))
        volume[:, self.flat_index] = x
        volume = volume.view(b, *self.volume_shape)
        return self.head(self.convs(volume)).squeeze(-1)
