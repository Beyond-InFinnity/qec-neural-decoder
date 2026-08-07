"""CNN decoder over the space-time syndrome grid.

For the repetition code, Stim emits detectors time-major: (rounds+1) frames of
(distance-1) spatial positions. Convolving over that (T, X) grid gives the
model the locality structure a flat MLP has to learn from scratch.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvDecoder(nn.Module):
    def __init__(
        self,
        grid: tuple[int, int],
        channels: int = 64,
        depth: int = 4,
        head: int = 128,
        dilations: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.grid = grid
        # Receptive field = 1 + 2*sum(dilations); it must exceed the longest
        # error chain (~d sites) or distant chain-endpoints can't be correlated.
        dil = tuple(dilations) if dilations else (1,) * depth
        convs: list[nn.Module] = []
        in_ch = 1
        for r in dil:
            convs += [
                nn.Conv2d(in_ch, channels, kernel_size=3, padding=r, dilation=r),
                nn.ReLU(),
            ]
            in_ch = channels
        self.convs = nn.Sequential(*convs)
        t, x = grid
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * t * x, head),
            nn.ReLU(),
            nn.Linear(head, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, s = self.grid
        x = x.view(-1, 1, t, s)
        return self.head(self.convs(x)).squeeze(-1)
