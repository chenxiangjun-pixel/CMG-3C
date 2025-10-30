"""Edge prediction head."""
from __future__ import annotations

import torch
import torch.nn as nn


class EdgeHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
