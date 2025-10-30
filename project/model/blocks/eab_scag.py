"""Edge-aware skip connection gating."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeDetector(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, 1, kernel_size=1),
        )
        sobel_kernel = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            dtype=torch.float32,
        )
        self.register_buffer("sobel", sobel_kernel.view(1, 1, 3, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        learned = self.conv(x)
        sobel = F.conv3d(x, self.sobel.expand(x.size(1), 1, -1, -1, -1), groups=x.size(1), padding=1)
        sobel = sobel.abs().mean(dim=1, keepdim=True)
        edge = torch.sigmoid(learned + sobel)
        return edge


class SkipGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, decoder: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        x = torch.cat([skip, decoder], dim=1)
        aligned = self.align(x)
        gate = self.gate(aligned * edge)
        return skip * gate
