"""Lightweight CNN decoder."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..blocks.eab_scag import SkipGate


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.skip_gate = SkipGate(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        gated_skip = self.skip_gate(skip, x, edge)
        x = torch.cat([x, gated_skip], dim=1)
        return self.conv(x)


class CNNDecoder(nn.Module):
    def __init__(self, channels: List[int]):
        super().__init__()
        blocks = []
        for idx in range(len(channels) - 1, 0, -1):
            in_ch = channels[idx]
            out_ch = channels[idx - 1]
            blocks.append(UpBlock(in_ch, out_ch))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, bottleneck: torch.Tensor, skips: List[torch.Tensor], edges: List[torch.Tensor]) -> torch.Tensor:
        x = bottleneck
        for block, skip, edge in zip(self.blocks, reversed(skips), reversed(edges)):
            x = block(x, skip, edge)
        return x
