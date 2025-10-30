"""Encoder backbone with DMC blocks."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from ..blocks.dmc_gate import DMCMultiScaleBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dmc_cfg: dict):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.dmc = None
        self.dmc_enabled = dmc_cfg.get("enabled", True)
        if self.dmc_enabled:
            self.dmc = DMCMultiScaleBlock(out_channels, dmc_cfg.get("num_scales", 3), dmc_cfg.get("dilations", [1, 2, 4]), gate=dmc_cfg.get("gate", "softmax"))
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        gates = None
        if self.dmc is not None:
            x, gates = self.dmc(x)
        down = self.pool(x)
        return x, down, gates


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, num_stages: int, dmc_cfg: dict):
        super().__init__()
        channels = [base_channels * (2**i) for i in range(num_stages)]
        self.stages = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.stages.append(EncoderStage(prev_channels, ch, dmc_cfg))
            prev_channels = ch

    def forward(self, x: torch.Tensor):
        skips = []
        gates = []
        out = x
        for stage in self.stages:
            skip, out, gate = stage(out)
            skips.append(skip)
            gates.append(gate)
        return out, skips, gates
