"""Dynamic multi-scale context gating blocks."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class DMCMultiScaleBlock(nn.Module):
    def __init__(self, channels: int, num_scales: int, dilations: List[int], gate: str = "softmax"):
        super().__init__()
        self.num_scales = num_scales
        self.gate = gate
        self.branches = nn.ModuleList()
        for dilation in dilations[:num_scales]:
            self.branches.append(SeparableConv3d(channels, channels, kernel_size=3, dilation=dilation))
        self.gate_conv = nn.Conv3d(channels, num_scales, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        stacked = torch.stack(branch_outputs, dim=1)
        gate_logits = self.gate_conv(x)
        if self.gate == "softmax":
            gates = torch.softmax(gate_logits, dim=1)
        else:
            gates = torch.sigmoid(gate_logits)
            gates = gates / gates.sum(dim=1, keepdim=True).clamp(min=1e-6)
        gated = (stacked * gates.unsqueeze(2)).sum(dim=1)
        return gated, gates
