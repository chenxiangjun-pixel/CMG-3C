"""Utilities for Transformer Mask Decoder."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def init_queries(num_queries: int, dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(num_queries, dim, device=device)


def build_foreground_guidance(logits: torch.Tensor, downsample: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    fg = probs[:, 1:].sum(dim=1, keepdim=True)
    if downsample > 1:
        fg = F.avg_pool3d(fg, kernel_size=downsample, stride=downsample, padding=0)
    return fg


def match_masks(target: torch.Tensor, num_queries: int) -> torch.Tensor:
    """Compute soft IoU targets for transformer queries."""
    with torch.no_grad():
        target_oh = F.one_hot(target.long(), num_classes=3).permute(0, 4, 1, 2, 3).float()
        # collapse liver and tumor into foreground masks per query
        foreground = target_oh[:, 1:].sum(dim=1)
        foreground = foreground.unsqueeze(1)
        foreground = foreground.repeat(1, num_queries, 1, 1, 1)
    return foreground


def build_tmd_targets(logits: torch.Tensor, labels: torch.Tensor, downsample: int, num_queries: int) -> Dict[str, torch.Tensor]:
    fg_guidance = build_foreground_guidance(logits, downsample)
    targets = match_masks(labels, num_queries)
    return {"fg_guidance": fg_guidance, "targets": targets}
