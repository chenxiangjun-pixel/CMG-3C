"""Transformer Mask Decoder implementation."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.tmd_utils import build_foreground_guidance, init_queries


class TransformerMaskDecoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_queries: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))
        encoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Conv3d(feature_dim, hidden_dim, kernel_size=1)
        self.mask_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mask_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim // 2, num_queries, kernel_size=1),
        )
        self.fuse = nn.Conv3d(feature_dim + num_queries, feature_dim, kernel_size=1)

    def forward(self, features: torch.Tensor, coarse_logits: torch.Tensor, config: dict) -> Dict[str, torch.Tensor]:
        b, c, d, h, w = features.shape
        proj = self.proj(features)
        flattened = proj.view(b, proj.shape[1], -1).permute(0, 2, 1)
        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1)
        decoder_out = self.decoder(queries, flattened)
        mask_tokens = self.mask_proj(decoder_out).permute(0, 2, 1).contiguous()
        mask_tokens = mask_tokens.view(b, -1, d, h, w)
        mask_logits = self.mask_head(mask_tokens)
        fused = torch.cat([features, mask_logits], dim=1)
        fused = self.fuse(fused)
        guidance = build_foreground_guidance(coarse_logits, config.get("fg_mask_down", 4))
        return {"refined": fused, "mask_logits": mask_logits, "guidance": guidance}
