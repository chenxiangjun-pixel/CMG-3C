"""Full network assembly."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .backbones.convnet import Encoder
from .blocks.eab_scag import EdgeDetector
from .decoders.cnn_decoder import CNNDecoder
from .decoders.tmd_decoder import TransformerMaskDecoder
from .heads.edge_head import EdgeHead
from .heads.seg_head import SegmentationHead
from ..utils.tmd_utils import build_tmd_targets


class DMCEABTMDNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        self.config = config
        base = model_cfg.get("base_channels", 32)
        num_stages = model_cfg.get("encoder", {}).get("num_stages", 4)
        self.channels = [base * (2 ** i) for i in range(num_stages)]
        self.encoder = Encoder(model_cfg["in_channels"], base, num_stages, model_cfg["encoder"].get("dmc", {}))
        self.use_eab = model_cfg.get("skip", {}).get("eab_enabled", True)
        if self.use_eab:
            self.edge_detectors = nn.ModuleList([EdgeDetector(ch) for ch in self.channels[:-1]])
        else:
            self.edge_detectors = nn.ModuleList([nn.Identity() for _ in self.channels[:-1]])
        self.cnn_decoder = None
        if model_cfg.get("decoder", {}).get("cnn", {}).get("enabled", True):
            self.cnn_decoder = CNNDecoder(self.channels)
        tmd_cfg = model_cfg.get("decoder", {}).get("tmd", {})
        self.tmd_enabled = tmd_cfg.get("enabled", True)
        if self.tmd_enabled:
            self.tmd_decoder = TransformerMaskDecoder(
                feature_dim=self.channels[0],
                hidden_dim=tmd_cfg.get("hidden_dim", 128),
                num_queries=tmd_cfg.get("num_queries", 16),
                num_heads=tmd_cfg.get("num_heads", 4),
                num_layers=tmd_cfg.get("decoder_layers", 3),
                dropout=tmd_cfg.get("dropout", 0.1),
            )
        else:
            self.tmd_decoder = None
        self.seg_head = SegmentationHead(self.channels[0], model_cfg["num_classes"])
        self.edge_head = EdgeHead(self.channels[0])

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        bottleneck, skips, gates = self.encoder(x)
        decoder_skips = skips[:-1]
        edges = []
        for det, skip in zip(self.edge_detectors, decoder_skips):
            edge_map = det(skip) if self.use_eab else torch.ones_like(skip[:, :1])
            edges.append(edge_map)
        if self.cnn_decoder is not None:
            dec = self.cnn_decoder(bottleneck, decoder_skips, edges)
        else:
            dec = bottleneck
        seg_logits = self.seg_head(dec)
        outputs: Dict[str, torch.Tensor] = {"seg_logits": seg_logits}
        edge_logits = self.edge_head(dec) if self.use_eab else None
        if edge_logits is not None:
            outputs["edge_logits"] = edge_logits
        if self.tmd_decoder is not None:
            tmd_out = self.tmd_decoder(dec, seg_logits, self.config["model"]["decoder"]["tmd"])
            outputs["tmd_logits"] = tmd_out["mask_logits"]
            outputs["refined_features"] = tmd_out["refined"]
            if labels is not None:
                tmd_cfg = self.config["model"]["decoder"]["tmd"]
                tmd_targets = build_tmd_targets(
                    seg_logits,
                    labels.squeeze(1),
                    tmd_cfg.get("fg_mask_down", 4),
                    tmd_cfg.get("num_queries", 16),
                )
                outputs["tmd_targets"] = tmd_targets["targets"]
                outputs["tmd_guidance"] = tmd_targets["fg_guidance"]
        if gates is not None:
            gates_tensor = torch.stack([g for g in gates if g is not None], dim=1) if any(g is not None for g in gates) else None
            if gates_tensor is not None:
                outputs["gates"] = gates_tensor
        return outputs
