"""Loss definitions including segmentation, boundary, and regularization terms."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = pred.softmax(dim=1)
    target_oh = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
    dims = tuple(range(2, pred.dim()))
    intersection = torch.sum(pred * target_oh, dims)
    denominator = torch.sum(pred + target_oh, dims)
    loss = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    return loss.mean()


def focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None) -> torch.Tensor:
    logpt = F.log_softmax(pred, dim=1)
    target_oh = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
    pt = torch.exp(logpt)
    focal = (1 - pt) ** gamma * logpt
    if alpha is not None:
        focal = focal * alpha
    loss = -(target_oh * focal).sum(dim=1)
    return loss.mean()


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_oh = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
    return F.binary_cross_entropy_with_logits(pred, target_oh)


def boundary_dice_loss(pred_edge: torch.Tensor, target_edge: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred_edge = torch.sigmoid(pred_edge)
    dims = tuple(range(2, pred_edge.dim()))
    intersection = torch.sum(pred_edge * target_edge, dims)
    denominator = torch.sum(pred_edge + target_edge, dims)
    loss = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    return loss.mean()


def hd_surrogate_loss(pred_edge: torch.Tensor, target_edge: torch.Tensor) -> torch.Tensor:
    pred_edge = torch.sigmoid(pred_edge)
    diff = torch.abs(pred_edge - target_edge)
    return diff.mean()


def gate_entropy_loss(gates: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = gates.clamp(min=eps, max=1 - eps)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    return entropy.mean()


def tmd_iou_loss(pred_masks: torch.Tensor, target_masks: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = torch.sigmoid(pred_masks)
    intersection = (pred * target_masks).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target_masks.sum(dim=(2, 3, 4)) - intersection
    return 1.0 - (intersection + smooth) / (union + smooth)


def combine_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_pred: Optional[torch.Tensor],
    edge_target: Optional[torch.Tensor],
    gates: Optional[torch.Tensor],
    tmd_pred: Optional[torch.Tensor],
    tmd_target: Optional[torch.Tensor],
    config: dict,
) -> dict:
    losses = {}
    weights = config.get("losses", {})
    main_cfg = weights.get("main", {})
    loss = 0.0
    if main_cfg.get("dice", 0) > 0:
        dice = dice_loss(pred, target) * main_cfg["dice"]
        losses["dice"] = dice
        loss = loss + dice
    if main_cfg.get("focal", 0) > 0:
        focal = focal_loss(pred, target) * main_cfg["focal"]
        losses["focal"] = focal
        loss = loss + focal
    if main_cfg.get("bce", 0) > 0:
        bce = bce_loss(pred, target) * main_cfg["bce"]
        losses["bce"] = bce
        loss = loss + bce

    if edge_pred is not None and edge_target is not None:
        boundary_cfg = config["losses"].get("boundary", {})
        weight = boundary_cfg.get("weight", 0.0)
        if weight > 0:
            if boundary_cfg.get("type", config["model"]["skip"].get("boundary_loss_type", "boundary_dice")) == "hd":
                boundary = hd_surrogate_loss(edge_pred, edge_target) * weight
            else:
                boundary = boundary_dice_loss(edge_pred, edge_target) * weight
            losses["boundary"] = boundary
            loss = loss + boundary

    if gates is not None:
        gate_cfg = config["losses"].get("gate_entropy", {})
        weight = gate_cfg.get("weight", 0.0)
        if weight > 0:
            entropy = gate_entropy_loss(gates) * weight
            losses["gate_entropy"] = entropy
            loss = loss + entropy

    if tmd_pred is not None and tmd_target is not None:
        tmd_cfg = config["losses"].get("tmd", {})
        weight = tmd_cfg.get("iou_weight", 0.0)
        if weight > 0:
            tmd_loss = tmd_iou_loss(tmd_pred, tmd_target) * weight
            losses["tmd"] = tmd_loss
            loss = loss + tmd_loss

    losses["total"] = loss
    return losses
