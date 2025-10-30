"""Metric computation for volumetric segmentation."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from scipy import ndimage


CLASSES = {0: "background", 1: "liver", 2: "tumor"}


def compute_dice(pred: np.ndarray, target: np.ndarray, cls: int) -> float:
    pred_mask = pred == cls
    target_mask = target == cls
    intersection = (pred_mask & target_mask).sum()
    denom = pred_mask.sum() + target_mask.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def compute_iou(pred: np.ndarray, target: np.ndarray, cls: int) -> float:
    pred_mask = pred == cls
    target_mask = target == cls
    intersection = (pred_mask & target_mask).sum()
    union = (pred_mask | target_mask).sum()
    if union == 0:
        return 1.0
    return intersection / union


def surface_distances(pred_mask: np.ndarray, target_mask: np.ndarray, voxel_spacing: np.ndarray) -> np.ndarray:
    pred_distance = ndimage.distance_transform_edt(~pred_mask, sampling=voxel_spacing)
    target_distance = ndimage.distance_transform_edt(~target_mask, sampling=voxel_spacing)
    pred_to_target = pred_distance[target_mask]
    target_to_pred = target_distance[pred_mask]
    return np.concatenate([pred_to_target, target_to_pred])


def compute_hd95_asd(pred: np.ndarray, target: np.ndarray, cls: int, spacing: np.ndarray) -> Dict[str, float]:
    pred_mask = pred == cls
    target_mask = target == cls
    if pred_mask.sum() == 0 and target_mask.sum() == 0:
        return {"hd95": 0.0, "asd": 0.0}
    if pred_mask.sum() == 0 or target_mask.sum() == 0:
        return {"hd95": float("inf"), "asd": float("inf")}
    distances = surface_distances(pred_mask, target_mask, spacing)
    hd95 = np.percentile(distances, 95)
    asd = float(distances.mean())
    return {"hd95": float(hd95), "asd": asd}


def evaluate_case(pred: np.ndarray, target: np.ndarray, spacing: np.ndarray) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for cls, name in CLASSES.items():
        if cls == 0:
            continue
        dice = compute_dice(pred, target, cls)
        iou = compute_iou(pred, target, cls)
        hd_metrics = compute_hd95_asd(pred, target, cls, spacing)
        results[name] = {"dice": dice, "iou": iou, **hd_metrics}
    avg = {metric: float(np.mean([results[name][metric] for name in results])) for metric in ["dice", "iou", "hd95", "asd"]}
    results["average"] = avg
    return results


def tensor_to_numpy(pred: torch.Tensor) -> np.ndarray:
    return pred.detach().cpu().numpy()
