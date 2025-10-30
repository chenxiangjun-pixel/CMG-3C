"""Visualization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path("/project")


sns.set_style("whitegrid")


def save_overlay_slice(image: np.ndarray, prediction: np.ndarray, target: np.ndarray, axis: int, index: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_slice = np.take(image, index, axis=axis)
    pred_slice = np.take(prediction, index, axis=axis)
    target_slice = np.take(target, index, axis=axis)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image_slice, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image_slice, cmap="gray")
    plt.imshow(pred_slice, cmap="jet", alpha=0.4)
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image_slice, cmap="gray")
    plt.imshow(target_slice, cmap="jet", alpha=0.4)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metrics(history: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.savefig(path)
    plt.close()


def save_attention_heatmap(attention: np.ndarray, path: Path, title: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(attention, cmap="magma")
    if title:
        plt.title(title)
    plt.savefig(path)
    plt.close()
