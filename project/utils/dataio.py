"""Data IO and preprocessing utilities for LiTS2017."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

ROOT = Path("/project")
RAW_ROOT = ROOT / "data" / "raw" / "LiTS2017"
PROCESSED_ROOT = ROOT / "data" / "processed"


@dataclass
class VolumeMetadata:
    """Metadata associated with a processed volume."""

    case_id: str
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...]
    intensity_stats: Tuple[float, float]

    def to_json(self) -> dict:
        return {
            "case_id": self.case_id,
            "spacing": list(self.spacing),
            "origin": list(self.origin),
            "direction": list(self.direction),
            "intensity_stats": list(self.intensity_stats),
        }


HU_WINDOW = (-200.0, 250.0)
EPS = 1e-6


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Load a NIfTI file and return array, affine, spacing."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    zooms = img.header.get_zooms()
    if len(zooms) >= 3:
        spacing = tuple(float(z) for z in zooms[:3])
    else:
        spacing = (1.0, 1.0, 1.0)
    return data, affine, spacing


def save_nifti(array: np.ndarray, reference: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing(reference.GetSpacing())
    image.SetDirection(reference.GetDirection())
    image.SetOrigin(reference.GetOrigin())
    sitk.WriteImage(image, str(path))


def to_lps(image: sitk.Image) -> sitk.Image:
    """Ensure image orientation is LPS."""
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    return orienter.Execute(image)


def read_image(path: Path) -> sitk.Image:
    image = sitk.ReadImage(str(path))
    image = to_lps(image)
    return image


def resample(image: sitk.Image, spacing: Tuple[float, float, float], is_label: bool) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(math.ceil(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, spacing)]
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    resample_filter.SetOutputSpacing(spacing)
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetOutputOrigin(image.GetOrigin())
    resample_filter.SetDefaultPixelValue(0)
    resampled = resample_filter.Execute(image)
    return resampled


def apply_window(image: sitk.Image, window: Tuple[float, float]) -> sitk.Image:
    lower, upper = window
    return sitk.Clamp(image, lowerBound=lower, upperBound=upper)


def zscore_normalize(array: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    mean = float(array.mean())
    std = float(array.std() + EPS)
    normed = (array - mean) / std
    return normed.astype(np.float32), (mean, std)


def minmax_normalize(array: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    arr_min = float(array.min())
    arr_max = float(array.max())
    normed = (array - arr_min) / max(arr_max - arr_min, EPS)
    return normed.astype(np.float32), (arr_min, arr_max)


def normalize(array: np.ndarray, method: str) -> Tuple[np.ndarray, Tuple[float, float]]:
    if method == "zscore":
        return zscore_normalize(array)
    if method == "minmax":
        return minmax_normalize(array)
    raise ValueError(f"Unsupported normalization method: {method}")


def ensure_binary_labels(label: np.ndarray) -> np.ndarray:
    unique = np.unique(label)
    if not set(unique).issubset({0, 1, 2}):
        label = np.clip(label, 0, 2)
    return label.astype(np.uint8)


def compute_foreground_ratio(label: np.ndarray) -> float:
    fg = float((label > 0).sum())
    total = float(label.size)
    return fg / max(total, 1.0)


def write_metadata(path: Path, entries: List[VolumeMetadata]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.to_json() for e in entries], f, indent=2)


def load_metadata(path: Path) -> List[VolumeMetadata]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [VolumeMetadata(**entry) for entry in data]


def list_raw_cases() -> List[str]:
    images = sorted(RAW_ROOT.glob("volume-*.nii*"))
    return [img.name.replace("volume-", "").split(".")[0] for img in images]


def save_split_json(path: Path, splits: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def load_split_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_spacing(image: sitk.Image, expected: Tuple[float, float, float]) -> None:
    spacing = image.GetSpacing()
    for idx, (sp, exp) in enumerate(zip(spacing, expected)):
        assert abs(sp - exp) < 1e-3, f"Spacing mismatch at axis {idx}: {sp} vs {exp}"


def summary_statistics(values: Iterable[float]) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))
