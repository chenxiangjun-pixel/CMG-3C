"""Preprocess LiTS2017 dataset."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk

import sys

ROOT = Path("/project")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import dataio

RAW_DIR = ROOT / "data" / "raw" / "LiTS2017"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_SPACING = (1.0, 1.0, 1.0)
HU_WINDOW = (-200, 250)
SEED = 2025
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LiTS2017 dataset")
    parser.add_argument("--spacing", nargs=3, type=float, default=DEFAULT_SPACING, help="Target spacing")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--normalize", choices=["zscore", "minmax"], default="zscore")
    return parser.parse_args()


def find_case_path(case_id: str) -> Tuple[Path, Path]:
    for ext in [".nii", ".nii.gz"]:
        img = RAW_DIR / f"volume-{case_id}{ext}"
        lbl = RAW_DIR / f"segmentation-{case_id}{ext}"
        if img.exists() and lbl.exists():
            return img, lbl
    raise FileNotFoundError(f"Case {case_id} not found")


def compute_edge_map(label: np.ndarray) -> np.ndarray:
    eroded = sitk.GetArrayFromImage(sitk.BinaryErode(sitk.GetImageFromArray(label > 0), [1, 1, 1]))
    edge = (label > 0).astype(np.float32) - eroded.astype(np.float32)
    edge = np.clip(edge, 0, 1)
    return edge.astype(np.float32)


def process_case(case_id: str, spacing: Tuple[float, float, float], normalize: str) -> Dict:
    img_path, lbl_path = find_case_path(case_id)
    image = dataio.read_image(img_path)
    label = dataio.read_image(lbl_path)
    img_spacing = image.GetSpacing()
    lbl_spacing = label.GetSpacing()
    for idx, (i_s, l_s) in enumerate(zip(img_spacing, lbl_spacing)):
        assert abs(i_s - l_s) < 1e-3, f"Image/label spacing mismatch at axis {idx} for case {case_id}"
    image = dataio.apply_window(image, HU_WINDOW)
    image = dataio.resample(image, spacing, is_label=False)
    label = dataio.resample(label, spacing, is_label=True)
    img_np = sitk.GetArrayFromImage(image).astype(np.float32)
    lbl_np = sitk.GetArrayFromImage(label).astype(np.uint8)
    lbl_np = dataio.ensure_binary_labels(lbl_np)
    normed, stats = dataio.normalize(img_np, normalize)
    edge = compute_edge_map(lbl_np)
    meta = {
        "case_id": case_id,
        "spacing": spacing,
        "shape": normed.shape,
        "stats": stats,
    }
    return {"image": normed, "label": lbl_np, "edge": edge, "meta": meta}


def split_cases(cases: List[str], seed: int, train_ratio: float, val_ratio: float) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    rng.shuffle(cases)
    n = len(cases)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_cases = cases[:train_end]
    val_cases = cases[train_end:val_end]
    test_cases = cases[val_end:]
    return {"train": train_cases, "val": val_cases, "test": test_cases}


def main() -> None:
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cases = dataio.list_raw_cases()
    splits = split_cases(cases, args.seed, args.train_ratio, args.val_ratio)
    dataio.save_split_json(PROCESSED_DIR / "splits.json", splits)
    for split, split_cases in splits.items():
        split_dir = PROCESSED_DIR / split
        split_dir.mkdir(exist_ok=True, parents=True)
        for case_id in split_cases:
            processed = process_case(case_id, tuple(args.spacing), args.normalize)
            out_path = split_dir / f"{case_id}.npy"
            np.save(out_path, processed)
            print(f"Processed {case_id} -> {out_path}")


if __name__ == "__main__":
    main()
