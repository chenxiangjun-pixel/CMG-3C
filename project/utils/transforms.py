"""Data augmentation and dataset utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from monai.transforms import (
        Compose,
        RandAffined,
        RandFlipd,
        RandGaussianNoised,
        RandGaussianSmoothd,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandSpatialCropSamplesd,
        ToTensord,
    )
except ImportError:  # pragma: no cover - MONAI is optional
    Compose = None  # type: ignore


ROOT = Path("/project")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    image: torch.Tensor
    label: torch.Tensor
    edge: Optional[torch.Tensor]
    meta: Dict


class LitSDataset(Dataset):
    """Dataset for processed LiTS volumes stored as .npy files."""

    def __init__(
        self,
        cases: Sequence[Path],
        augment: bool,
        config: Dict,
    ) -> None:
        self.cases = list(cases)
        self.augment = augment
        self.config = config
        self.transforms = self._build_transforms()

    def _build_transforms(self) -> Optional[Callable]:
        if Compose is None:
            return None
        patch_size = tuple(self.config.get("patch_size", [96, 160, 160]))
        num_samples = 1 if not self.augment else 1
        spatial_keys = ["image", "label"]
        ops = [
            RandSpatialCropSamplesd(spatial_keys, roi_size=patch_size, num_samples=num_samples, random_center=True, random_size=False),
            RandFlipd(spatial_keys, prob=0.5, spatial_axis=[0, 1, 2]),
            RandAffined(
                spatial_keys,
                prob=0.3 if self.augment else 0.0,
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(3, 3, 3),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
            ),
            RandScaleIntensityd("image", factors=0.1, prob=0.5),
            RandShiftIntensityd("image", offsets=0.1, prob=0.5),
            RandGaussianNoised("image", prob=0.1),
            RandGaussianSmoothd("image", prob=0.1, sigma_x=(0.5, 1.5)),
            ToTensord(keys=["image", "label"]),
        ]
        return Compose(ops)

    def __len__(self) -> int:
        return len(self.cases)

    def _load_case(self, path: Path) -> Dict:
        arrays = np.load(path, allow_pickle=True).item()
        data = {
            "image": arrays["image"].astype(np.float32),
            "label": arrays["label"].astype(np.uint8),
            "edge": arrays.get("edge", None),
            "meta": arrays["meta"],
        }
        return data

    def __getitem__(self, idx: int) -> Sample:
        case_path = self.cases[idx]
        data = self._load_case(case_path)
        image = data["image"]
        label = data["label"]
        edge = data["edge"]

        if self.transforms is not None:
            items = {"image": image[None], "label": label[None]}
            items = self.transforms(items)
            image = items["image"]
            label = items["label"].long()
        else:
            image = torch.from_numpy(image[None])
            label = torch.from_numpy(label[None]).long()

        if edge is not None:
            edge_tensor = torch.from_numpy(edge[None].astype(np.float32))
        else:
            edge_tensor = None
        return Sample(image=image, label=label, edge=edge_tensor, meta=data["meta"])


def collate_fn(samples: List[Sample]) -> Dict:
    images = torch.stack([s.image for s in samples], dim=0)
    labels = torch.stack([s.label for s in samples], dim=0)
    edges = None
    if samples[0].edge is not None:
        edges = torch.stack([s.edge for s in samples if s.edge is not None], dim=0)
    metas = [s.meta for s in samples]
    return {"image": images, "label": labels, "edge": edges, "meta": metas}
