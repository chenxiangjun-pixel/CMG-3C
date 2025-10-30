"""Training entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path("/project")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.net import DMCEABTMDNet
from utils.config import load_yaml_config
from utils.dataio import load_split_json
from utils.train_utils import Trainer, build_optimizer, build_scheduler
from utils.transforms import LitSDataset, collate_fn, set_determinism


CONFIG_PATH = ROOT / "config" / "default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DMC–EAB/SCAG–TMD model")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--amp", type=bool, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    return parser.parse_args()


def override_config(cfg: dict, args: argparse.Namespace) -> dict:
    if args.amp is not None:
        cfg["training"]["amp"] = bool(args.amp)
    if args.grad_accum is not None:
        cfg["training"]["grad_accum"] = args.grad_accum
    return cfg


def create_dataloaders(cfg: dict) -> tuple:
    processed_dir = Path(cfg["data"]["processed_dir"])
    splits = load_split_json(processed_dir / "splits.json")
    train_cases = [processed_dir / cfg["data"]["train_split"] / f"{case}.npy" for case in splits[cfg["data"]["train_split"]]]
    val_cases = [processed_dir / cfg["data"]["val_split"] / f"{case}.npy" for case in splits[cfg["data"]["val_split"]]]
    train_dataset = LitSDataset(train_cases, augment=True, config=cfg["data"])
    val_dataset = LitSDataset(val_cases, augment=False, config=cfg["data"])
    batch_size = cfg["data"].get("batch_size", 2)
    if not torch.cuda.is_available():
        batch_size = 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
        persistent_workers=cfg["data"].get("persistent_workers", True),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(Path(args.config))
    cfg = override_config(cfg, args)
    set_determinism(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("hardware", {}).get("gpu", True) else "cpu")
    model = DMCEABTMDNet(cfg).to(device)
    log_dir = Path(cfg["logging"]["dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    train_loader, val_loader = create_dataloaders(cfg)
    trainer = Trainer(model, optimizer, scheduler, train_loader, val_loader, device, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
