"""Evaluation script."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path("/project")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.net import DMCEABTMDNet
from utils.config import load_yaml_config
from utils.dataio import load_split_json
from utils.metrics import evaluate_case
from utils.transforms import LitSDataset, collate_fn
from utils.viz import save_overlay_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "weights" / "best.pth"))
    return parser.parse_args()


def load_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> DMCEABTMDNet:
    model = DMCEABTMDNet(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def create_loader(cfg: dict, split: str) -> DataLoader:
    processed_dir = Path(cfg["data"]["processed_dir"])
    splits = load_split_json(processed_dir / "splits.json")
    cases = [processed_dir / split / f"{case}.npy" for case in splits[split]]
    dataset = LitSDataset(cases, augment=False, config=cfg["data"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    return loader


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, Path(args.checkpoint), device)
    loader = create_loader(cfg, args.split)
    metrics_dir = Path(cfg["logging"]["dir"]) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(cfg["logging"]["dir"]) / "vis" / args.split
    vis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    overall = {"dice": [], "iou": [], "hd95": [], "asd": []}
    for sample in loader:
        image = sample["image"].to(device)
        label = sample["label"]
        meta = sample["meta"][0]
        with torch.no_grad():
            outputs = model(image)
            logits = outputs["seg_logits"]
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
        target = label.numpy()[0, 0]
        spacing = np.array(meta.get("spacing", [1.0, 1.0, 1.0]))
        case_metrics = evaluate_case(pred, target, spacing)
        for metric in overall:
            overall[metric].append(case_metrics["average"][metric])
        rows.append({"case": meta["case_id"], **case_metrics["average"]})
        mid_slice = pred.shape[0] // 2
        save_overlay_slice(image.cpu().numpy()[0, 0], pred, target, axis=0, index=mid_slice, path=vis_dir / f"{meta['case_id']}.png")

    overall_avg = {metric: float(np.mean(values)) for metric, values in overall.items()}
    rows.append({"case": "overall", **overall_avg})
    csv_path = metrics_dir / f"{args.split}_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
