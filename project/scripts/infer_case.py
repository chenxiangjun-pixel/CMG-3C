"""Inference for single case."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/project")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.net import DMCEABTMDNet
from utils.config import load_yaml_config
from utils.viz import save_overlay_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer a single case")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "weights" / "best.pth"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMCEABTMDNet(cfg).to(device)
    state = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    case_path = Path(cfg["data"]["processed_dir"]) / cfg["data"].get("val_split", "val") / f"{args.case}.npy"
    if not case_path.exists():
        case_path = Path(cfg["data"]["processed_dir"]) / cfg["data"].get("test_split", "test") / f"{args.case}.npy"
    data = np.load(case_path, allow_pickle=True).item()
    image = torch.from_numpy(data["image"][None, None]).to(device)
    with torch.no_grad():
        outputs = model(image)
        logits = outputs["seg_logits"]
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
    target = data["label"]
    out_dir = Path(cfg["logging"]["dir"]) / "infer" / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    mid_slice = pred.shape[0] // 2
    save_overlay_slice(data["image"], pred, target, axis=0, index=mid_slice, path=out_dir / "overlay.png")
    np.save(out_dir / "prediction.npy", pred)
    print(f"Saved inference results to {out_dir}")


if __name__ == "__main__":
    main()
