"""Training utilities including Trainer abstraction."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .losses import combine_losses
from .metrics import evaluate_case
from .transforms import collate_fn

ROOT = Path("/project")


@dataclass
class AverageMeter:
    value: float = 0.0
    count: int = 0
    total: float = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.value = val
        self.total += val * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


@dataclass
class EMA:
    model: nn.Module
    decay: float
    shadow: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


def build_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    opt_cfg = config["training"]["optimizer"]
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_cfg["name"].lower() == "adamw":
        optimizer = optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    else:
        optimizer = optim.Adam(params, lr=opt_cfg["lr"])
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, config: dict) -> Optional[optim.lr_scheduler._LRScheduler]:
    sched_cfg = config["training"].get("scheduler", {})
    if sched_cfg.get("name", None) == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"], eta_min=sched_cfg.get("min_lr", 1e-6))
    return None


def adjust_learning_rate(optimizer: optim.Optimizer, factor: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] *= factor


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        config: dict,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.scaler = amp.GradScaler(enabled=config["training"].get("amp", True) and device.type == "cuda")
        self.start_epoch = 0
        self.best_metric = 0.0
        self.checkpoint_dir = Path(config["training"]["checkpoint"]["dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config["logging"]["dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer: Optional[SummaryWriter] = None
        if config["logging"].get("tensorboard", False):
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        self.ema = None
        if config["training"].get("ema", {}).get("enabled", False):
            self.ema = EMA(model, decay=config["training"]["ema"]["decay"])

    def train(self) -> None:
        epochs = self.config["training"]["epochs"]
        for epoch in range(self.start_epoch, epochs):
            metrics = self._train_one_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            if (epoch + 1) % self.config["training"].get("eval_interval", 5) == 0 and self.val_loader is not None:
                val_metric = self.validate(epoch)
                self._save_checkpoint(epoch, val_metric, is_best=val_metric > self.best_metric)
            else:
                self._save_checkpoint(epoch, metrics["dice"], is_best=False)

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        amp_enabled = self.scaler.is_enabled()
        grad_accum = self.config["training"].get("grad_accum", 1)
        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            edges = batch["edge"].to(self.device) if batch["edge"] is not None else None
            with amp.autocast(enabled=amp_enabled):
                outputs = self.model(images, labels if self.model.training else None)
                main_logits = outputs["seg_logits"]
                edge_logits = outputs.get("edge_logits")
                tmd_logits = outputs.get("tmd_logits")
                gates = outputs.get("gates")
                tmd_targets = outputs.get("tmd_targets")
                losses = combine_losses(
                    main_logits,
                    labels.squeeze(1),
                    edge_logits,
                    edges,
                    gates,
                    tmd_logits,
                    tmd_targets,
                    self.config,
                )
                loss = losses["total"] / grad_accum
            self.scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                if self.config["training"].get("grad_clip", None):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update()
            loss_meter.update(losses["total"].item())
            dice_meter.update(losses.get("dice", torch.tensor(0.0)).item())
            if self.tb_writer:
                global_step = epoch * len(self.train_loader) + step
                for name, value in losses.items():
                    self.tb_writer.add_scalar(f"loss/{name}", value.item(), global_step)
                self.tb_writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], global_step)
        metrics = {"loss": loss_meter.avg, "dice": dice_meter.avg}
        return metrics

    def validate(self, epoch: int) -> float:
        self.model.eval()
        if self.ema is not None:
            self.ema.apply()
        total_dice = []
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(images)
                logits = outputs["seg_logits"]
                preds = torch.argmax(logits, dim=1)
                for idx in range(preds.shape[0]):
                    result = evaluate_case(
                        preds[idx].cpu().numpy(),
                        labels[idx, 0].cpu().numpy(),
                        np.ones(3),
                    )
                    total_dice.append(result["average"]["dice"])
        avg_dice = float(sum(total_dice) / max(len(total_dice), 1))
        if self.tb_writer:
            self.tb_writer.add_scalar("val/dice", avg_dice, epoch)
        return avg_dice

    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metric": metric,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        ckpt_path = self.checkpoint_dir / f"checkpoint_{epoch:04d}.pth"
        torch.save(state, ckpt_path)
        keep_last = self.config["training"]["checkpoint"].get("keep_last", 5)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pth"))
        for old in checkpoints[:-keep_last]:
            old.unlink(missing_ok=True)
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(state, best_path)


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scaler: amp.GradScaler, path: Path, scheduler=None) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0)
