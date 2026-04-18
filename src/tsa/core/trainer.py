from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tsa.core.base_model import BaseModel
from tsa.core.metrics import EvalMetrics, compute_metrics
from tsa.utils.logging import ExperimentLogger


@dataclass
class TrainerConfig:
    max_epochs: int = 20
    grad_clip: float = 1.0
    eval_every: int = 500      # evaluate on val set every N steps
    log_every: int = 50        # log train loss every N steps
    warmup_steps: int = 100    # linear LR warmup from 1/warmup_steps → 1.0
    checkpoint_dir: Optional[str] = None
    device: str = "auto"       # "auto" | "cpu" | "cuda" | "mps"


class Trainer:
    """
    Unified training loop for all Genesis models.

    Every module (baseline, TSA, CLM, etc.) uses this same loop.
    Supports linear LR warmup, gradient clipping, periodic evaluation,
    and best-checkpoint saving.
    """

    def __init__(
        self,
        model: BaseModel,
        config: TrainerConfig,
        logger: ExperimentLogger,
    ) -> None:
        device = self._resolve_device(config.device)
        self.device = device
        self.model = model.to(device)
        self.config = config
        self.logger = logger
        self.step = 0
        self._base_lrs: list[float] = []

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        Train for max_epochs. Returns dict with best validation metrics.
        """
        cfg = self.config
        # Capture initial LRs before warmup modifies them
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        best_val_loss = float("inf")
        best_results: dict = {}

        for epoch in range(cfg.max_epochs):
            self.model.train()
            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.max_epochs}",
                leave=False,
            ):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                self._apply_warmup(optimizer)

                logits = self.model(x)
                task_loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-100,
                )
                # aux_loss() returns 0 for baseline; modules like TSA return
                # a depth regularization term that discourages excess compute.
                aux = self.model.aux_loss()
                if aux.device != task_loss.device:
                    aux = aux.to(task_loss.device)
                loss = task_loss + aux

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optimizer.step()
                self.step += 1

                if self.step % cfg.log_every == 0:
                    log_dict: dict = {
                        "train/loss": task_loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "step": self.step,
                        "epoch": epoch,
                    }
                    if aux.item() != 0.0:
                        log_dict["train/aux_loss"] = aux.item()
                    log_dict.update(self.model.get_extra_logs())
                    self.logger.log(log_dict)

                if self.step % cfg.eval_every == 0:
                    val = self.evaluate(val_loader)
                    self.logger.log({
                        "val/loss": val.loss,
                        "val/accuracy": val.accuracy,
                        "val/perplexity": val.perplexity,
                        "step": self.step,
                    })
                    print(
                        f"  step {self.step:>6} | "
                        f"val_loss={val.loss:.4f}  "
                        f"acc={val.accuracy:.3f}  "
                        f"ppl={val.perplexity:.2f}"
                    )
                    if val.loss < best_val_loss:
                        best_val_loss = val.loss
                        best_results = {
                            "best_val_loss": val.loss,
                            "best_val_accuracy": val.accuracy,
                            "best_val_perplexity": val.perplexity,
                            "best_step": self.step,
                        }
                        if cfg.checkpoint_dir:
                            self._save_checkpoint(cfg.checkpoint_dir)

        self.logger.finish()
        return best_results

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EvalMetrics:
        self.model.eval()
        total_loss, total_correct, total_tokens = 0.0, 0, 0

        for batch in loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            m = compute_metrics(logits, y)
            # Token-weight the loss so undersized last batches don't skew the avg
            total_loss += m.loss * m.tokens_seen
            total_correct += int(m.accuracy * m.tokens_seen)
            total_tokens += m.tokens_seen

        self.model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        return EvalMetrics(
            loss=avg_loss,
            accuracy=total_correct / max(total_tokens, 1),
            perplexity=math.exp(min(avg_loss, 100.0)),
            tokens_seen=total_tokens,
        )

    def _apply_warmup(self, optimizer: torch.optim.Optimizer) -> None:
        """Linear LR warmup: ramp from (1/warmup_steps) × base_lr to base_lr."""
        if not self._base_lrs or self.config.warmup_steps <= 0:
            return
        if self.step < self.config.warmup_steps:
            scale = (self.step + 1) / self.config.warmup_steps
            for pg, base_lr in zip(optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * scale
        elif self.step == self.config.warmup_steps:
            # Restore exact base LRs in case of float drift
            for pg, base_lr in zip(optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr

    def _save_checkpoint(self, checkpoint_dir: str) -> None:
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": self.model.state_dict(), "step": self.step},
            path / "best_checkpoint.pt",
        )
