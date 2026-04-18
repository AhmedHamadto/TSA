from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Abstract base for every Genesis model.

    Every research module (TSA, CLM, SGC, CKR) and the baseline transformer
    implements this interface so the Trainer can run any of them against any
    toy task without modification.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token ids
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        ...

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count parameters. Default: trainable only."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @abstractmethod
    def configure_optimizers(self, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        """Return a fully configured optimizer for this model."""
        ...

    def aux_loss(self) -> torch.Tensor:
        """
        Optional auxiliary loss added to the task loss during training.
        Override in modules that need it (e.g. TSA depth regularization).
        Default: zero (no effect on baseline).
        """
        return torch.tensor(0.0)

    def get_extra_logs(self) -> dict[str, float]:
        """
        Optional extra metrics to log alongside standard train/val metrics.
        Override to expose module-specific diagnostics (e.g. mean active fraction).
        Default: empty dict.
        """
        return {}
