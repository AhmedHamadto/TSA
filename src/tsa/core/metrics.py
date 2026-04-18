from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EvalMetrics:
    loss: float = 0.0
    accuracy: float = 0.0
    perplexity: float = 0.0
    tokens_seen: int = 0


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> EvalMetrics:
    """
    Compute loss, token accuracy, and perplexity from logits and targets.

    Args:
        logits:  (batch, seq_len, vocab_size)
        targets: (batch, seq_len)  — positions with ignore_index are excluded
    """
    vocab_size = logits.size(-1)

    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )

    mask = targets != ignore_index
    preds = logits.argmax(dim=-1)
    correct = (preds == targets) & mask
    n_tokens = mask.sum().item()
    accuracy = correct.sum().float().item() / max(n_tokens, 1)

    return EvalMetrics(
        loss=loss.item(),
        accuracy=accuracy,
        perplexity=math.exp(min(loss.item(), 100.0)),  # clamp to avoid overflow
        tokens_seen=n_tokens,
    )
