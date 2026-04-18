"""
Evaluation utilities for MLX models.

evaluate() estimates val loss by sampling random windows from the validation set.
Matches the evaluation approach in scripts/phase6_language.py.

BPC (bits per character) = val_loss / ln(2)
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from tsa.mlx.data import random_batch


def evaluate(
    model: nn.Module,
    data: mx.array,
    context_len: int,
    batch_size: int,
    n_batches: int = 20,
) -> float:
    """
    Estimate validation loss over n_batches random windows.

    Returns val loss in nats (not BPC). Divide by ln(2) to get BPC.

    MLX note: model.eval() disables dropout. mx.eval() is called after each
    batch to avoid accumulating a large lazy computation graph.
    """
    model.eval()
    total_loss = 0.0

    for _ in range(n_batches):
        x, y = random_batch(data, context_len, batch_size)
        logits = model(x)
        loss = mx.mean(
            nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
        )
        # Materialise before reading .item()
        mx.eval(loss)
        total_loss += loss.item()

    model.train()
    return total_loss / n_batches


def bpc(val_loss: float) -> float:
    """Convert nats loss to bits per character."""
    return val_loss / math.log(2)


def active_fraction_savings(
    mean_active_frac: float,
    n_layers: int,
) -> float:
    """
    Compute token-layer ops saved relative to baseline.

    Δ = 1 - (1 + (L-1) × α) / L
    where α = mean_active_frac, L = n_layers.

    This is the all-layers formula (includes mandatory stem block) and matches
    the metric used in the TSA paper (Table 2).
    """
    return 1.0 - (1.0 + (n_layers - 1) * mean_active_frac) / n_layers
