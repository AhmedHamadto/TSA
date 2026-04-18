"""
Loss-weighted adaptive curriculum for character-level language modeling.

This is the language equivalent of SGC — but simplified from the toy task
version which used a Generator MLP + REINFORCE.

Why simpler here (D028):
  Toy SGC had discrete task IDs (copy/reverse/sort) → a Generator MLP could
  learn a policy over that finite action space via REINFORCE.
  Language has no natural task decomposition. Every sequence chunk is its own
  "task" with a different difficulty. The signal is direct: chunks with higher
  loss need more training.

  Using the loss directly as a curriculum weight (prioritized experience replay)
  achieves the same goal — focus compute on harder examples — without the RL
  overhead and without needing a separate generator model.

  The connection to toy SGC is preserved in spirit:
    toy SGC:   Generator allocates more budget to tasks with gap (target − acc)
    lang SGC:  Sampler allocates more budget to chunks with high current loss
  Both route training signal toward weakest areas of the model's knowledge.

LossWeightedSampler:
  - Maintains per-chunk loss estimate (EMA-updated during training)
  - Samples chunk indices proportional to loss^alpha
  - alpha=0: uniform (same as baseline)
  - alpha=1: strictly proportional to loss
  - alpha=0.6: mild prioritization (default) — avoids collapsing to hardest chunks

Usage:
    sampler = LossWeightedSampler(n_chunks=len(train_ds))
    chunk_ids = sampler.sample(batch_size)                  # during training
    sampler.update(chunk_ids, per_sample_losses)            # after loss.backward()
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class LossWeightedSampler:
    """
    Adaptive chunk sampler for language curriculum.

    Maintains a loss estimate per training chunk (EMA) and samples chunks
    with probability proportional to loss^alpha.

    Higher alpha → more aggressive prioritization of hard chunks.
    alpha=0 → uniform sampling (equivalent to static curriculum).
    """

    def __init__(
        self,
        n_chunks: int,
        alpha: float = 0.6,
        ema_decay: float = 0.9,
    ) -> None:
        self.n_chunks = n_chunks
        self.alpha = alpha
        self.ema_decay = ema_decay
        # Initialise loss estimates uniformly — all chunks equally likely at start
        self.losses = torch.ones(n_chunks)
        self.n_updates = torch.zeros(n_chunks, dtype=torch.long)

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample chunk indices proportional to loss^alpha.

        Returns (batch_size,) LongTensor of chunk indices.
        Sampling is with replacement so rare hard chunks can be repeated.
        """
        weights = self.losses.clamp(min=1e-8) ** self.alpha
        weights = weights / weights.sum()
        return torch.multinomial(weights, batch_size, replacement=True)

    def update(self, chunk_indices: torch.Tensor, losses: torch.Tensor) -> None:
        """
        EMA-update per-chunk loss estimates after a training step.

        Args:
            chunk_indices: (B,) indices of chunks in this batch
            losses:        (B,) per-sample cross-entropy loss values (detached)
        """
        for idx, loss_val in zip(chunk_indices.tolist(), losses.tolist()):
            n = int(self.n_updates[idx].item())
            if n == 0:
                # First time seeing this chunk — initialise with observed loss
                self.losses[idx] = float(loss_val)
            else:
                d = self.ema_decay
                self.losses[idx] = d * self.losses[idx] + (1.0 - d) * float(loss_val)
            self.n_updates[idx] += 1

    @property
    def coverage(self) -> float:
        """Fraction of chunks seen at least once."""
        return (self.n_updates > 0).float().mean().item()

    @property
    def effective_loss_range(self) -> tuple[float, float]:
        """(min, max) of tracked loss estimates — useful for logging."""
        seen = self.losses[self.n_updates > 0]
        if len(seen) == 0:
            return (0.0, 0.0)
        return (seen.min().item(), seen.max().item())


def compute_per_sample_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean cross-entropy loss per sequence (not aggregated over batch).

    Args:
        logits:  (B, T, V) — model output
        targets: (B, T)    — ground-truth token ids

    Returns:
        losses: (B,) — mean token-level loss per sequence
    """
    B, T, V = logits.shape
    per_token = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="none",
    )
    return per_token.reshape(B, T).mean(dim=1)
