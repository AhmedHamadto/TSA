"""
Early Exit Transformer in MLX — baseline comparison for TSA.

Architecture:
  N transformer blocks (identical to BaselineTransformer).
  After each block l, a per-layer LayerNorm feeds into the shared tied-embedding
  projection to produce exit logits.

Training:
  All N layers are always computed. Loss = mean of CE at each exit (uniform weights).
  This is the standard patience-based / multi-exit training recipe.

  Key property: training cost is IDENTICAL to baseline (all layers always run).
  Compute savings only arise at inference. This contrasts with TSA, where soft
  gating reduces effective compute at BOTH train and inference.

Inference (per-token hard exit):
  After each block l (l≥1), compute max-softmax confidence for each token.
  Tokens where confidence > threshold stop processing and use layer l's logits
  for prediction. Remaining tokens continue through subsequent blocks.
  Block N-1 (final) always exits all remaining tokens.

  Attention remains dense throughout (all token positions needed for KV cache
  correctness), same rationale as sparse_infer.py.

Active fraction metric:
  active_frac = mean_over_tokens(exit_layer / (N-1))
  A token exiting at block 1 (earliest) contributes exit_layer/4 = 0.25 with N=6.
  active_frac = 1.0 means all tokens ran all layers (threshold too high).
  active_frac → 0 means all tokens exit at block 1 (threshold too low).

  Note: unlike TSA's formula (which uses the all-layers convention including the
  mandatory stem), EarlyExit computes active_frac across the N-1 non-stem exits.
  To align with the TLOps Δ formula in evaluate.py, use
  active_fraction_savings(active_frac, n_layers) after converting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from tsa.mlx.data import random_batch
from tsa.mlx.model import FeedForward, MultiHeadAttention, TransformerBlock


@dataclass
class EarlyExitConfig:
    vocab_size: int = 65
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    context_len: int = 128
    dropout: float = 0.1


class EarlyExitTransformer(nn.Module):
    """
    Transformer with per-layer auxiliary CE exits.

    After each of the N blocks, a dedicated LayerNorm (exit_lns[l]) normalises
    the hidden state before the tied-embedding projection produces exit logits.
    The final block's exit LN plays the role of the standard ln_f.

    Training path (__call__):
      - Computes all N blocks.
      - Stores exit logits for all N exits in _exit_logits (lazy arrays).
      - Returns final exit logits as the model output.
      - Training loss: mean of CE(exit_logits[l], y) for l in 0..N-1.

    Evaluation path (evaluate_early_exit):
      - Uses hard per-token threshold on max-softmax confidence.
      - Tokens exit as soon as confidence > threshold (from block 1 onwards).
      - Block 0 (stem) never triggers early exit — matches TSA's mandatory stem.
    """

    def __init__(self, config: EarlyExitConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.token_emb = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Embedding(config.context_len, d)
        self.emb_drop = nn.Dropout(config.dropout)

        self.blocks = [
            TransformerBlock(d, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ]

        # One exit LayerNorm per block; exit_lns[-1] doubles as the final ln_f.
        self.exit_lns = [nn.LayerNorm(d) for _ in range(config.n_layers)]

        self._apply_init()

        # Set in __call__; read by training loss function.
        self._exit_logits: list[mx.array] = []

    def _apply_init(self) -> None:
        """GPT-style init matching BaselineTransformer."""
        scale = 0.02
        self.token_emb.weight = mx.random.normal(self.token_emb.weight.shape) * scale
        self.pos_emb.weight = mx.random.normal(self.pos_emb.weight.shape) * scale

        residual_scale = scale / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            block.attn.proj.weight = (
                mx.random.normal(block.attn.proj.weight.shape) * residual_scale
            )
            block.ffn.fc2.weight = (
                mx.random.normal(block.ffn.fc2.weight.shape) * residual_scale
            )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Full forward pass (training mode — all layers always computed).

        Sets self._exit_logits: list of N lazy (B, T, vocab) tensors.
        Returns final exit's logits (same as exit_logits[-1]).
        """
        B, T = x.shape
        positions = mx.arange(T, dtype=mx.int32)
        h = self.emb_drop(self.token_emb(x) + self.pos_emb(positions))

        exit_logits: list[mx.array] = []
        for block, exit_ln in zip(self.blocks, self.exit_lns):
            h = block(h)
            logits_l = exit_ln(h) @ self.token_emb.weight.T  # (B, T, vocab)
            exit_logits.append(logits_l)

        self._exit_logits = exit_logits
        return exit_logits[-1]

    def num_params(self) -> int:
        total = 0

        def _walk(obj: object) -> None:
            nonlocal total
            if isinstance(obj, mx.array):
                total += obj.size
            elif isinstance(obj, dict):
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    _walk(v)

        _walk(self.parameters())
        return total


# ── Loss function for training ────────────────────────────────────────────────

def early_exit_loss(
    model: EarlyExitTransformer,
    x: mx.array,
    y: mx.array,
) -> mx.array:
    """
    Training loss: uniform average of CE at all N exit points.

    Calling model(x) populates model._exit_logits. Gradients flow to all exit
    LNs and through all blocks (dense backprop, same cost as baseline).
    """
    _ = model(x)  # populates _exit_logits
    V = model.config.vocab_size
    losses = [
        mx.mean(nn.losses.cross_entropy(el.reshape(-1, V), y.reshape(-1)))
        for el in model._exit_logits
    ]
    return mx.stack(losses).mean()


# ── Inference with hard per-token exit ───────────────────────────────────────

def evaluate_early_exit(
    model: EarlyExitTransformer,
    data: mx.array,
    context_len: int,
    batch_size: int,
    threshold: float,
    n_batches: int = 20,
) -> tuple[float, float]:
    """
    Evaluate using hard per-token exit. Returns (val_loss, active_frac).

    active_frac = mean fraction of the (N-1) non-stem exits that each token runs.
    A token exiting at block l (1-indexed from 1) uses l/(N-1) of post-stem compute.
    active_frac=1.0 means no early exits; active_frac=1/(N-1) means all exit at block 1.

    Architecture note: attention is always dense (all-token KV for correct semantics).
    Exit decisions only skip subsequent FFN + attention computation for that token.
    Implementation uses soft mx.where masking (no gather/scatter, no CPU-GPU sync),
    so active_frac here measures logical exits, not actual FLOPs skipped.
    The accompanying active_frac_to_tl_savings() converts to TLOps metric.

    Soft-mask approach: h always has shape (B, T, d) even for exited tokens.
    Exited tokens' keys/values remain in h for attention correctness.
    Only final_logits accumulation uses per-token masking.
    """
    model.eval()
    total_loss = 0.0
    total_active = 0.0
    n_layers = model.config.n_layers
    V = model.config.vocab_size

    for _ in range(n_batches):
        x, y = random_batch(data, context_len, batch_size)
        B, T = x.shape

        # Forward pass with exit tracking
        positions = mx.arange(T, dtype=mx.int32)
        h = model.token_emb(x) + model.pos_emb(positions)  # no dropout at eval

        # Block 0 (stem): always runs, never triggers early exit
        h = model.blocks[0](h)
        stem_logits = model.exit_lns[0](h) @ model.token_emb.weight.T  # (B, T, V)

        # active: True = token has not yet exited (still needs more processing)
        active = mx.ones((B, T), dtype=mx.float32)  # 1=active, 0=exited

        # final_logits: accumulate exit-layer logits per token
        # Initialise with stem logits (fallback if threshold never triggers)
        final_logits = stem_logits

        # ops_used: how many post-stem blocks each token ran (0..N-1)
        ops_used = mx.zeros((B, T))

        # Blocks 1..N-1 (post-stem)
        for l in range(1, n_layers):
            block = model.blocks[l]
            exit_ln = model.exit_lns[l]
            is_last = (l == n_layers - 1)

            h = block(h)  # dense — all tokens (including exited) for correct KV
            ops_used = ops_used + active  # count post-stem block for still-active tokens

            logits_l = exit_ln(h) @ model.token_emb.weight.T  # (B, T, V)

            # Confidence: max probability in softmax distribution
            p_max = mx.softmax(logits_l, axis=-1).max(axis=-1)  # (B, T)

            if is_last:
                # Last block: force exit all remaining active tokens
                exiting = active  # all still-active tokens exit now
            else:
                # Exit if active AND confident
                exiting = active * (p_max > threshold)  # (B, T), float 0/1

            # Update final_logits where exiting
            final_logits = mx.where(exiting[..., None] > 0.5, logits_l, final_logits)

            # Mark exiting tokens as inactive
            active = active * (1.0 - exiting)

        mx.eval(final_logits, ops_used)

        # Val loss using per-token exit logits
        loss = mx.mean(
            nn.losses.cross_entropy(final_logits.reshape(-1, V), y.reshape(-1))
        )
        mx.eval(loss)
        total_loss += loss.item()

        # Active fraction: mean post-stem blocks used / max post-stem blocks (N-1)
        mean_ops = ops_used.mean().item()
        # Normalise: include mandatory stem (1 block) in the convention matching TSA
        # active_frac = (1 + mean_post_stem_ops) / n_layers
        active_frac_this = (1.0 + mean_ops) / n_layers
        total_active += active_frac_this

    model.train()
    return total_loss / n_batches, total_active / n_batches
