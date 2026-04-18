"""
Content-aware sparse transformer block for TSA.

The "sparse" in TSA doesn't mean a sparse attention matrix (like Longformer).
It means sparse participation: each token participates in each block with
a learned weight in [0, 1], rather than unconditionally.

The block update is gated per-token:
    delta_attn = MultiHeadAttention(LayerNorm(h))
    delta_ffn  = FFN(LayerNorm(h + active * delta_attn))
    h_out = h + active * delta_attn + active * delta_ffn

Where active = 1 - halt_prob.

Properties:
    halt_prob = 0 (active = 1): identical to baseline TransformerBlock — full update
    halt_prob = 1 (active = 0): h_out = h — state unchanged, no effective computation
    halt_prob ∈ (0, 1):         smooth interpolation — gradient flows through halt_prob

This design is fully differentiable, so we can learn routing end-to-end.
At inference, halt_prob can be binarised (threshold at 0.5) for hard routing
and eventually replaced with actual sparse kernel calls for efficiency.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tsa.benchmarks.baseline_transformer import FeedForward, MultiHeadAttention


class SparseTransformerBlock(nn.Module):
    """
    Pre-norm transformer block with per-token gated updates.

    Reuses MultiHeadAttention and FeedForward from the baseline to ensure
    the architectural delta between baseline and TSA is isolated to the
    routing mechanism, not the attention/FFN implementation.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        h: torch.Tensor,
        halt_prob: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h:         (batch, seq_len, d_model) — input hidden states
            halt_prob: (batch, seq_len) in [0, 1] — per-token halting gate
                       0 = fully active, 1 = fully halted
            mask:      (seq_len, seq_len) causal attention mask
        Returns:
            h_out: (batch, seq_len, d_model)
        """
        # active ∈ [0, 1], shape (B, T, 1) for broadcasting over d_model
        active = (1.0 - halt_prob).unsqueeze(-1)

        h = h + active * self.attn(self.ln1(h), mask)
        h = h + active * self.ffn(self.ln2(h))
        return h
