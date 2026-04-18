"""
Baseline decoder-only transformer in MLX.

Architecture mirrors tsa.benchmarks.baseline_transformer exactly:
  token_emb + pos_emb → N × TransformerBlock → LayerNorm → tied output head

Key MLX differences from PyTorch:
  - __call__ not forward
  - int32 for token ids (MLX GPU dtype constraint)
  - Weight tying: h @ token_emb.weight.T (no separate Linear head module)
  - Module lists as Python lists (MLX traverses them automatically for parameters)
  - Causal mask rebuilt per sequence length (no register_buffer equivalent in MLX)
  - -1e9 instead of -inf for masking: avoids NaN in all-masked rows
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class BaselineConfig:
    vocab_size: int = 65
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    context_len: int = 128
    dropout: float = 0.1


class MultiHeadAttention(nn.Module):
    """
    Fused QKV multi-head attention with causal masking.

    Single fused QKV projection (one large matmul) instead of three separate
    projections — matches the PyTorch baseline exactly.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, C = x.shape

        # Fused QKV: (B, T, 3*d_model) → (B, T, 3, n_heads, head_dim)
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        # Extract heads: (B, n_heads, T, head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        # Scaled dot-product: (B, n_heads, T, T)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Causal mask: positions where j > i should be -1e9 before softmax
        # -1e9 instead of -inf avoids NaN when all positions in a row are masked
        mask = mx.triu(mx.ones((T, T), dtype=mx.bool_), k=1)
        attn = mx.where(mask, -1e9, attn)
        attn = mx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # Weighted sum → reshape back to (B, T, d_model)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop2 = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attn → residual; LN → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BaselineTransformer(nn.Module):
    """
    Decoder-only transformer baseline in MLX.

    Matches tsa.benchmarks.baseline_transformer.BaselineTransformer.
    GPT-style init: N(0, 0.02) for embeddings, scaled residuals (1/sqrt(2L)).
    """

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.token_emb = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Embedding(config.context_len, d)
        self.emb_drop = nn.Dropout(config.dropout)

        # Python list — MLX auto-traverses lists of nn.Module to collect parameters
        self.blocks = [
            TransformerBlock(d, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ]
        self.ln_f = nn.LayerNorm(d)
        # No separate output head Linear — use weight-tied projection in __call__

        self._apply_init()

    def _apply_init(self) -> None:
        """
        GPT-style weight init.
        Embeddings: N(0, 0.02). Residual projections: scaled by 1/sqrt(2*L).
        """
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
        Args:
            x: (B, T) int32 token ids
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = x.shape
        positions = mx.arange(T, dtype=mx.int32)
        h = self.emb_drop(self.token_emb(x) + self.pos_emb(positions))

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        # Weight-tied output projection: (B, T, d_model) @ (d_model, vocab_size)
        return h @ self.token_emb.weight.T

    def num_params(self) -> int:
        """Count total parameters by walking the parameter tree."""
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
