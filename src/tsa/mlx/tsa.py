"""
Topological Sparse Attention (TSA) transformer in MLX.

Architecture mirrors tsa.modules.topological_attention exactly:
  Embedding
    │
  Block_0  ← always active (stem — no routing decision until a representation exists)
    │
  Router_0 → halt_prob_0  (B, T) in [0,1]
    │
  Block_1  ← gated by (1 - halt_prob_0)
    │
  ...
    │
  LayerNorm → tied head → logits

N blocks, N-1 routers.

Depth regularisation: depth_reg_weight × mean(active_fracs_per_layer)
Minimising this encourages tokens to halt early; task loss provides the counter-force.

MLX note on side effects in __call__:
  _depth_reg_loss is set as an attribute during __call__. The training loop's
  loss_fn reads it and adds it to the task loss. Because MLX builds computation
  graphs lazily (arrays carry their graph as a reference), _depth_reg_loss retains
  gradient connectivity to the router weights even after being stored as an attribute.
  Gradients flow correctly as long as loss_fn accesses _depth_reg_loss within the
  same nn.value_and_grad execution context as the model's __call__.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from tsa.mlx.model import FeedForward, MultiHeadAttention


@dataclass
class TSAConfig:
    vocab_size: int = 65
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    context_len: int = 128
    dropout: float = 0.1
    depth_reg_weight: float = 0.001  # λ: tradeoff between efficiency and accuracy


class TokenRouter(nn.Module):
    """
    Lightweight per-token halting router: Linear → ReLU → Linear → Sigmoid.

    Hidden dim = d_model // 4 — routing should cost << one transformer block.
    Final bias = -1.0 at init — mild preference for continuing (not halting)
    at the start of training, before the router has learned anything.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = max(d_model // 4, 16)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        # Initialise bias to -1.0 so sigmoid(-1) ≈ 0.27 halt_prob at init
        self.fc2.bias = mx.full((1,), -1.0)

    def __call__(self, h: mx.array) -> mx.array:
        """
        Args:
            h: (B, T, d_model)
        Returns:
            halt_prob: (B, T) in [0, 1]
        """
        return mx.sigmoid(self.fc2(self.act(self.fc1(h))).squeeze(-1))


class SparseBlock(nn.Module):
    """
    Pre-norm transformer block with per-token gated updates.

    Update rule:
        active = (1 - halt_prob)   shape: (B, T, 1) for broadcasting
        h = h + active * attn(ln1(h))
        h = h + active * ffn(ln2(h))

    halt_prob = 0  →  identical to a standard TransformerBlock
    halt_prob = 1  →  h unchanged (token fully halted)
    Gradient flows through halt_prob at all values (smooth gating).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def __call__(self, h: mx.array, halt_prob: mx.array) -> mx.array:
        """
        Args:
            h:         (B, T, d_model)
            halt_prob: (B, T) in [0, 1]
        Returns:
            h_out: (B, T, d_model)
        """
        active = (1.0 - halt_prob)[..., None]  # (B, T, 1) for broadcasting over d_model
        h = h + active * self.attn(self.ln1(h))
        h = h + active * self.ffn(self.ln2(h))
        return h


class TSATransformer(nn.Module):
    """
    TSA transformer in MLX. Matches variable_depth.TSATransformer.

    After each __call__, two attributes are updated:
      _depth_reg_loss:        mx.array scalar (lazy) — depth regularisation term
      _last_mean_active_frac: float — mean fraction of active tokens (for logging)

    The training loop's loss_fn reads _depth_reg_loss and adds it to the task
    loss (scaled by depth_reg_weight). Because this happens within the same
    nn.value_and_grad execution, gradients flow back through _depth_reg_loss
    to the router parameters.

    _last_mean_active_frac is materialised with .item() for logging — this is
    safe because it's only read after the gradient step (after mx.eval()).
    """

    def __init__(self, config: TSAConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.token_emb = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Embedding(config.context_len, d)
        self.emb_drop = nn.Dropout(config.dropout)

        self.blocks = [
            SparseBlock(d, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ]
        self.routers = [TokenRouter(d) for _ in range(config.n_layers - 1)]

        self.ln_f = nn.LayerNorm(d)

        self._apply_init()

        # Placeholders; replaced every __call__
        self._depth_reg_loss: mx.array = mx.array(0.0)
        self._last_mean_active_frac: float = 1.0

    def _apply_init(self) -> None:
        """GPT-style init matching variable_depth.TSATransformer._init_weights."""
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

        # Block 0: always active — no routing on the first block (stem)
        zero_halt = mx.zeros((B, T))
        h = self.blocks[0](h, halt_prob=zero_halt)

        # Blocks 1..N-1: each gated by its router
        active_fracs: list[mx.array] = []
        for router, block in zip(self.routers, self.blocks[1:]):
            halt_prob = router(h)                     # (B, T)
            active_frac = 1.0 - mx.mean(halt_prob)   # scalar tensor
            active_fracs.append(active_frac)
            h = block(h, halt_prob=halt_prob)

        # Depth regularisation: mean active fraction across routing points
        if active_fracs:
            depth_reg = mx.stack(active_fracs).mean()
        else:
            depth_reg = mx.array(0.0)

        # Store the lazy array — training loop adds it to task loss within the same
        # nn.value_and_grad context, so gradient flows through it to router weights.
        # Do NOT call .item() here: that forces eager evaluation inside the grad
        # context, potentially breaking the gradient tape.
        # Instead, the training loop reads _last_mean_active_frac after mx.eval().
        self._depth_reg_loss = depth_reg
        # _last_mean_active_frac is updated after mx.eval() in the training loop
        # by explicitly evaluating _depth_reg_loss. See train.py.

        h = self.ln_f(h)
        return h @ self.token_emb.weight.T

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
