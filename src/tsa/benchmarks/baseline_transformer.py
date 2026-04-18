"""
Standard decoder-only transformer baseline.

This is the architecture every Genesis module must beat.
Nothing novel here — just a clean, well-initialised vanilla transformer
with all the standard tricks (pre-norm, weight tying, GPT-style init).

Architecture:
    token_emb + pos_emb → N × TransformerBlock → LayerNorm → linear head

The output projection shares weights with the token embedding (weight tying).
This reduces parameters and gives the model consistent token representations
at both input and output.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim import AdamW

from tsa.core.base_model import BaseModel


class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head attention with causal masking.

    Uses a single fused QKV projection rather than three separate ones —
    this is faster on hardware because it's one large matmul instead of three small ones.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, d_model)
            mask: (seq_len, seq_len) bool — True positions are masked OUT
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape
        # Fused QKV projection then reshape to (B, T, 3, n_heads, head_dim)
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)       # each: (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)             # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, T, T)

        # Build causal mask if none provided: token i may only attend to j ≤ i
        if mask is None:
            causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            mask = ~causal  # True = mask out (upper triangle)

        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network: Linear → GELU → Dropout → Linear → Dropout.

    The 4× expansion ratio (d_ff = 4 × d_model) is standard from the original
    transformer paper. GELU outperforms ReLU slightly on language tasks.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block.

    Pre-norm (LN before attention/FFN) vs post-norm (LN after):
    Pre-norm trains more stably at larger depths because gradients flow
    directly through the residual stream without passing through LayerNorm.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class BaselineTransformer(BaseModel):
    """
    Standard decoder-only transformer. The baseline everything else must beat.

    Config params:
        vocab_size:  total vocabulary size (including special tokens)
        d_model:     embedding / hidden dimension
        n_heads:     number of attention heads
        n_layers:    number of transformer blocks
        d_ff:        feed-forward inner dimension (typically 4 × d_model)
        max_seq_len: maximum sequence length (for positional embeddings)
        dropout:     dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection: maps hidden states back to vocab logits
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output projection.
        # (Press & Wolf, 2017) — improves perplexity and halves embedding parameter count.
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """
        GPT-style initialisation:
        - Embeddings: N(0, 0.02)
        - Residual projections scaled by 1/√(2 × n_layers) to prevent the residual
          stream from growing too large as depth increases.
        """
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=residual_std)
            nn.init.normal_(block.ffn.net[-2].weight, std=residual_std)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token ids
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device)

        h = self.emb_drop(self.token_emb(x) + self.pos_emb(positions))

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        return self.head(h)

    def configure_optimizers(self, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        """
        AdamW with decoupled weight decay.

        Biases, LayerNorm params, and embeddings are excluded from weight decay —
        applying decay to these hurts performance rather than regularising the model.
        Everything else (attention weights, FFN weights) gets weight decay.
        """
        decay_params, no_decay_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ("bias", "ln", "emb")):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
        )
