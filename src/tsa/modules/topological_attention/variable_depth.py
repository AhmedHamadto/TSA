"""
Adaptive depth mechanism — the top-level TSA model.

Architecture:
    Embedding
        │
    Block_0  ← always executed (the "stem" — no routing decision yet)
        │
    Router_0 → halt_prob_0  (B, T)
        │
    Block_1  ← gated by (1 - halt_prob_0)
        │
    Router_1 → halt_prob_1
        │
    Block_2  ← gated by (1 - halt_prob_1)
        │
      ...
        │
    LayerNorm → Linear head → logits

The model has N blocks and N-1 routers.
Block 0 always executes. The first routing decision happens after block 0,
before block 1.

Key metrics tracked during training:
    mean_active_frac: mean fraction of tokens that are active (not halted)
                      across all routing points.
                      Baseline value = 1.0 (all tokens always active).
                      TSA target < 1.0 (fewer active = more efficient).

Depth regularisation loss:
    depth_reg_loss = mean(active_frac_per_layer)
    This is added to the task loss with weight depth_reg_weight.
    It incentivises the model to use fewer blocks per token.
    The task loss provides the counterforce: halt too early → bad accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import AdamW

from tsa.core.base_model import BaseModel
from tsa.modules.topological_attention.graph_router import TokenRouter
from tsa.modules.topological_attention.sparse_attention import SparseTransformerBlock


@dataclass
class TSAConfig:
    vocab_size: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6         # same as baseline for fair comparison
    d_ff: int = 512
    max_seq_len: int = 64
    dropout: float = 0.1
    depth_reg_weight: float = 0.01   # λ: tradeoff between efficiency and accuracy


class TSATransformer(BaseModel):
    """
    Topological Sparse Attention transformer.

    Hypothesis: variable compute per token (content-difficulty routing) achieves
    the same accuracy as the fixed-depth baseline while using less average compute.

    To test the hypothesis, compare:
        baseline.val_accuracy  vs  tsa.val_accuracy      (should be ~equal)
        baseline.mean_active   vs  tsa.mean_active        (TSA should be < 1.0)

    If both hold: hypothesis supported. If accuracy drops: λ is too high.
    If mean_active stays near 1.0: λ is too low, or the routing never fires.
    """

    def __init__(self, config: TSAConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        H = config.n_heads
        L = config.n_layers
        d_ff = config.d_ff

        self.token_emb = nn.Embedding(config.vocab_size, d, padding_idx=0)
        self.pos_emb = nn.Embedding(config.max_seq_len, d)
        self.emb_drop = nn.Dropout(config.dropout)

        # N transformer blocks (same count as baseline for fair comparison)
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(d, H, d_ff, config.dropout)
            for _ in range(L)
        ])

        # N-1 routers: one after each block except the last
        # (No router after the last block — nothing left to route to)
        self.routers = nn.ModuleList([
            TokenRouter(d)
            for _ in range(L - 1)
        ])

        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying, same as baseline

        self._init_weights()

        # State updated every forward pass; trainer reads these via aux_loss() / get_extra_logs()
        # Register as buffer so it tracks model device (avoids CPU/GPU mismatch)
        self.register_buffer("_last_depth_reg_loss", torch.tensor(0.0))
        self._last_mean_active_frac: float = 1.0

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * len(self.blocks))
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
        assert T <= self.config.max_seq_len, (
            f"Input seq_len {T} exceeds max_seq_len {self.config.max_seq_len}. "
            f"Truncate input or increase TSAConfig.max_seq_len."
        )
        positions = torch.arange(T, device=x.device)
        h = self.emb_drop(self.token_emb(x) + self.pos_emb(positions))

        # Block 0: always fully active (halt_prob = 0 everywhere)
        # This is the "stem" — every token needs at least one layer to build
        # a contextualised representation before routing makes sense.
        h = self.blocks[0](
            h,
            halt_prob=torch.zeros(B, T, device=x.device),
        )

        # Blocks 1..N-1: each preceded by a router
        active_fracs: list[torch.Tensor] = []
        for router, block in zip(self.routers, self.blocks[1:]):
            halt_prob = router(h)                       # (B, T) in [0, 1]
            active_frac = 1.0 - halt_prob.mean()        # scalar tensor
            active_fracs.append(active_frac)
            h = block(h, halt_prob=halt_prob)

        # Depth regularisation: penalise high average active fraction.
        # active_frac = mean(1 - halt_prob) per layer; we minimise its mean.
        # Baseline equivalent would have active_frac = 1.0 everywhere.
        if active_fracs:
            depth_reg_loss = torch.stack(active_fracs).mean()
            self._last_mean_active_frac = depth_reg_loss.item()
        else:
            depth_reg_loss = torch.tensor(0.0, device=x.device)
            self._last_mean_active_frac = 1.0

        self._last_depth_reg_loss = depth_reg_loss

        h = self.ln_f(h)
        return self.head(h)

    def aux_loss(self) -> torch.Tensor:
        """
        Depth regularisation loss scaled by depth_reg_weight (λ).
        Added to the task loss by the Trainer automatically.
        """
        return self._last_depth_reg_loss * self.config.depth_reg_weight

    def get_extra_logs(self) -> dict[str, float]:
        """Log mean_active_frac alongside standard metrics."""
        return {"train/mean_active_frac": self._last_mean_active_frac}

    def configure_optimizers(self, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ("bias", "ln", "emb")):
                no_decay.append(param)
            else:
                decay.append(param)
        return AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
        )
