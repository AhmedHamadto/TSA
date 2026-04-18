"""
Token router for Topological Sparse Attention.

The router answers one question per token per layer:
"Does this token need more computation, or is its representation good enough?"

Output: halt_prob ∈ [0, 1] per token.
    halt_prob ≈ 1 → token is confident/easy — halt, stop updating
    halt_prob ≈ 0 → token is uncertain/hard — continue, apply next block

The router is intentionally cheap (tiny MLP). If routing costs as much as
a transformer block, we've traded one expensive operation for two.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TokenRouter(nn.Module):
    """
    Lightweight per-token halting router.

    Architecture: Linear → ReLU → Linear → Sigmoid
    Hidden dim is d_model // 4 so routing costs ~4× less than a full block.

    The router is conditioned on the token's current hidden state, which
    encodes all context the token has accumulated so far. A token that has
    already absorbed enough context to predict its contribution will have
    a high-entropy (easy) hidden state, which the router learns to detect.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden_dim = max(d_model // 4, 16)  # floor at 16 to avoid degenerate tiny models
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialise the final bias slightly negative so the router starts
        # with a mild preference for continuing (halt_prob < 0.5).
        # This prevents the model from halting everything on step 0 before
        # it has learned anything useful.
        nn.init.constant_(self.mlp[-1].bias, -1.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, seq_len, d_model) — current hidden states
        Returns:
            halt_prob: (batch, seq_len) in [0, 1]
        """
        return torch.sigmoid(self.mlp(h).squeeze(-1))
