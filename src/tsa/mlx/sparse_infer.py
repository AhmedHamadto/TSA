"""
Hard-threshold sparse TSA inference for wall-clock benchmarking.

Provides two router variants and a sparse block for measuring whether TSA's
token-layer-op savings translate to actual throughput gains on Metal GPU.

Two router variants:
  FixedRouter   — every token gets the same halt_prob (uniform).
                  Used for soft-gating benchmarks where the gate is continuous.
  BimodalRouter — active_frac tokens get halt_prob=0.0, the rest get 1.0.
                  Simulates the bimodal gate distribution a trained TSA learns,
                  and is the correct input for hard-threshold sparse execution.

GatherSparseBlock — hard-threshold sparse FFN (gather/scatter on Metal GPU).
  Attention: always dense (all-to-all semantics preserved from training).
  FFN: sparse — tokens where halt_prob >= threshold skip the FFN entirely.
  Includes a CPU-GPU sync (mx.eval + np.where) to materialise active indices.
  The sync cost is included in all timings — it is a real execution cost.

Architecture decision: attention stays dense.
  FFN is per-token (independent) → gather/scatter is semantically correct.
  Attention is all-to-all → skipping individual tokens changes the attention
  pattern relative to training, producing a different model. Dense attention
  preserves exact inference equivalence to the trained TSA.

For inference benchmarking only. Not for training (hard threshold has no
gradient, and the CPU-GPU sync breaks the MLX gradient tape).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from tsa.mlx.model import FeedForward, MultiHeadAttention
from tsa.mlx.tsa import TSAConfig, TSATransformer


# ── Synthetic routers for benchmarking ───────────────────────────────────────

class FixedRouter(nn.Module):
    """
    Constant-output router: every token gets the same halt_prob.

    Use for soft-gating benchmarks. The continuous gate (1-halt_prob) is
    applied to all tokens regardless — no sparse savings, only router overhead.

    halt_prob_val = 1.0 - desired_active_frac
    e.g. active_frac=0.833 → halt_prob_val=0.167
    """
    def __init__(self, halt_prob_val: float) -> None:
        super().__init__()
        self.halt_prob_val = halt_prob_val

    def __call__(self, h: mx.array) -> mx.array:
        shape = h.shape  # (B, T, d_model)
        return mx.full((shape[0], shape[1]), self.halt_prob_val)


class BimodalRouter(nn.Module):
    """
    Bimodal router: active_frac tokens get halt_prob=0.0, the rest get 1.0.

    Simulates the bimodal gate distribution a trained TSA converges to
    (sigmoid + depth regularisation pushes gates toward 0 or 1).

    Use for sparse-execution benchmarks: tokens with halt_prob >= threshold
    (default 0.5) are gathered and skipped in the FFN. At active_frac=0.833,
    the last 16.7% of positions per sequence skip the FFN.

    Token assignment is deterministic (first n_active positions are active)
    for reproducible timing.
    """
    def __init__(self, active_frac: float) -> None:
        super().__init__()
        self.active_frac = active_frac

    def __call__(self, h: mx.array) -> mx.array:
        shape = h.shape  # (B, T, d_model)
        T = shape[1]
        n_active = int(T * self.active_frac)
        row = mx.concatenate([mx.zeros((n_active,)), mx.ones((T - n_active,))])
        return mx.broadcast_to(row[None, :], (shape[0], T))


# ── Sparse block for inference ────────────────────────────────────────────────

class GatherSparseBlock(nn.Module):
    """
    Hard-threshold sparse transformer block for inference.

    Attention: hard binary gate applied after dense all-to-all attention.
    FFN: gather active tokens → run FFN → scatter results back.

    Execution cost:
      1. Full attention on all tokens (dense)
      2. Hard-gate attn residual (binary mask multiply)
      3. CPU-GPU sync: mx.eval(active_mask) + np.where  [real overhead]
      4. Gather active tokens
      5. FFN on gathered tokens  [sparse savings here]
      6. Scatter add results back
      7. mx.eval(h_flat) to materialise scatter result before next block

    The CPU-GPU sync (step 3) is the dominant overhead at small batch sizes
    and typical active fractions. This is measured honestly in the benchmark.
    """

    def __init__(
        self,
        ln1: nn.LayerNorm,
        attn: MultiHeadAttention,
        ln2: nn.LayerNorm,
        ffn: FeedForward,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.ln1 = ln1
        self.attn = attn
        self.ln2 = ln2
        self.ffn = ffn
        self.threshold = threshold

    def __call__(self, h: mx.array, halt_prob: mx.array) -> mx.array:
        B, T, d = h.shape

        # Attention: dense, hard binary gate on residual
        active_hard = (halt_prob < self.threshold).astype(mx.float32)[..., None]  # (B,T,1)
        h = h + active_hard * self.attn(self.ln1(h))

        # FFN: sparse gather/scatter
        active_mask_flat = (halt_prob < self.threshold).reshape(B * T)  # (B*T,) bool

        # CPU-GPU sync: materialise boolean mask to get active indices
        mx.eval(active_mask_flat)
        active_np = np.array(active_mask_flat)
        active_indices = np.where(active_np)[0]

        if len(active_indices) == 0:
            return h  # All tokens halted — skip FFN entirely

        active_idx = mx.array(active_indices, dtype=mx.int32)
        h_flat = h.reshape(B * T, d)

        # Gather → LN + FFN → scatter add
        h_active = h_flat[active_idx]                # (n_active, d)
        ffn_out = self.ffn(self.ln2(h_active))       # (n_active, d) — sparse!
        h_flat = h_flat.at[active_idx].add(ffn_out)  # scatter
        mx.eval(h_flat)

        return h_flat.reshape(B, T, d)


# ── Factory functions ─────────────────────────────────────────────────────────

def make_soft_tsa(config: TSAConfig, active_frac: float) -> TSATransformer:
    """
    TSATransformer with FixedRouter at the given active_frac.

    Benchmarks soft-gating overhead: router adds computation, continuous gate
    scales all token residuals, but all tokens are still fully processed.
    """
    model = TSATransformer(config)
    halt_prob_val = 1.0 - active_frac
    for i in range(len(model.routers)):
        model.routers[i] = FixedRouter(halt_prob_val)
    model.eval()
    return model


def make_sparse_tsa(
    config: TSAConfig,
    active_frac: float,
    threshold: float = 0.5,
) -> TSATransformer:
    """
    TSATransformer with BimodalRouter and GatherSparseBlocks.

    Benchmarks sparse execution: active_frac tokens processed by FFN,
    (1-active_frac) tokens skip FFN. Includes CPU-GPU sync overhead.
    Attention remains dense throughout.
    """
    model = TSATransformer(config)
    for i in range(len(model.routers)):
        model.routers[i] = BimodalRouter(active_frac)
    for i in range(1, len(model.blocks)):
        orig = model.blocks[i]
        model.blocks[i] = GatherSparseBlock(
            ln1=orig.ln1,
            attn=orig.attn,
            ln2=orig.ln2,
            ffn=orig.ffn,
            threshold=threshold,
        )
    model.eval()
    return model
