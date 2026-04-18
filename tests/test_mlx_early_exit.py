"""
Tests for the MLX EarlyExitTransformer.

Covers:
1. EarlyExitTransformer forward: output shape, no NaN, exit logits stored
2. early_exit_loss: finite, decreases with training
3. evaluate_early_exit: valid loss and active_frac range
4. Threshold effects: higher threshold → higher active_frac

Pattern follows test_mlx_models.py — synthetic token data, no download.
Skipped on non-MLX hardware.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")

if MLX_AVAILABLE:
    from tsa.mlx.data import random_batch
    from tsa.mlx.early_exit import (
        EarlyExitConfig,
        EarlyExitTransformer,
        early_exit_loss,
        evaluate_early_exit,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB = 32
D = 64
H = 2
L = 3
T = 16
B = 4
D_FF = 128


@pytest.fixture
def ee_cfg():
    return EarlyExitConfig(
        vocab_size=VOCAB, d_model=D, n_heads=H, n_layers=L,
        d_ff=D_FF, context_len=T, dropout=0.0,
    )


@pytest.fixture
def ee_model(ee_cfg):
    mx.random.seed(42)
    return EarlyExitTransformer(ee_cfg)


@pytest.fixture
def dummy_tokens():
    """Synthetic token array (no download required)."""
    return mx.array(
        list(range(VOCAB)) * 200,  # 6400 tokens
        dtype=mx.int32,
    )


# ---------------------------------------------------------------------------
# EarlyExitTransformer — forward pass
# ---------------------------------------------------------------------------

class TestEarlyExitTransformerForward:
    def test_output_shape(self, ee_model, dummy_tokens):
        x, _ = random_batch(dummy_tokens, context_len=T, batch_size=B)
        logits = ee_model(x)
        mx.eval(logits)
        assert logits.shape == (B, T, VOCAB)

    def test_no_nan(self, ee_model, dummy_tokens):
        x, _ = random_batch(dummy_tokens, context_len=T, batch_size=B)
        logits = ee_model(x)
        mx.eval(logits)
        assert not mx.any(mx.isnan(logits)).item()

    def test_exit_logits_count(self, ee_model, dummy_tokens):
        """After __call__, _exit_logits should have exactly N entries (one per block)."""
        x, _ = random_batch(dummy_tokens, context_len=T, batch_size=B)
        _ = ee_model(x)
        mx.eval(*ee_model._exit_logits)
        assert len(ee_model._exit_logits) == L

    def test_exit_logits_shapes(self, ee_model, dummy_tokens):
        """Each exit logit should be (B, T, vocab)."""
        x, _ = random_batch(dummy_tokens, context_len=T, batch_size=B)
        _ = ee_model(x)
        mx.eval(*ee_model._exit_logits)
        for l, el in enumerate(ee_model._exit_logits):
            assert el.shape == (B, T, VOCAB), f"Exit {l} shape mismatch: {el.shape}"

    def test_final_exit_matches_return(self, ee_model, dummy_tokens):
        """Return value must equal _exit_logits[-1]."""
        x, _ = random_batch(dummy_tokens, context_len=T, batch_size=B)
        logits = ee_model(x)
        mx.eval(logits, ee_model._exit_logits[-1])
        assert mx.allclose(logits, ee_model._exit_logits[-1]).item()

    def test_param_count_positive(self, ee_model):
        assert ee_model.num_params() > 0

    def test_exit_lns_count(self, ee_cfg):
        """One exit LayerNorm per block."""
        model = EarlyExitTransformer(ee_cfg)
        assert len(model.exit_lns) == L

    def test_blocks_count(self, ee_cfg):
        model = EarlyExitTransformer(ee_cfg)
        assert len(model.blocks) == L


# ---------------------------------------------------------------------------
# early_exit_loss
# ---------------------------------------------------------------------------

class TestEarlyExitLoss:
    def test_finite(self, ee_model, dummy_tokens):
        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
        loss = early_exit_loss(ee_model, x, y)
        mx.eval(loss)
        assert math.isfinite(loss.item())

    def test_positive(self, ee_model, dummy_tokens):
        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
        loss = early_exit_loss(ee_model, x, y)
        mx.eval(loss)
        assert loss.item() > 0.0

    def test_populates_exit_logits(self, ee_model, dummy_tokens):
        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
        _ = early_exit_loss(ee_model, x, y)
        assert len(ee_model._exit_logits) == L

    def test_gradient_flows(self, ee_model, dummy_tokens):
        """Gradient from early_exit_loss should reach all exit LN weights."""
        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)

        def loss_fn(model, x, y):
            return early_exit_loss(model, x, y)

        loss_and_grad = nn.value_and_grad(ee_model, loss_fn)
        loss, grads = loss_and_grad(ee_model, x, y)
        mx.eval(loss, grads)

        assert math.isfinite(loss.item())

        # exit_lns should have gradients
        exit_ln_grads = grads.get("exit_lns", [])
        assert len(exit_ln_grads) == L, (
            f"Expected {L} exit_ln gradient dicts, got {len(exit_ln_grads)}"
        )
        # At least one exit LN should have non-zero weight gradient
        has_grad = any(
            isinstance(g, dict) and mx.any(mx.abs(g.get("weight", mx.array([0.0]))) > 1e-12).item()
            for g in exit_ln_grads
        )
        assert has_grad, "Exit LN gradients all zero — loss not flowing through exit heads"

    def test_loss_decreases_with_training(self, ee_cfg, dummy_tokens):
        """Early exit loss should decrease after 20 training steps."""
        mx.random.seed(0)
        model = EarlyExitTransformer(ee_cfg)
        optimizer = optim.AdamW(learning_rate=1e-3)

        def loss_fn(m, x, y):
            return early_exit_loss(m, x, y)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
        init_loss, _ = loss_and_grad(model, x, y)
        mx.eval(init_loss)
        initial = init_loss.item()

        for _ in range(20):
            x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
            loss_arr, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
        final_loss, _ = loss_and_grad(model, x, y)
        mx.eval(final_loss)
        final = final_loss.item()

        assert final < initial, f"Loss did not decrease: {initial:.4f} → {final:.4f}"


# ---------------------------------------------------------------------------
# evaluate_early_exit
# ---------------------------------------------------------------------------

class TestEvaluateEarlyExit:
    def test_returns_tuple(self, ee_model, dummy_tokens):
        result = evaluate_early_exit(
            ee_model, dummy_tokens, context_len=T, batch_size=B,
            threshold=0.5, n_batches=2,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_finite_loss(self, ee_model, dummy_tokens):
        val_loss, _ = evaluate_early_exit(
            ee_model, dummy_tokens, context_len=T, batch_size=B,
            threshold=0.5, n_batches=2,
        )
        assert math.isfinite(val_loss)
        assert val_loss > 0.0

    def test_active_frac_in_range(self, ee_model, dummy_tokens):
        """active_frac ∈ (0, 1] always."""
        _, af = evaluate_early_exit(
            ee_model, dummy_tokens, context_len=T, batch_size=B,
            threshold=0.5, n_batches=2,
        )
        assert 0.0 < af <= 1.0, f"active_frac out of range: {af}"

    def test_threshold_zero_minimum_active(self, ee_model, dummy_tokens):
        """threshold=0.0 → all tokens exit at block 1 → active_frac near 1/L."""
        _, af = evaluate_early_exit(
            ee_model, dummy_tokens, context_len=T, batch_size=B,
            threshold=0.0, n_batches=3,
        )
        # With threshold=0, every token exits at the first post-stem block (l=1)
        # ops_used=1 for each token (one post-stem block ran), mean_ops=1
        # active_frac = (1 + 1) / L = 2/L
        expected = 2.0 / L
        assert abs(af - expected) < 0.01, f"Expected ~{expected:.3f} at threshold=0, got {af:.3f}"

    def test_threshold_one_maximum_active(self, ee_model, dummy_tokens):
        """threshold=1.0 → no early exits → active_frac = 1.0."""
        _, af = evaluate_early_exit(
            ee_model, dummy_tokens, context_len=T, batch_size=B,
            threshold=1.0, n_batches=3,
        )
        # With threshold=1.0, confidence never exceeds 1.0, so no early exits
        # until the forced last block → ops_used = L-1 for all tokens
        # active_frac = (1 + (L-1)) / L = L/L = 1.0
        assert abs(af - 1.0) < 0.01, f"Expected ~1.0 at threshold=1.0, got {af:.3f}"

    def test_higher_threshold_higher_active(self, ee_model, dummy_tokens):
        """Monotonicity: higher threshold → higher (or equal) active_frac."""
        _, af_low = evaluate_early_exit(
            ee_model, dummy_tokens, context_len=T, batch_size=B,
            threshold=0.3, n_batches=3,
        )
        _, af_high = evaluate_early_exit(
            ee_model, dummy_tokens, context_len=T, batch_size=B,
            threshold=0.9, n_batches=3,
        )
        assert af_low <= af_high + 0.05, (
            f"Expected higher threshold → more active; got {af_low:.3f} > {af_high:.3f}"
        )
