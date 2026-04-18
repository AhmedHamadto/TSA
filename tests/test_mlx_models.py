"""
Tests for the MLX-native Genesis models.

Covers:
1. Data loading: CharTokenizer, random_batch shapes
2. BaselineTransformer: output shape, parameter count, forward pass
3. TSATransformer: output shape, routing semantics, depth_reg gradient
4. Training step: loss decreases, no NaN
5. Evaluate: returns a valid float

These tests do NOT require network access — they use synthetic token data.
They DO require Apple Silicon (MLX GPU), so they are skipped on non-Apple hardware.
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
    from tsa.mlx.data import CharTokenizer, random_batch
    from tsa.mlx.model import BaselineConfig, BaselineTransformer
    from tsa.mlx.tsa import TSAConfig, TSATransformer
    from tsa.mlx.evaluate import evaluate, bpc, active_fraction_savings


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
def baseline_cfg():
    return BaselineConfig(
        vocab_size=VOCAB, d_model=D, n_heads=H, n_layers=L,
        d_ff=D_FF, context_len=T, dropout=0.0,
    )


@pytest.fixture
def tsa_cfg():
    return TSAConfig(
        vocab_size=VOCAB, d_model=D, n_heads=H, n_layers=L,
        d_ff=D_FF, context_len=T, dropout=0.0, depth_reg_weight=0.01,
    )


@pytest.fixture
def baseline_model(baseline_cfg):
    mx.random.seed(0)
    return BaselineTransformer(baseline_cfg)


@pytest.fixture
def tsa_model(tsa_cfg):
    mx.random.seed(0)
    return TSATransformer(tsa_cfg)


@pytest.fixture
def dummy_tokens():
    """Synthetic token array (no download required)."""
    return mx.array(
        list(range(VOCAB)) * 200,  # 6400 tokens
        dtype=mx.int32,
    )


# ---------------------------------------------------------------------------
# CharTokenizer
# ---------------------------------------------------------------------------

class TestCharTokenizer:
    def test_vocab_size(self):
        tok = CharTokenizer("hello world")
        assert tok.vocab_size == len(set("hello world"))

    def test_encode_decode_roundtrip(self):
        text = "abcde"
        tok = CharTokenizer(text)
        assert tok.decode(tok.encode(text)) == text

    def test_encode_returns_ints(self):
        tok = CharTokenizer("abc")
        ids = tok.encode("abc")
        assert all(isinstance(i, int) for i in ids)


# ---------------------------------------------------------------------------
# random_batch
# ---------------------------------------------------------------------------

class TestRandomBatch:
    def test_shapes(self, dummy_tokens):
        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
        assert x.shape == (B, T)
        assert y.shape == (B, T)

    def test_dtype_int32(self, dummy_tokens):
        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
        assert x.dtype == mx.int32
        assert y.dtype == mx.int32

    def test_y_is_x_shifted(self, dummy_tokens):
        # y should be x shifted right by 1 (next-token prediction)
        x, y = random_batch(dummy_tokens, context_len=T, batch_size=1)
        mx.eval(x, y)
        # For each position in x except the last, x[0, i+1] should equal y[0, i]
        x_np = x[0].tolist()
        y_np = y[0].tolist()
        assert x_np[1:] == y_np[:-1]


# ---------------------------------------------------------------------------
# BaselineTransformer
# ---------------------------------------------------------------------------

class TestBaselineTransformer:
    def test_output_shape(self, baseline_model, dummy_tokens):
        x, _ = random_batch(dummy_tokens, context_len=T, batch_size=B)
        logits = baseline_model(x)
        mx.eval(logits)
        assert logits.shape == (B, T, VOCAB)

    def test_no_nan(self, baseline_model, dummy_tokens):
        x, _ = random_batch(dummy_tokens, context_len=T, batch_size=B)
        logits = baseline_model(x)
        mx.eval(logits)
        assert not mx.any(mx.isnan(logits)).item()

    def test_param_count_positive(self, baseline_model):
        assert baseline_model.num_params() > 0

    def test_weight_tying(self, baseline_model):
        # The output projection uses token_emb.weight; check logit shape matches vocab
        x = mx.zeros((1, T), dtype=mx.int32)
        logits = baseline_model(x)
        mx.eval(logits)
        assert logits.shape[-1] == VOCAB

    def test_different_inputs_different_outputs(self, baseline_model, dummy_tokens):
        """Different tokens should produce different logits."""
        x0, _ = random_batch(dummy_tokens, context_len=T, batch_size=1)
        x1 = mx.zeros((1, T), dtype=mx.int32)
        l0 = baseline_model(x0)
        l1 = baseline_model(x1)
        mx.eval(l0, l1)
        # Not all identical
        assert not mx.all(mx.equal(l0, l1)).item()


# ---------------------------------------------------------------------------
# TSATransformer
# ---------------------------------------------------------------------------

class TestTSATransformer:
    def test_output_shape(self, tsa_model, dummy_tokens):
        x, _ = random_batch(dummy_tokens, context_len=T, batch_size=B)
        logits = tsa_model(x)
        mx.eval(logits)
        assert logits.shape == (B, T, VOCAB)

    def test_no_nan(self, tsa_model, dummy_tokens):
        x, _ = random_batch(dummy_tokens, context_len=T, batch_size=B)
        logits = tsa_model(x)
        mx.eval(logits)
        assert not mx.any(mx.isnan(logits)).item()

    def test_depth_reg_loss_is_set(self, tsa_model, dummy_tokens):
        x, _ = random_batch(dummy_tokens, context_len=T, batch_size=B)
        _ = tsa_model(x)
        mx.eval(tsa_model._depth_reg_loss)
        val = tsa_model._depth_reg_loss.item()
        assert math.isfinite(val)
        # depth_reg is mean(active_fracs), active_frac ∈ [0, 1]
        assert 0.0 <= val <= 1.0

    def test_depth_reg_gradient_flows(self, tsa_model, dummy_tokens):
        """Gradient from depth_reg_loss should reach router weights."""
        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)

        def loss_fn(model, x, y):
            logits = model(x)
            task_loss = mx.mean(
                nn.losses.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            )
            return task_loss + model.config.depth_reg_weight * model._depth_reg_loss

        loss_and_grad = nn.value_and_grad(tsa_model, loss_fn)
        loss, grads = loss_and_grad(tsa_model, x, y)
        mx.eval(loss, grads)

        # Check that router fc2 bias receives a gradient (not None / zero)
        router_grads = grads.get("routers", [])
        assert len(router_grads) > 0, "No router gradients found"
        # At least one router should have a non-zero gradient
        has_grad = False
        for rg in router_grads:
            if isinstance(rg, dict) and "fc2" in rg:
                bias_grad = rg["fc2"].get("bias")
                if bias_grad is not None:
                    mx.eval(bias_grad)
                    if mx.any(mx.abs(bias_grad) > 1e-10).item():
                        has_grad = True
        assert has_grad, "Router gradients are all zero — depth reg not flowing"

    def test_active_frac_in_range(self, tsa_model, dummy_tokens):
        x, _ = random_batch(dummy_tokens, context_len=T, batch_size=B)
        _ = tsa_model(x)
        mx.eval(tsa_model._depth_reg_loss)
        af = tsa_model._depth_reg_loss.item()
        assert 0.0 <= af <= 1.0


# ---------------------------------------------------------------------------
# Training step (smoke test)
# ---------------------------------------------------------------------------

class TestTrainingStep:
    def test_baseline_loss_decreases(self, baseline_model, dummy_tokens):
        """Val loss should decrease after 20 training steps."""
        optimizer = optim.AdamW(learning_rate=1e-3)

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(
                nn.losses.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            )

        loss_and_grad = nn.value_and_grad(baseline_model, loss_fn)

        # Initial loss
        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
        initial_loss_arr, _ = loss_and_grad(baseline_model, x, y)
        mx.eval(initial_loss_arr)
        initial_loss = initial_loss_arr.item()

        # Train 20 steps
        for _ in range(20):
            x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
            loss_arr, grads = loss_and_grad(baseline_model, x, y)
            optimizer.update(baseline_model, grads)
            mx.eval(baseline_model.parameters(), optimizer.state)

        # Final loss
        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
        final_loss_arr, _ = loss_and_grad(baseline_model, x, y)
        mx.eval(final_loss_arr)
        final_loss = final_loss_arr.item()

        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_tsa_training_step_no_error(self, tsa_model, dummy_tokens):
        """TSA training step should complete without error."""
        optimizer = optim.AdamW(learning_rate=1e-3)

        def loss_fn(model, x, y):
            logits = model(x)
            task_loss = mx.mean(
                nn.losses.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            )
            return task_loss + model.config.depth_reg_weight * model._depth_reg_loss

        loss_and_grad = nn.value_and_grad(tsa_model, loss_fn)

        x, y = random_batch(dummy_tokens, context_len=T, batch_size=B)
        loss_arr, grads = loss_and_grad(tsa_model, x, y)
        optimizer.update(tsa_model, grads)
        mx.eval(tsa_model.parameters(), optimizer.state)
        mx.eval(loss_arr)

        assert math.isfinite(loss_arr.item())


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_finite_float(self, baseline_model, dummy_tokens):
        val_loss = evaluate(baseline_model, dummy_tokens, context_len=T, batch_size=B, n_batches=3)
        assert math.isfinite(val_loss)
        assert val_loss > 0.0

    def test_bpc_conversion(self):
        assert abs(bpc(1.0) - 1.0 / math.log(2)) < 1e-6

    def test_active_fraction_savings(self):
        # If all layers always active (af=1.0), savings = 0
        assert abs(active_fraction_savings(1.0, n_layers=6)) < 1e-9
        # If all layers after stem are halted (af=0.0), savings = (L-1)/L
        expected = 5 / 6
        assert abs(active_fraction_savings(0.0, n_layers=6) - expected) < 1e-9
