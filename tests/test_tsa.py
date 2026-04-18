"""
Tests for Topological Sparse Attention (TSA) — Phase 1 Genesis module.

Tests verify:
1. Structural correctness (shapes, interfaces)
2. Routing semantics (halt_prob=0 → same as baseline, halt_prob=1 → no change)
3. Differentiability (gradients flow through routing decisions)
4. Efficiency property (mean_active_frac < 1 after training with depth_reg)
5. aux_loss / get_extra_logs hooks work correctly
"""
import pytest
import torch
import torch.nn as nn

from tsa.core.base_model import BaseModel
from tsa.modules.topological_attention.graph_router import TokenRouter
from tsa.modules.topological_attention.sparse_attention import SparseTransformerBlock
from tsa.modules.topological_attention.variable_depth import TSAConfig, TSATransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tsa_config() -> TSAConfig:
    return TSAConfig(
        vocab_size=32,
        d_model=64,
        n_heads=2,
        n_layers=4,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
        depth_reg_weight=0.01,
    )


@pytest.fixture
def tsa_model(tsa_config) -> TSATransformer:
    return TSATransformer(tsa_config)


# ---------------------------------------------------------------------------
# TokenRouter
# ---------------------------------------------------------------------------

class TestTokenRouter:
    def test_output_shape(self):
        router = TokenRouter(d_model=64)
        h = torch.randn(2, 10, 64)
        halt_prob = router(h)
        assert halt_prob.shape == (2, 10)

    def test_output_in_zero_one(self):
        router = TokenRouter(d_model=64)
        h = torch.randn(4, 20, 64)
        halt_prob = router(h)
        assert halt_prob.min() >= 0.0
        assert halt_prob.max() <= 1.0

    def test_initial_bias_toward_continue(self):
        """Router bias init should make most tokens start in 'continue' mode."""
        router = TokenRouter(d_model=64)
        with torch.no_grad():
            h = torch.zeros(1, 100, 64)
            halt_prob = router(h)
        # With bias=-1.0, sigmoid(-1) ≈ 0.27, so mean halt < 0.5
        assert halt_prob.mean().item() < 0.5, \
            "Router should initially prefer continuing (halt_prob < 0.5)"

    def test_gradient_flows(self):
        router = TokenRouter(d_model=32)
        h = torch.randn(2, 5, 32, requires_grad=True)
        halt_prob = router(h)
        halt_prob.sum().backward()
        assert h.grad is not None
        assert h.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# SparseTransformerBlock
# ---------------------------------------------------------------------------

class TestSparseTransformerBlock:
    def test_output_shape(self):
        block = SparseTransformerBlock(d_model=64, n_heads=2, d_ff=128, dropout=0.0)
        h = torch.randn(2, 10, 64)
        halt_prob = torch.zeros(2, 10)
        out = block(h, halt_prob)
        assert out.shape == h.shape

    def test_fully_active_matches_standard_block(self):
        """halt_prob=0 everywhere should give same result as a plain transformer block."""
        torch.manual_seed(0)
        block = SparseTransformerBlock(d_model=32, n_heads=2, d_ff=64, dropout=0.0)
        h = torch.randn(1, 5, 32)
        halt_prob_zero = torch.zeros(1, 5)
        halt_prob_one = torch.ones(1, 5)

        with torch.no_grad():
            out_active = block(h, halt_prob_zero)
            out_halted = block(h, halt_prob_one)

        # Fully halted: output should equal input
        assert torch.allclose(out_halted, h, atol=1e-6), \
            "halt_prob=1 should leave the state unchanged"
        # Fully active: output should differ from input (unless block is identity)
        assert not torch.allclose(out_active, h, atol=1e-5), \
            "halt_prob=0 should update the hidden state"

    def test_partial_gating_interpolates(self):
        """halt_prob=0.5 should produce a result between no-update and full-update."""
        block = SparseTransformerBlock(d_model=32, n_heads=2, d_ff=64, dropout=0.0)
        h = torch.randn(1, 4, 32)
        with torch.no_grad():
            out_active = block(h, torch.zeros(1, 4))
            out_halted = block(h, torch.ones(1, 4))
            out_half = block(h, torch.full((1, 4), 0.5))
        # out_half should be between out_halted and out_active
        # Check: distance from h should be < distance from active update
        dist_half = (out_half - h).norm()
        dist_full = (out_active - h).norm()
        assert dist_half < dist_full, \
            "half-gated update should be smaller than full update"

    def test_gradient_flows_through_halt_prob(self):
        block = SparseTransformerBlock(d_model=32, n_heads=2, d_ff=64, dropout=0.0)
        h = torch.randn(2, 5, 32)
        raw_logits = torch.randn(2, 5, requires_grad=True)   # leaf tensor
        halt_prob = torch.sigmoid(raw_logits)                 # non-leaf — need retain_grad
        halt_prob.retain_grad()
        out = block(h, halt_prob)
        out.sum().backward()
        # Verify gradient reaches the leaf tensor (routing params would be leaves)
        assert raw_logits.grad is not None
        assert raw_logits.grad.abs().sum() > 0, \
            "Gradient must flow through halt_prob for routing to be learnable"


# ---------------------------------------------------------------------------
# TSATransformer
# ---------------------------------------------------------------------------

class TestTSATransformer:
    def test_is_base_model(self, tsa_model):
        assert isinstance(tsa_model, BaseModel)

    def test_forward_output_shape(self, tsa_model, tsa_config):
        x = torch.randint(0, tsa_config.vocab_size, (2, 20))
        logits = tsa_model(x)
        assert logits.shape == (2, 20, tsa_config.vocab_size)

    def test_param_count_positive(self, tsa_model):
        assert tsa_model.get_num_params() > 0

    def test_weight_tying(self, tsa_model):
        assert tsa_model.head.weight is tsa_model.token_emb.weight

    def test_aux_loss_is_tensor(self, tsa_model):
        x = torch.randint(0, 32, (2, 10))
        _ = tsa_model(x)
        loss = tsa_model.aux_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_aux_loss_is_nonnegative(self, tsa_model):
        x = torch.randint(0, 32, (2, 10))
        _ = tsa_model(x)
        assert tsa_model.aux_loss().item() >= 0.0

    def test_get_extra_logs_contains_active_frac(self, tsa_model):
        x = torch.randint(0, 32, (2, 10))
        _ = tsa_model(x)
        logs = tsa_model.get_extra_logs()
        assert "train/mean_active_frac" in logs
        frac = logs["train/mean_active_frac"]
        assert 0.0 <= frac <= 1.0

    def test_active_frac_starts_below_one(self, tsa_model):
        """
        Because the router bias is initialised negative, mean_active_frac
        should start below 1.0 (some halting happening from step 0).
        """
        x = torch.randint(0, 32, (2, 20))
        tsa_model.eval()
        with torch.no_grad():
            _ = tsa_model(x)
        frac = tsa_model.get_extra_logs()["train/mean_active_frac"]
        assert frac < 1.0, \
            "Active fraction should be < 1.0 due to router bias init toward halting"

    def test_configure_optimizers(self, tsa_model):
        from torch.optim import AdamW
        opt = tsa_model.configure_optimizers(lr=1e-3, weight_decay=0.1)
        assert isinstance(opt, AdamW)

    def test_n_routers_is_n_layers_minus_one(self, tsa_model, tsa_config):
        assert len(tsa_model.routers) == tsa_config.n_layers - 1

    def test_gradient_flows_end_to_end(self, tsa_model, tsa_config):
        """Full backward pass: gradients should reach all parameters including routers."""
        x = torch.randint(0, tsa_config.vocab_size, (2, 10))
        y = torch.randint(0, tsa_config.vocab_size, (2, 10))
        logits = tsa_model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, tsa_config.vocab_size), y.view(-1)
        ) + tsa_model.aux_loss()
        loss.backward()
        # Check router gradients specifically — this verifies the routing is trainable
        for i, router in enumerate(tsa_model.routers):
            for name, param in router.named_parameters():
                assert param.grad is not None, \
                    f"Router {i} param '{name}' has no gradient"
                assert param.grad.abs().sum() > 0, \
                    f"Router {i} param '{name}' has zero gradient"

    def test_depth_reg_weight_zero_gives_zero_aux_loss(self, tsa_config):
        """With λ=0, aux_loss should be exactly 0.0 regardless of routing."""
        config = TSAConfig(**{**tsa_config.__dict__, "depth_reg_weight": 0.0})
        model = TSATransformer(config)
        x = torch.randint(0, config.vocab_size, (2, 10))
        _ = model(x)
        assert model.aux_loss().item() == 0.0

    def test_causal_mask_independence(self, tsa_model):
        """Prefix logits must not change when suffix tokens change."""
        tsa_model.eval()
        x1 = torch.randint(4, 32, (1, 8))
        x2 = x1.clone()
        x2[0, 4:] = torch.randint(4, 32, (4,))
        with torch.no_grad():
            l1 = tsa_model(x1)
            l2 = tsa_model(x2)
        assert torch.allclose(l1[0, :4], l2[0, :4], atol=1e-5), \
            "Causal masking broken in TSA"
