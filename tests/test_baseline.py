"""
Tests for the baseline transformer and toy task datasets.
Run with: pytest tests/test_baseline.py -v
"""
import math

import pytest
import torch

from tsa.benchmarks.baseline_transformer import BaselineTransformer
from tsa.benchmarks.toy_tasks import (
    BOS,
    EOS,
    NUM_SPECIAL,
    PAD,
    SEP,
    ToyTaskDataset,
    make_dataloaders,
)
from tsa.core.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model() -> BaselineTransformer:
    """Tiny model for fast tests — not meant to converge."""
    return BaselineTransformer(
        vocab_size=32,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Baseline Transformer
# ---------------------------------------------------------------------------

class TestBaselineTransformer:
    def test_forward_output_shape(self, small_model):
        x = torch.randint(0, 32, (2, 20))
        logits = small_model(x)
        assert logits.shape == (2, 20, 32), f"Expected (2, 20, 32), got {logits.shape}"

    def test_param_count_positive(self, small_model):
        assert small_model.get_num_params() > 0

    def test_weight_tying(self, small_model):
        # Input embedding and output head should share the exact same tensor
        assert small_model.head.weight is small_model.token_emb.weight

    def test_configure_optimizers_returns_adamw(self, small_model):
        from torch.optim import AdamW
        opt = small_model.configure_optimizers(lr=1e-3, weight_decay=0.1)
        assert isinstance(opt, AdamW)

    def test_forward_deterministic_at_eval(self, small_model):
        small_model.eval()
        x = torch.randint(0, 32, (1, 10))
        with torch.no_grad():
            logits1 = small_model(x)
            logits2 = small_model(x)
        assert torch.allclose(logits1, logits2)

    def test_causal_mask_independence(self, small_model):
        """Logits at position i must not change when tokens at positions > i change."""
        small_model.eval()
        x1 = torch.randint(NUM_SPECIAL, 32, (1, 8))
        x2 = x1.clone()
        x2[0, 4:] = torch.randint(NUM_SPECIAL, 32, (4,))  # change suffix
        with torch.no_grad():
            l1 = small_model(x1)
            l2 = small_model(x2)
        # First 4 positions should be identical
        assert torch.allclose(l1[0, :4], l2[0, :4], atol=1e-5), \
            "Causal mask broken: prefix logits changed when suffix changed"


# ---------------------------------------------------------------------------
# Toy Tasks
# ---------------------------------------------------------------------------

class TestToyTaskDataset:
    def test_copy_target_equals_source(self):
        ds = ToyTaskDataset("copy", seq_len=10, n_samples=50, vocab_size=32)
        # full_seq = [BOS, src..., SEP, tgt..., EOS]
        # x = full_seq[:-1]; src = x[1:11], tgt = x[12:22]
        x, y = ds[0]
        src = x[1:11]
        tgt = x[12:22]
        assert (src == tgt).all(), "Copy task: target should equal source"

    def test_reverse_target_is_flipped(self):
        ds = ToyTaskDataset("reverse", seq_len=10, n_samples=50, vocab_size=32)
        x, y = ds[0]
        src = x[1:11]
        tgt = x[12:22]
        assert (src.flip(0) == tgt).all(), "Reverse task: target should be reversed source"

    def test_sort_target_is_sorted(self):
        ds = ToyTaskDataset("sort", seq_len=10, n_samples=50, vocab_size=32)
        x, y = ds[0]
        tgt = x[12:22]
        assert (tgt == tgt.sort().values).all(), "Sort task: target should be sorted"

    def test_even_odd_target_layout(self):
        ds = ToyTaskDataset("even_odd", seq_len=10, n_samples=50, vocab_size=32)
        x, y = ds[0]
        src = x[1:11]
        tgt = x[12:22]
        expected = torch.cat([src[0::2], src[1::2]])
        assert (tgt == expected).all(), "even_odd task: target layout wrong"

    def test_loss_mask_covers_source_only(self):
        seq_len = 10
        ds = ToyTaskDataset("copy", seq_len=seq_len, n_samples=10, vocab_size=32)
        _, y = ds[0]
        # Should have exactly seq_len+1 unmasked positions (tgt tokens + EOS)
        n_unmasked = (y != -100).sum().item()
        assert n_unmasked == seq_len + 1, \
            f"Expected {seq_len + 1} unmasked positions, got {n_unmasked}"

    def test_sample_length(self):
        seq_len = 15
        ds = ToyTaskDataset("reverse", seq_len=seq_len, n_samples=10, vocab_size=32)
        x, y = ds[0]
        expected_len = 2 * seq_len + 2  # full_seq has 2*seq_len+3 tokens; x,y are len-1
        assert x.shape == (expected_len,)
        assert y.shape == (expected_len,)

    def test_special_tokens_not_in_data(self):
        ds = ToyTaskDataset("copy", seq_len=10, n_samples=100, vocab_size=32)
        # Source and target tokens should all be >= NUM_SPECIAL
        assert (ds.sources >= NUM_SPECIAL).all()
        assert (ds.targets >= NUM_SPECIAL).all()

    def test_even_odd_raises_on_odd_seq_len(self):
        with pytest.raises(ValueError, match="even seq_len"):
            ToyTaskDataset("even_odd", seq_len=11, n_samples=10, vocab_size=32)

    def test_vocab_size_validation(self):
        with pytest.raises(ValueError):
            ToyTaskDataset("copy", seq_len=10, n_samples=10, vocab_size=4)


class TestMakeDataloaders:
    def test_batch_shapes(self):
        train_loader, val_loader = make_dataloaders(
            "copy", seq_len=10, vocab_size=32,
            n_train=200, n_val=50, batch_size=16,
        )
        x, y = next(iter(train_loader))
        expected_seq = 2 * 10 + 2  # 22
        assert x.shape == (16, expected_seq)
        assert y.shape == (16, expected_seq)

    def test_train_val_different_seeds(self):
        train_loader, val_loader = make_dataloaders(
            "copy", seq_len=10, vocab_size=32,
            n_train=64, n_val=32, batch_size=32,
        )
        x_train, _ = next(iter(train_loader))
        x_val, _ = next(iter(val_loader))
        # Different seeds → different data (highly unlikely to be identical)
        assert not torch.equal(x_train[:32], x_val[:32])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions_accuracy_one(self):
        vocab_size, seq_len = 32, 10
        targets = torch.full((2, seq_len), 5, dtype=torch.long)
        logits = torch.zeros(2, seq_len, vocab_size)
        logits[:, :, 5] = 100.0  # very confident prediction of token 5
        m = compute_metrics(logits, targets)
        assert m.accuracy == pytest.approx(1.0)
        assert m.loss < 0.01

    def test_ignore_index_excluded(self):
        vocab_size, seq_len = 32, 10
        targets = torch.full((1, seq_len), 5, dtype=torch.long)
        targets[0, :5] = -100  # mask first half
        logits = torch.zeros(1, seq_len, vocab_size)
        logits[:, :, 5] = 100.0
        m = compute_metrics(logits, targets)
        # Only 5 tokens counted
        assert m.tokens_seen == 5
        assert m.accuracy == pytest.approx(1.0)

    def test_perplexity_is_exp_loss(self):
        vocab_size, seq_len = 32, 10
        targets = torch.randint(0, vocab_size, (2, seq_len))
        logits = torch.randn(2, seq_len, vocab_size)
        m = compute_metrics(logits, targets)
        assert m.perplexity == pytest.approx(math.exp(m.loss), rel=1e-4)
