"""
Tests for Phase 6: Shakespeare dataset + language SGC components.

Tests verify:
1. CharTokenizer: roundtrip, vocab size, coverage
2. ShakespeareDataset: shapes, chunk count, non-overlapping windows
3. LossWeightedSampler: shapes, sampling distribution, EMA updates
4. compute_per_sample_loss: shape, values in range
5. Integration: dataset + sampler work together
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tsa.benchmarks.shakespeare import (
    CharTokenizer,
    ShakespeareDataset,
    ShakespeareConfig,
)
from tsa.modules.curriculum_gen.language_sgc import (
    LossWeightedSampler,
    compute_per_sample_loss,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them.\n" * 20
)


@pytest.fixture
def tokenizer():
    return CharTokenizer(SAMPLE_TEXT)


@pytest.fixture
def token_tensor(tokenizer):
    ids = tokenizer.encode(SAMPLE_TEXT)
    return torch.tensor(ids, dtype=torch.long)


@pytest.fixture
def small_dataset(token_tensor):
    return ShakespeareDataset(token_tensor, context_len=16)


# ---------------------------------------------------------------------------
# CharTokenizer
# ---------------------------------------------------------------------------

class TestCharTokenizer:
    def test_vocab_size_matches_unique_chars(self, tokenizer):
        expected = len(set(SAMPLE_TEXT))
        assert tokenizer.vocab_size == expected

    def test_encode_returns_integers(self, tokenizer):
        ids = tokenizer.encode("hello")
        assert all(isinstance(i, int) for i in ids)

    def test_encode_length_matches_text(self, tokenizer):
        text = "hello world"
        ids = tokenizer.encode(text)
        assert len(ids) == len(text)

    def test_roundtrip(self, tokenizer):
        text = SAMPLE_TEXT[:100]
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_decode_from_tensor(self, tokenizer):
        ids = torch.tensor(tokenizer.encode("test"))
        result = tokenizer.decode(ids)
        assert result == "test"

    def test_all_chars_in_vocab(self, tokenizer):
        for c in set(SAMPLE_TEXT):
            assert c in tokenizer.stoi, f"Char {repr(c)} not in vocab"

    def test_indices_are_contiguous(self, tokenizer):
        indices = sorted(tokenizer.itos.keys())
        assert indices == list(range(tokenizer.vocab_size))


# ---------------------------------------------------------------------------
# ShakespeareDataset
# ---------------------------------------------------------------------------

class TestShakespeareDataset:
    def test_output_shapes(self, small_dataset):
        x, y = small_dataset[0]
        assert x.shape == (16,)
        assert y.shape == (16,)

    def test_x_y_shifted_by_one(self, token_tensor):
        ds = ShakespeareDataset(token_tensor, context_len=8)
        x, y = ds[0]
        # x is tokens 0..7, y is tokens 1..8
        assert torch.all(x[1:] == y[:-1])

    def test_len_is_number_of_complete_chunks(self, token_tensor):
        context_len = 16
        ds = ShakespeareDataset(token_tensor, context_len)
        expected = len(token_tensor) // (context_len + 1)
        assert len(ds) == expected

    def test_chunks_are_non_overlapping(self, token_tensor):
        ds = ShakespeareDataset(token_tensor, context_len=8)
        x0, _ = ds[0]
        x1, _ = ds[1]
        # First token of chunk 1 should be 9 positions after start of chunk 0
        assert not torch.equal(x0, x1)

    def test_getitem_index_zero(self, small_dataset):
        x, y = small_dataset[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_getitem_last_chunk(self, small_dataset):
        last_idx = len(small_dataset) - 1
        x, y = small_dataset[last_idx]
        assert x.shape == (small_dataset.context_len,)

    def test_get_chunk_raw(self, small_dataset):
        chunk = small_dataset.get_chunk_raw(0)
        assert chunk.shape == (small_dataset.context_len + 1,)
        x, y = small_dataset[0]
        assert torch.equal(chunk[:-1], x)
        assert torch.equal(chunk[1:], y)

    def test_vocab_values_in_range(self, small_dataset, tokenizer):
        for i in range(min(10, len(small_dataset))):
            x, y = small_dataset[i]
            assert x.min() >= 0
            assert x.max() < tokenizer.vocab_size
            assert y.min() >= 0
            assert y.max() < tokenizer.vocab_size


# ---------------------------------------------------------------------------
# LossWeightedSampler
# ---------------------------------------------------------------------------

class TestLossWeightedSampler:
    def test_sample_shape(self):
        sampler = LossWeightedSampler(n_chunks=100)
        ids = sampler.sample(32)
        assert ids.shape == (32,)

    def test_sample_indices_in_range(self):
        sampler = LossWeightedSampler(n_chunks=50)
        ids = sampler.sample(200)
        assert ids.min() >= 0
        assert ids.max() < 50

    def test_alpha_zero_is_uniform(self):
        """alpha=0 → all chunks equally likely."""
        sampler = LossWeightedSampler(n_chunks=4, alpha=0.0)
        # Give chunk 0 a very high loss
        sampler.update(torch.tensor([0]), torch.tensor([100.0]))
        counts = torch.zeros(4)
        for _ in range(1000):
            ids = sampler.sample(4)
            for i in ids.tolist():
                counts[i] += 1
        # With alpha=0 all weights are 1 regardless of loss — uniform
        fracs = counts / counts.sum()
        assert all(abs(f - 0.25) < 0.05 for f in fracs.tolist()), \
            f"Expected ~uniform, got {fracs.tolist()}"

    def test_high_alpha_favours_hard_chunks(self):
        """High loss chunks should be sampled more with alpha > 0."""
        sampler = LossWeightedSampler(n_chunks=2, alpha=1.0)
        # Chunk 0: easy (low loss), chunk 1: hard (high loss)
        sampler.update(torch.tensor([0, 0, 0]), torch.tensor([0.5, 0.5, 0.5]))
        sampler.update(torch.tensor([1, 1, 1]), torch.tensor([3.0, 3.0, 3.0]))
        counts = torch.zeros(2)
        for _ in range(500):
            ids = sampler.sample(2)
            counts[0] += (ids == 0).sum()
            counts[1] += (ids == 1).sum()
        frac_hard = counts[1] / counts.sum()
        assert frac_hard > 0.6, f"Hard chunk should be sampled more; got {frac_hard:.2f}"

    def test_ema_update_changes_loss(self):
        sampler = LossWeightedSampler(n_chunks=10, ema_decay=0.9)
        original = sampler.losses[3].item()
        sampler.update(torch.tensor([3]), torch.tensor([5.0]))
        updated = sampler.losses[3].item()
        # First update: should equal the observed loss directly (no EMA on first)
        assert abs(updated - 5.0) < 1e-5

    def test_ema_converges_to_new_value(self):
        sampler = LossWeightedSampler(n_chunks=5, ema_decay=0.9)
        # Repeat update many times with same loss — should converge
        for _ in range(100):
            sampler.update(torch.tensor([0]), torch.tensor([2.0]))
        assert abs(sampler.losses[0].item() - 2.0) < 0.01

    def test_coverage_increases_over_sampling(self):
        sampler = LossWeightedSampler(n_chunks=20)
        assert sampler.coverage == 0.0
        sampler.update(torch.arange(10), torch.ones(10))
        assert sampler.coverage == 0.5

    def test_effective_loss_range(self):
        sampler = LossWeightedSampler(n_chunks=5)
        sampler.update(torch.tensor([0, 1, 2]), torch.tensor([1.0, 2.0, 3.0]))
        lo, hi = sampler.effective_loss_range
        assert lo < hi
        assert abs(lo - 1.0) < 1e-5
        assert abs(hi - 3.0) < 1e-5

    def test_update_preserves_other_chunks(self):
        sampler = LossWeightedSampler(n_chunks=10)
        initial = sampler.losses.clone()
        sampler.update(torch.tensor([0]), torch.tensor([99.0]))
        # Other chunks should be unchanged
        assert torch.all(sampler.losses[1:] == initial[1:])


# ---------------------------------------------------------------------------
# compute_per_sample_loss
# ---------------------------------------------------------------------------

class TestComputePerSampleLoss:
    def test_output_shape(self):
        B, T, V = 4, 16, 65
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        loss = compute_per_sample_loss(logits, targets)
        assert loss.shape == (B,)

    def test_values_non_negative(self):
        B, T, V = 2, 8, 32
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        loss = compute_per_sample_loss(logits, targets)
        assert (loss >= 0).all()

    def test_perfect_prediction_near_zero(self):
        """If model always predicts the correct token, loss should be ~0."""
        B, T, V = 2, 4, 10
        targets = torch.zeros(B, T, dtype=torch.long)
        logits = torch.full((B, T, V), -100.0)
        logits[:, :, 0] = 100.0  # huge logit for token 0
        loss = compute_per_sample_loss(logits, targets)
        assert (loss < 0.01).all()

    def test_per_sample_not_aggregate(self):
        """Each sample should have independent loss values."""
        B, T, V = 3, 8, 20
        # Give different target patterns to each sample
        logits = torch.zeros(B, T, V)
        logits[0, :, 0] = 10.0   # easy: high logit matches target
        logits[1, :, 1] = 10.0   # wrong: high logit but targets are 0
        logits[2, :, 0] = -10.0  # hard: logit opposes target
        targets = torch.zeros(B, T, dtype=torch.long)  # all predict token 0
        loss = compute_per_sample_loss(logits, targets)
        # Sample 0 easy, sample 2 hardest
        assert loss[0] < loss[1]
        assert loss[1] < loss[2]
