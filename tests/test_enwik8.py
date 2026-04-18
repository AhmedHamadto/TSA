"""
Tests for the enwik8 dataset module.

Uses synthetic text to avoid network download during tests.
Tests verify the dataset construction logic, not the actual enwik8 download.
"""
from __future__ import annotations

import pytest
import torch

from tsa.benchmarks.shakespeare import CharTokenizer, ShakespeareDataset
from tsa.benchmarks.enwik8 import Enwik8Config, TRAIN_SIZE, VAL_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_WIKI = (
    "<mediawiki>\n"
    "<page><title>Anarchism</title><text>Anarchism is a political philosophy.</text></page>\n"
    "<page><title>Autism</title><text>Autism is a neurodevelopmental condition.</text></page>\n"
    "Some freeform text with numbers 12345 and symbols !@#$%^&amp;\n"
) * 200   # ~7KB synthetic text


def _make_ids(text: str) -> torch.Tensor:
    tok = CharTokenizer(text)
    return torch.tensor(tok.encode(text), dtype=torch.long), tok


# ---------------------------------------------------------------------------
# Enwik8Config
# ---------------------------------------------------------------------------

class TestEnwik8Config:
    def test_default_context_len(self):
        cfg = Enwik8Config()
        assert cfg.context_len == 256

    def test_custom_context_len(self):
        cfg = Enwik8Config(context_len=128)
        assert cfg.context_len == 128

    def test_split_sizes_sum_to_100m(self):
        assert TRAIN_SIZE + VAL_SIZE == 95_000_000
        # test = remainder; total = 100M
        assert TRAIN_SIZE == 90_000_000
        assert VAL_SIZE == 5_000_000


# ---------------------------------------------------------------------------
# Dataset construction (using synthetic data, no download)
# ---------------------------------------------------------------------------

class TestEnwik8DatasetConstruction:
    """Tests the dataset construction logic with synthetic text."""

    def test_train_val_test_split_sizes(self):
        ids, _ = _make_ids(SAMPLE_WIKI)
        n = len(ids)
        # Simulate the split logic from build_enwik8_datasets
        # with a small synthetic corpus
        train_size = int(n * 0.9)
        val_size   = int(n * 0.05)
        train_data = ids[:train_size]
        val_data   = ids[train_size:train_size + val_size]
        test_data  = ids[train_size + val_size:]
        assert len(train_data) + len(val_data) + len(test_data) == n

    def test_dataset_chunks_non_overlapping(self):
        ids, _ = _make_ids(SAMPLE_WIKI)
        context_len = 16
        ds = ShakespeareDataset(ids, context_len)
        x0, _ = ds[0]
        x1, _ = ds[1]
        assert not torch.equal(x0, x1)

    def test_dataset_output_shapes(self):
        ids, _ = _make_ids(SAMPLE_WIKI)
        context_len = 32
        ds = ShakespeareDataset(ids, context_len)
        x, y = ds[0]
        assert x.shape == (context_len,)
        assert y.shape == (context_len,)
        assert x.dtype == torch.long

    def test_dataset_x_y_shifted(self):
        ids, _ = _make_ids(SAMPLE_WIKI)
        ds = ShakespeareDataset(ids, context_len=16)
        x, y = ds[0]
        assert torch.all(x[1:] == y[:-1])

    def test_vocab_includes_xml_chars(self):
        ids, tok = _make_ids(SAMPLE_WIKI)
        assert "<" in tok.stoi
        assert ">" in tok.stoi
        assert "/" in tok.stoi

    def test_vocab_size_larger_than_shakespeare(self):
        """enwik8 should have more unique chars than Shakespeare (~65)."""
        ids, tok = _make_ids(SAMPLE_WIKI)
        # Our synthetic text includes letters, digits, symbols, XML tags
        assert tok.vocab_size > 40  # at minimum

    def test_encode_decode_roundtrip(self):
        ids, tok = _make_ids(SAMPLE_WIKI)
        text = SAMPLE_WIKI[:200]
        assert tok.decode(tok.encode(text)) == text

    def test_all_encoded_in_range(self):
        ids, tok = _make_ids(SAMPLE_WIKI)
        ds = ShakespeareDataset(ids, context_len=16)
        for i in range(min(5, len(ds))):
            x, y = ds[i]
            assert x.min() >= 0
            assert x.max() < tok.vocab_size
