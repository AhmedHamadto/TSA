"""
Tiny-Shakespeare character-level language modeling dataset.

Downloads ~1MB Shakespeare corpus from karpathy/char-rnn programmatically.
No manual file placement required — cached to data/tinyshakespeare.txt.

Provides:
  CharTokenizer   — character-level encode/decode, vocab built from corpus
  ShakespeareDataset — PyTorch Dataset returning (x, y) context windows
  build_shakespeare_datasets — train/val/test split, returns everything needed

Character-level LM choice:
  No BPE, no external tokenizer. Vocab ≈ 65 chars.
  Next-char prediction task: x = context[:-1], y = context[1:].
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DEFAULT_CACHE = Path(__file__).parent.parent.parent.parent / "data" / "tinyshakespeare.txt"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_shakespeare(cache_path: Path | str | None = None) -> str:
    """
    Download tiny-shakespeare, cache locally, and return text.
    If the cache file exists, use it without re-downloading.
    """
    path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        print(f"  Downloading Shakespeare corpus → {path} …")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        print(f"  Downloaded: {path.stat().st_size / 1024:.0f} KB")
    else:
        print(f"  Using cached corpus: {path} ({path.stat().st_size / 1024:.0f} KB)")

    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    """
    Character-level tokenizer built from corpus vocabulary.

    Vocab = sorted unique characters in the corpus.
    Indices are 0-based; no special tokens, no padding.
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self.itos: dict[int, str] = {i: c for c, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class ShakespeareConfig:
    context_len: int = 128
    train_frac: float = 0.8
    val_frac: float = 0.1
    # test_frac is implicitly 1 - train_frac - val_frac = 0.1
    cache_path: str | None = None   # None = DEFAULT_CACHE


class ShakespeareDataset(Dataset):
    """
    Fixed-stride character windows for next-char prediction.

    Splits the token tensor into non-overlapping chunks of length context_len.
    x = chunk[:-1]  (context)
    y = chunk[1:]   (targets, shifted by 1)

    Non-overlapping chunks make indexing by chunk_id unambiguous —
    required for the SGC loss-weighted sampler to track per-chunk difficulty.
    """

    def __init__(self, data: torch.Tensor, context_len: int) -> None:
        self.data = data
        self.context_len = context_len
        # Number of complete chunks (last partial chunk discarded)
        self.n_chunks = len(data) // (context_len + 1)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * (self.context_len + 1)
        chunk = self.data[start : start + self.context_len + 1]
        return chunk[:-1], chunk[1:]

    def get_chunk_raw(self, idx: int) -> torch.Tensor:
        """Return the raw context+1 token tensor for chunk idx (for batched SGC sampling)."""
        start = idx * (self.context_len + 1)
        return self.data[start : start + self.context_len + 1]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_shakespeare_datasets(
    config: ShakespeareConfig | None = None,
) -> tuple[ShakespeareDataset, ShakespeareDataset, ShakespeareDataset, CharTokenizer]:
    """
    Download, tokenize, and split the Shakespeare corpus.

    Returns:
        train_dataset, val_dataset, test_dataset, tokenizer
    """
    cfg = config or ShakespeareConfig()
    text = download_shakespeare(cfg.cache_path)
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = len(ids)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)

    train_data = ids[:n_train]
    val_data = ids[n_train : n_train + n_val]
    test_data = ids[n_train + n_val :]

    train_ds = ShakespeareDataset(train_data, cfg.context_len)
    val_ds = ShakespeareDataset(val_data, cfg.context_len)
    test_ds = ShakespeareDataset(test_data, cfg.context_len)

    print(f"  Vocab size:    {tokenizer.vocab_size}")
    print(f"  Total chars:   {n:,}")
    print(f"  Train chunks:  {len(train_ds):,}  ({len(train_data):,} chars)")
    print(f"  Val chunks:    {len(val_ds):,}  ({len(val_data):,} chars)")
    print(f"  Test chunks:   {len(test_ds):,}  ({len(test_data):,} chars)")

    return train_ds, val_ds, test_ds, tokenizer
