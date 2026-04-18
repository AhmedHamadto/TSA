"""
Character-level data loading for MLX models.

Reuses the same Shakespeare corpus file as the PyTorch version (data/tinyshakespeare.txt).
If already cached from a PyTorch run, no re-download happens.

Key difference from PyTorch version:
  - Token IDs stored as mx.int32 (MLX GPU does not expose int64 for indexing)
  - random_batch uses fancy indexing instead of a Python loop for batch assembly
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import mlx.core as mx

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
# Reuse the same cache path as tsa.benchmarks.shakespeare so the file is
# shared between PyTorch and MLX runs — no double download.
DEFAULT_CACHE = Path(__file__).parent.parent.parent.parent / "data" / "tinyshakespeare.txt"


class CharTokenizer:
    """Character-level tokenizer — identical logic to tsa.benchmarks.shakespeare."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self.itos: dict[int, str] = {i: c for c, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_shakespeare(
    cache_path: Path | str | None = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[mx.array, mx.array, mx.array, CharTokenizer]:
    """
    Load (or download) Shakespeare corpus and return train/val/test splits.

    Returns:
        train_data, val_data, test_data: mx.array of dtype int32
        tokenizer: CharTokenizer built from the full corpus
    """
    path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        print(f"  Downloading Shakespeare → {path} …")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        print(f"  Downloaded: {path.stat().st_size // 1024} KB")
    else:
        print(f"  Using cached corpus: {path} ({path.stat().st_size // 1024} KB)")

    text = path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)

    # int32: MLX GPU arrays don't support int64 for indexing operations
    ids = mx.array(tokenizer.encode(text), dtype=mx.int32)

    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_data = ids[:n_train]
    val_data = ids[n_train : n_train + n_val]
    test_data = ids[n_train + n_val :]

    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Total chars: {n:,}")
    print(f"  Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")

    return train_data, val_data, test_data, tokenizer


ENWIK8_ZIP_URL = "http://mattmahoney.net/dc/enwik8.zip"
ENWIK8_CACHE_TXT = Path(__file__).parent.parent.parent.parent / "data" / "enwik8.txt"
ENWIK8_CACHE_ZIP = Path(__file__).parent.parent.parent.parent / "data" / "enwik8.zip"

ENWIK8_TRAIN_SIZE = 90_000_000
ENWIK8_VAL_SIZE   =  5_000_000
# test = remainder (~5M)


def load_enwik8(
    cache_path: Path | str | None = None,
) -> tuple[mx.array, mx.array, mx.array, CharTokenizer]:
    """
    Load enwik8 (cached or downloaded) and return standard 90M/5M/5M splits.

    Memory note: encoding 100M chars into a Python list creates ~3GB of temporary
    objects. We use numpy.fromiter with a generator to avoid materialising the
    full list — peak RAM stays near 400MB for the numpy array.

    Returns:
        train_data, val_data, test_data: mx.array of dtype int32
        tokenizer: CharTokenizer built from the full 100M-char corpus
    """
    import zipfile

    txt_path = Path(cache_path) if cache_path is not None else ENWIK8_CACHE_TXT

    if not txt_path.exists():
        zip_path = ENWIK8_CACHE_ZIP
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            print(f"  Downloading enwik8 → {zip_path} …")
            urllib.request.urlretrieve(ENWIK8_ZIP_URL, zip_path)
            print(f"  Downloaded: {zip_path.stat().st_size // 1024 // 1024} MB")
        print("  Extracting enwik8 …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            raw_bytes = zf.read("enwik8")
        text_raw = raw_bytes.decode("utf-8", errors="replace")
        txt_path.write_text(text_raw, encoding="utf-8")
        print(f"  Extracted: {len(text_raw):,} characters → {txt_path}")
    else:
        print(f"  Using cached enwik8: {txt_path} ({txt_path.stat().st_size // 1024 // 1024} MB)")

    print("  Reading enwik8.txt …", end="", flush=True)
    text = txt_path.read_text(encoding="utf-8")
    print(f" {len(text):,} chars")

    print("  Building vocabulary …", end="", flush=True)
    tokenizer = CharTokenizer(text)
    print(f" vocab_size={tokenizer.vocab_size}")

    # Vectorized encoding: encode text to UTF-32-LE (4 bytes per code point),
    # then use numpy fancy indexing to map code points → token IDs.
    # ~100× faster than numpy.fromiter with a Python generator, because it avoids
    # per-character Python overhead and macOS memory-compression page faults.
    import numpy as np
    print("  Encoding characters …", end="", flush=True)
    chars = sorted(set(text))
    max_cp = max(ord(c) for c in chars)
    cp_to_id = np.full(max_cp + 1, 0, dtype=np.int32)
    for c, idx in tokenizer.stoi.items():
        if ord(c) <= max_cp:
            cp_to_id[ord(c)] = idx
    codepoints = np.frombuffer(text.encode("utf-32-le"), dtype=np.int32)
    ids_np = cp_to_id[codepoints]
    print(f" done ({ids_np.nbytes // 1024 // 1024} MB numpy)")

    ids = mx.array(ids_np)
    del ids_np  # free numpy array — MLX has its own copy now

    n = len(ids)
    train_data = ids[:ENWIK8_TRAIN_SIZE]
    val_data   = ids[ENWIK8_TRAIN_SIZE : ENWIK8_TRAIN_SIZE + ENWIK8_VAL_SIZE]
    test_data  = ids[ENWIK8_TRAIN_SIZE + ENWIK8_VAL_SIZE :]

    print(f"  Splits — train: {len(train_data):,}  val: {len(val_data):,}  test: {len(test_data):,}")
    return train_data, val_data, test_data, tokenizer


def random_batch(
    data: mx.array,
    context_len: int,
    batch_size: int,
) -> tuple[mx.array, mx.array]:
    """
    Sample batch_size random context windows from a token array.

    Uses fancy indexing (gather) instead of a Python loop — one MLX op
    instead of batch_size separate slice ops.

    Returns:
        x: (batch_size, context_len) int32 input tokens
        y: (batch_size, context_len) int32 target tokens (x shifted right by 1)
    """
    max_start = len(data) - context_len - 1
    assert max_start > 0, (
        f"Data length {len(data)} too short for context_len={context_len}. "
        f"Need at least {context_len + 2} tokens."
    )
    # Random start positions for each sequence in the batch
    starts = mx.random.randint(0, max_start, shape=(batch_size,), dtype=mx.int32)

    # Build index matrix: each row is [starts[b], starts[b]+1, ..., starts[b]+T-1]
    t_offsets = mx.arange(context_len, dtype=mx.int32)      # (T,)
    x_indices = starts[:, None] + t_offsets[None, :]         # (B, T)
    y_indices = x_indices + 1                                # (B, T)

    x = data[x_indices]  # gather: (N,) indexed by (B, T) → (B, T)
    y = data[y_indices]
    return x, y
