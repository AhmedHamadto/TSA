"""
enwik8 character-level language modeling dataset.

Downloads the first 10^8 bytes of English Wikipedia (enwik8) from the
canonical source at http://mattmahoney.net/dc/enwik8.zip.

Standard train/val/test split:
  train: first 90M characters
  val:   next  5M characters
  test:  final 5M characters

Decision D032: raw XML kept — no stripping.
  Reason: enwik8 is a byte-level benchmark. Stripping XML would:
    (a) change the byte count, invalidating standard BPC comparisons
    (b) require a parser that introduces its own hyperparameters
    (c) make our result incomparable to published enwik8 numbers
  The model learns to handle XML syntax as part of the vocabulary.

Character tokenization is identical to Shakespeare (CharTokenizer from
tsa.benchmarks.shakespeare), making the two experiments directly comparable.
The enwik8 vocabulary is larger (~200 chars including Unicode) vs Shakespeare (~65).
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
import urllib.request

from tsa.benchmarks.shakespeare import CharTokenizer, ShakespeareDataset

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data"
DEFAULT_CACHE_ZIP = DEFAULT_CACHE_DIR / "enwik8.zip"
DEFAULT_CACHE_TXT = DEFAULT_CACHE_DIR / "enwik8.txt"

# Standard enwik8 split sizes (characters)
TRAIN_SIZE = 90_000_000
VAL_SIZE   =  5_000_000
# test = remainder (~5M)


@dataclass
class Enwik8Config:
    context_len: int = 256
    cache_dir: str | None = None  # None → DEFAULT_CACHE_DIR


def download_enwik8(cache_dir: Path | None = None) -> str:
    """
    Download and extract enwik8, cache locally, return text.

    enwik8 is stored as a ZIP containing a single binary file named 'enwik8'.
    The file is UTF-8 (mostly ASCII) with some Unicode characters.
    """
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    zip_path = cache / "enwik8.zip"
    txt_path = cache / "enwik8.txt"

    if not txt_path.exists():
        if not zip_path.exists():
            print(f"  Downloading enwik8 → {zip_path} …")
            urllib.request.urlretrieve(ENWIK8_URL, zip_path)
            print(f"  Downloaded: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

        print(f"  Extracting enwik8 …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # The archive contains a file named 'enwik8' (no extension)
            raw_bytes = zf.read("enwik8")

        # Decode as UTF-8, replace any non-decodable bytes with the replacement char
        text = raw_bytes.decode("utf-8", errors="replace")
        txt_path.write_text(text, encoding="utf-8")
        print(f"  Extracted: {len(text):,} characters → {txt_path}")
    else:
        print(f"  Using cached enwik8: {txt_path} ({txt_path.stat().st_size / 1024 / 1024:.0f} MB)")
        text = txt_path.read_text(encoding="utf-8")

    return text


def build_enwik8_datasets(
    config: Enwik8Config | None = None,
) -> tuple[ShakespeareDataset, ShakespeareDataset, ShakespeareDataset, CharTokenizer]:
    """
    Download, tokenize, and split enwik8 into train/val/test.

    Uses the standard 90M/5M/5M character split.
    Reuses ShakespeareDataset for the fixed-stride windowing —
    the dataset class is corpus-agnostic.

    Returns:
        train_dataset, val_dataset, test_dataset, tokenizer
    """
    cfg = config or Enwik8Config()

    text = download_enwik8(cache_dir=cfg.cache_dir)

    # Build vocabulary from full corpus (all 10^8 chars)
    print(f"  Building vocabulary …")
    tokenizer = CharTokenizer(text)

    # Encode to integer tensor
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Standard split
    train_data = ids[:TRAIN_SIZE]
    val_data   = ids[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    test_data  = ids[TRAIN_SIZE + VAL_SIZE :]

    train_ds = ShakespeareDataset(train_data, cfg.context_len)
    val_ds   = ShakespeareDataset(val_data,   cfg.context_len)
    test_ds  = ShakespeareDataset(test_data,  cfg.context_len)

    print(f"  Vocab size:    {tokenizer.vocab_size}")
    print(f"  Total chars:   {len(ids):,}")
    print(f"  Train chunks:  {len(train_ds):,}  ({len(train_data):,} chars)")
    print(f"  Val chunks:    {len(val_ds):,}  ({len(val_data):,} chars)")
    print(f"  Test chunks:   {len(test_ds):,}  ({len(test_data):,} chars)")

    return train_ds, val_ds, test_ds, tokenizer
