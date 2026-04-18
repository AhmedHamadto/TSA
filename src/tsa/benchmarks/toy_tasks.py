"""
Toy benchmark tasks for Genesis research.

All tasks use a decoder-only seq2seq framing:
    full sequence = [BOS] src_tokens [SEP] tgt_tokens [EOS]

The model receives full_seq[:-1] as input and predicts full_seq[1:] as targets.
Loss is masked to zero on the source portion — the model is only scored on
whether it correctly generates the output after the SEP token.

This framing means any decoder-only architecture can be evaluated on any task
with no task-specific modifications, making it clean for architecture comparisons.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset

# Special token IDs — consistent across all tasks
PAD = 0   # padding (unused in generation, reserved)
BOS = 1   # beginning of sequence
EOS = 2   # end of sequence
SEP = 3   # separator between source and target
NUM_SPECIAL = 4  # regular vocab tokens start here

TaskName = Literal["copy", "reverse", "sort", "even_odd"]


class ToyTaskDataset(Dataset):
    """
    Dataset for a single toy task.

    Args:
        task:       "copy" | "reverse" | "sort" | "even_odd"
        seq_len:    length of the source (and target) sequence
        n_samples:  number of samples to generate
        vocab_size: total vocab size including special tokens (must be > 4)
        seed:       RNG seed for reproducibility
    """

    def __init__(
        self,
        task: TaskName,
        seq_len: int,
        n_samples: int,
        vocab_size: int = 32,
        seed: int = 42,
    ) -> None:
        if vocab_size <= NUM_SPECIAL:
            raise ValueError(f"vocab_size must be > {NUM_SPECIAL}, got {vocab_size}")
        if task == "even_odd" and seq_len % 2 != 0:
            raise ValueError("even_odd task requires even seq_len")

        self.task = task
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        rng = torch.Generator()
        rng.manual_seed(seed)

        # Source tokens drawn from the regular vocab (no special tokens in the data)
        self.sources = torch.randint(
            NUM_SPECIAL, vocab_size, (n_samples, seq_len), generator=rng
        )
        self.targets = self._compute_targets(self.sources)

    def _compute_targets(self, sources: torch.Tensor) -> torch.Tensor:
        """Compute the target sequence for each source based on task type."""
        if self.task == "copy":
            return sources.clone()

        elif self.task == "reverse":
            return sources.flip(dims=[1])

        elif self.task == "sort":
            # Sort token IDs ascending — tests whether the model can learn ordering
            return sources.sort(dim=1).values

        elif self.task == "even_odd":
            # Rearrange: all even-indexed tokens first, then odd-indexed tokens
            # e.g. [a, b, c, d] → [a, c, b, d]
            even = sources[:, 0::2]
            odd = sources[:, 1::2]
            return torch.cat([even, odd], dim=1)

        raise ValueError(f"Unknown task: {self.task!r}")

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: input token ids  — shape (2 * seq_len + 2,)
            y: target token ids — shape (2 * seq_len + 2,)
                positions corresponding to the source are set to -100 (masked from loss)
        """
        src = self.sources[idx]   # (seq_len,)
        tgt = self.targets[idx]   # (seq_len,)

        # Build full sequence: [BOS, src..., SEP, tgt..., EOS]
        # Total length: 1 + seq_len + 1 + seq_len + 1 = 2*seq_len + 3
        full = torch.cat([
            torch.tensor([BOS]),
            src,
            torch.tensor([SEP]),
            tgt,
            torch.tensor([EOS]),
        ])

        x = full[:-1]        # input: drop last token  → length 2*seq_len + 2
        y = full[1:].clone() # targets: shift left by 1 → length 2*seq_len + 2

        # Mask the source portion from the loss.
        # y = [src[0], src[1], ..., src[N-1], SEP, tgt[0], ...]
        #       ^---- these seq_len+1 positions are masked ----^
        # We only compute loss on the output tokens (tgt + EOS).
        y[: self.seq_len + 1] = -100

        return x, y


def make_dataloaders(
    task: TaskName,
    seq_len: int,
    vocab_size: int,
    n_train: int,
    n_val: int,
    batch_size: int,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for a toy task."""
    train_ds = ToyTaskDataset(task, seq_len, n_train, vocab_size, seed=seed)
    val_ds = ToyTaskDataset(task, seq_len, n_val, vocab_size, seed=seed + 1)

    # pin_memory only works on CUDA; disable on MPS/CPU to avoid warnings
    _pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=_pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=_pin
    )
    return train_loader, val_loader
