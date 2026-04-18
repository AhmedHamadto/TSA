from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 64


@dataclass
class TrainConfig:
    batch_size: int = 128
    max_epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200
    eval_every: int = 500
    checkpoint_dir: Optional[str] = None
    device: str = "auto"


@dataclass
class TaskConfig:
    name: str = "copy"        # copy | reverse | sort | even_odd
    seq_len: int = 20
    n_train: int = 50_000
    n_val: int = 5_000
    vocab_size: int = 32      # should match ModelConfig.vocab_size


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    wandb_project: Optional[str] = "tsa"
    seed: int = 42


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(
        name=raw.get("name", "experiment"),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        task=TaskConfig(**raw.get("task", {})),
        wandb_project=raw.get("wandb_project", "tsa"),
        seed=raw.get("seed", 42),
    )
