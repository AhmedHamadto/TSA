"""
Training loop for MLX models.

Supports both BaselineTransformer and TSATransformer with:
  - Cosine LR schedule with linear warmup
  - AdamW optimizer (gradient clipping via optim.clip_grad_norm)
  - Periodic evaluation and val loss tracking
  - Token-layer ops tracking for compute efficiency comparison

MLX training pattern:
  1. Define loss_fn(model, x, y) → scalar loss
  2. loss_and_grad = nn.value_and_grad(model, loss_fn)
  3. Each step: loss, grads = loss_and_grad(model, x, y)
  4. optimizer.update(model, grads)
  5. mx.eval(model.parameters(), optimizer.state)  ← forces GPU execution

mx.eval() is called each step because MLX is lazy — ops are queued but not
executed until materialization is requested. Without mx.eval(), all steps would
batch up and execute together when we finally read a value.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from tsa.mlx.data import random_batch
from tsa.mlx.evaluate import evaluate
from tsa.mlx.model import BaselineTransformer
from tsa.mlx.tsa import TSATransformer


@dataclass
class TrainConfig:
    train_steps: int = 5_000
    batch_size: int = 64
    context_len: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200
    eval_interval: int = 250
    seed: int = 42
    loss_thresholds: list[float] = field(default_factory=lambda: [3.0, 2.5, 2.0])


@dataclass
class ConditionResult:
    label: str
    val_losses: list[float]
    eval_steps: list[int]
    token_layer_ops: list[float]   # cumulative ops at each eval
    active_fracs: list[float]      # TSA only — mean_active_frac at each eval
    steps_to_threshold: dict[float, int | None]
    final_val_loss: float
    final_bpc: float
    elapsed: float
    n_params: int


def _cosine_lr(step: int, T_max: int, lr: float, eta_min: float) -> float:
    """Cosine annealing with linear warmup built in."""
    return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * step / T_max)) / 2


def train_condition(
    label: str,
    model: BaselineTransformer | TSATransformer,
    train_data: mx.array,
    val_data: mx.array,
    cfg: TrainConfig,
) -> ConditionResult:
    """
    Train one model condition for cfg.train_steps steps.

    Handles both Baseline and TSA: detects TSATransformer by checking for
    _depth_reg_loss attribute and adds it to the task loss when present.
    """
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    n_params = model.num_params()
    print(f"  Params: {n_params:,}")

    is_tsa = isinstance(model, TSATransformer)
    n_layers = model.config.n_layers

    # AdamW optimizer. MLX applies weight_decay to ALL parameters.
    # This differs from PyTorch's selective weight decay (no decay for biases/embeddings).
    # The difference is small and doesn't affect the 1-2% verification threshold.
    optimizer = optim.AdamW(
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # Define the loss function. For TSA, depth_reg is added here so gradients
    # flow back to the router weights via the lazy _depth_reg_loss array.
    if is_tsa:
        def loss_fn(model: TSATransformer, x: mx.array, y: mx.array) -> mx.array:
            logits = model(x)
            task_loss = mx.mean(
                nn.losses.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
                )
            )
            # _depth_reg_loss was set during model(x) above — it's lazy and
            # carries gradient connectivity to router weights
            return task_loss + model.config.depth_reg_weight * model._depth_reg_loss
    else:
        def loss_fn(model: BaselineTransformer, x: mx.array, y: mx.array) -> mx.array:
            logits = model(x)
            return mx.mean(
                nn.losses.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
                )
            )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    val_losses: list[float] = []
    eval_steps: list[int] = []
    tl_ops_log: list[float] = []
    active_frac_log: list[float] = []
    steps_to_threshold: dict[float, int | None] = {t: None for t in cfg.loss_thresholds}

    cumulative_ops = 0.0
    active_frac = 1.0  # running value for ops tracking

    mx.random.seed(cfg.seed)
    model.train()
    t0 = time.time()

    for step in range(1, cfg.train_steps + 1):
        # ── LR schedule: linear warmup then cosine decay ─────────────────────
        if step < cfg.warmup_steps:
            lr = cfg.lr * step / cfg.warmup_steps
        else:
            lr = _cosine_lr(step, cfg.train_steps, cfg.lr, cfg.lr * 0.1)
        optimizer.learning_rate = lr

        # ── Batch + forward + backward ───────────────────────────────────────
        x, y = random_batch(train_data, cfg.context_len, cfg.batch_size)
        loss, grads = loss_and_grad(model, x, y)

        # ── Gradient clipping ─────────────────────────────────────────────────
        # clip_grad_norm returns (clipped_grads, total_norm) — unpack to avoid
        # passing the tuple as gradients to the optimizer
        grads, _ = optim.clip_grad_norm(grads, max_norm=cfg.grad_clip)

        # ── Parameter update ──────────────────────────────────────────────────
        optimizer.update(model, grads)

        # ── Force GPU execution ───────────────────────────────────────────────
        # MLX is lazy — without mx.eval, ops queue up and execute all at once.
        # We eval here so each step's GPU work completes before the next step.
        mx.eval(model.parameters(), optimizer.state)

        # ── Read TSA active fraction (after eval, safe to materialise) ────────
        if is_tsa:
            mx.eval(model._depth_reg_loss)
            active_frac = float(model._depth_reg_loss.item())
            model._last_mean_active_frac = active_frac

        # ── Token-layer ops tracking ──────────────────────────────────────────
        # Block 0 always runs (cost = 1 layer), remaining n-1 blocks run at
        # fraction active_frac. Effective layers = 1 + (n_layers-1) * active_frac
        effective_layers = 1 + (n_layers - 1) * active_frac
        cumulative_ops += cfg.batch_size * cfg.context_len * effective_layers

        # ── Periodic evaluation ────────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_data, cfg.context_len, cfg.batch_size)
            val_losses.append(val_loss)
            eval_steps.append(step)
            tl_ops_log.append(cumulative_ops)
            active_frac_log.append(active_frac)

            for thresh in cfg.loss_thresholds:
                if steps_to_threshold[thresh] is None and val_loss <= thresh:
                    steps_to_threshold[thresh] = step

            bpc = val_loss / math.log(2)
            af_str = f"  active={active_frac:.3f}" if is_tsa else ""
            print(f"  step {step:5d}  val_loss={val_loss:.4f}  bpc={bpc:.4f}{af_str}")

    elapsed = time.time() - t0
    final_val_loss = val_losses[-1] if val_losses else float("nan")
    final_bpc = final_val_loss / math.log(2)

    print(f"\n  Final val loss: {final_val_loss:.4f}  BPC: {final_bpc:.4f}")
    print(f"  Wall time: {elapsed:.1f}s  ({elapsed/cfg.train_steps*1000:.1f}ms/step)")
    for t, s in steps_to_threshold.items():
        print(f"  Steps to val_loss≤{t}: {s if s is not None else 'NOT REACHED'}")

    return ConditionResult(
        label=label,
        val_losses=val_losses,
        eval_steps=eval_steps,
        token_layer_ops=tl_ops_log,
        active_fracs=active_frac_log,
        steps_to_threshold=steps_to_threshold,
        final_val_loss=final_val_loss,
        final_bpc=final_bpc,
        elapsed=elapsed,
        n_params=n_params,
    )
