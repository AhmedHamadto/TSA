"""
Experiment: BaselineTransformer vs TSATransformer on copy + sort tasks.

Measures:
  - val_accuracy    (does TSA match baseline accuracy?)
  - val_perplexity  (secondary quality metric)
  - mean_active_frac (TSA efficiency signal — fraction of tokens still active per layer)

Hypothesis supported if:
  |acc_baseline - acc_tsa| < 2%  AND  mean_active_frac << 1.0
"""
from __future__ import annotations

import time
from pathlib import Path


import torch
from tqdm import tqdm

from tsa.benchmarks.baseline_transformer import BaselineTransformer
from tsa.benchmarks.toy_tasks import make_dataloaders
from tsa.core.trainer import Trainer, TrainerConfig
from tsa.modules.topological_attention.variable_depth import TSAConfig, TSATransformer
from tsa.utils.logging import ExperimentLogger


# ------------------------------------------------------------------
# Shared config — identical for both models so comparison is fair
# ------------------------------------------------------------------
MODEL_KWARGS = dict(
    vocab_size=32, d_model=128, n_heads=4, n_layers=6,
    d_ff=512, max_seq_len=64, dropout=0.1,
)
DATA_KWARGS = dict(
    seq_len=20, vocab_size=32,
    n_train=10_000, n_val=1_000,
    batch_size=256, seed=42,
)
TRAINER_CFG = TrainerConfig(
    max_epochs=10,
    grad_clip=1.0,
    eval_every=100,
    log_every=9999,   # suppress step-level noise
    warmup_steps=50,
    device="auto",
)


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------
def run(model, task: str, label: str) -> dict:
    train_loader, val_loader = make_dataloaders(task, **DATA_KWARGS)
    logger = ExperimentLogger(project=None, name=label, config={}, log_dir="experiments")
    trainer = Trainer(model, TRAINER_CFG, logger)
    optimizer = model.configure_optimizers(lr=3e-4, weight_decay=0.1)

    t0 = time.time()
    best = trainer.train(train_loader, val_loader, optimizer)
    elapsed = time.time() - t0

    # Snapshot mean_active_frac after training
    active_frac: float | None = None
    if hasattr(model, "get_extra_logs"):
        logs = model.get_extra_logs()
        if "train/mean_active_frac" in logs:
            active_frac = logs["train/mean_active_frac"]

    return {
        "accuracy":    best.get("best_val_accuracy", float("nan")),
        "perplexity":  best.get("best_val_perplexity", float("nan")),
        "step":        best.get("best_step", -1),
        "active_frac": active_frac,
        "params":      model.get_num_params(),
        "time_s":      elapsed,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    tasks = ["copy", "sort"]
    rows: list[dict] = []

    for task in tasks:
        print(f"\n{'='*55}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*55}")

        print(f"\n→ Baseline [{task}]")
        baseline = BaselineTransformer(**MODEL_KWARGS)
        r = run(baseline, task, f"baseline_{task}")
        rows.append({"model": "Baseline", "task": task, **r})

        print(f"\n→ TSA  λ=0.01  [{task}]")
        tsa_cfg = TSAConfig(**MODEL_KWARGS, depth_reg_weight=0.01)
        tsa = TSATransformer(tsa_cfg)
        r = run(tsa, task, f"tsa_{task}")
        rows.append({"model": "TSA", "task": task, **r})

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    sep = "=" * 72
    print(f"\n\n{sep}")
    print("  GENESIS PHASE 1 RESULTS — Baseline vs TSA")
    print(sep)
    print(f"  {'Model':<10} {'Task':<6} {'Acc':>8} {'PPL':>8} {'ActiveFrac':>12} {'Params':>10} {'Time':>7}")
    print(f"  {'-'*68}")
    for r in rows:
        af = f"{r['active_frac']:.3f}" if r["active_frac"] is not None else "  1.000"
        print(
            f"  {r['model']:<10} {r['task']:<6} "
            f"{r['accuracy']:>8.4f} {r['perplexity']:>8.3f} "
            f"{af:>12} {r['params']:>10,} {r['time_s']:>6.1f}s"
        )
    print(sep)

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n  HYPOTHESIS CHECK:")
    for task in tasks:
        b = next(r for r in rows if r["model"] == "Baseline" and r["task"] == task)
        t = next(r for r in rows if r["model"] == "TSA"      and r["task"] == task)
        acc_delta = t["accuracy"] - b["accuracy"]
        af = t["active_frac"]
        efficiency = (1.0 - af) if af is not None else 0.0

        print(f"\n  [{task}]")
        print(f"    acc Δ = {acc_delta:+.4f}  (|Δ| < 0.02 target)")
        if af is not None:
            print(f"    efficiency gain = {efficiency:.1%}  (> 5% target)")
        if abs(acc_delta) < 0.02 and efficiency > 0.05:
            verdict = "✓  HYPOTHESIS SUPPORTED"
        elif abs(acc_delta) < 0.02 and efficiency <= 0.05:
            verdict = "~  Routing not firing — try higher λ"
        else:
            verdict = "✗  Accuracy gap too large — try lower λ"
        print(f"    → {verdict}")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
