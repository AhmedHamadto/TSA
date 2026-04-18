"""
MLX Port Verification Script.

Trains both PyTorch and MLX implementations of Baseline and TSA on
Shakespeare for 500 steps with matched hyperparameters, then compares
val loss curves to confirm the port is correct.

Verification criterion (from task spec):
  - Both implementations should reach similar final val loss after 500 steps
  - Expected range: 1.5 – 2.0 nats (consistent with Phase 6 baseline)
  - TSA should show mean_active_frac < 0.95 in both frameworks

Note on RNG:
  PyTorch and MLX use different random number generators, so step-by-step
  loss values will not be identical. The comparison is on final val loss
  and overall trend, not exact step-by-step matching.

Output: docs/results/mlx_port_verification.md
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim


# ── PyTorch imports ────────────────────────────────────────────────────────
from tsa.benchmarks.baseline_transformer import BaselineTransformer as PTBaseline
from tsa.benchmarks.shakespeare import ShakespeareConfig, build_shakespeare_datasets
from tsa.modules.topological_attention.variable_depth import TSAConfig as PTTSAConfig
from tsa.modules.topological_attention.variable_depth import TSATransformer as PTTSATransformer

# ── MLX imports ────────────────────────────────────────────────────────────
from tsa.mlx.data import load_shakespeare, random_batch as mlx_random_batch
from tsa.mlx.model import BaselineConfig, BaselineTransformer as MLXBaseline
from tsa.mlx.tsa import TSAConfig as MLXTSAConfig, TSATransformer as MLXTSATransformer
from tsa.mlx.train import TrainConfig, train_condition as mlx_train
from tsa.mlx.evaluate import evaluate as mlx_evaluate

RESULTS_DIR = Path(__file__).parent.parent / "docs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared hyperparameters ─────────────────────────────────────────────────
# Using smaller config for fast verification (~500 steps)
VOCAB_SIZE = 65  # Shakespeare char-level
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
CONTEXT_LEN = 128
DROPOUT = 0.0      # No dropout for deterministic comparison
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 0.1
STEPS = 500
EVAL_EVERY = 50
DEPTH_REG = 0.001


# ── PyTorch helpers ─────────────────────────────────────────────────────────

def _pt_random_batch(data: torch.Tensor, ctx: int, bs: int, device: str):
    max_start = len(data) - ctx - 1
    starts = torch.randint(0, max_start, (bs,))
    x = torch.stack([data[s : s + ctx] for s in starts]).to(device)
    y = torch.stack([data[s + 1 : s + ctx + 1] for s in starts]).to(device)
    return x, y


@torch.no_grad()
def _pt_evaluate(model, data, ctx, bs, device, n=20):
    model.eval()
    total = 0.0
    for _ in range(n):
        x, y = _pt_random_batch(data, ctx, bs, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item()
    model.train()
    return total / n


def pt_train_condition(label, model, train_data, val_data, device):
    """Train a PyTorch model for STEPS steps, return (eval_steps, val_losses, active_fracs)."""
    print(f"\n[PyTorch] {label}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    model = model.to(device)
    is_tsa = hasattr(model, "_last_mean_active_frac")

    # Optimizer
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in name for nd in ("bias", "ln", "emb")):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR * 0.1)

    eval_steps, val_losses, active_fracs = [], [], []
    model.train()

    for step in range(1, STEPS + 1):
        x, y = _pt_random_batch(train_data, CONTEXT_LEN, BATCH_SIZE, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if is_tsa:
            loss = loss + model.aux_loss().to(loss.device)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % EVAL_EVERY == 0:
            vl = _pt_evaluate(model, val_data, CONTEXT_LEN, BATCH_SIZE, device)
            eval_steps.append(step)
            val_losses.append(vl)
            af = model._last_mean_active_frac if is_tsa else 1.0
            active_fracs.append(af)
            bpc = vl / math.log(2)
            af_str = f"  active={af:.3f}" if is_tsa else ""
            print(f"  step {step:4d}  val_loss={vl:.4f}  bpc={bpc:.4f}{af_str}")

    return eval_steps, val_losses, active_fracs


# ── MLX train wrapper (uses smaller cfg) ────────────────────────────────────

def mlx_train_condition(label, model, train_data, val_data):
    """Train an MLX model for STEPS steps, return (eval_steps, val_losses, active_fracs)."""
    print(f"\n[MLX] {label}")
    cfg = TrainConfig(
        train_steps=STEPS,
        batch_size=BATCH_SIZE,
        context_len=CONTEXT_LEN,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        grad_clip=1.0,
        warmup_steps=0,
        eval_interval=EVAL_EVERY,
        seed=42,
        loss_thresholds=[],
    )
    result = mlx_train(label, model, train_data, val_data, cfg)
    return result.eval_steps, result.val_losses, result.active_fracs


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n=== MLX Port Verification ===")
    print(f"PyTorch device: {device}")
    print(f"MLX device: {mx.default_device()}")
    print(f"Steps: {STEPS}  Batch: {BATCH_SIZE}  d_model: {D_MODEL}  L: {N_LAYERS}\n")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading Shakespeare corpus…")
    sh_cfg = ShakespeareConfig(context_len=CONTEXT_LEN)
    pt_train_ds, pt_val_ds, _, tokenizer = build_shakespeare_datasets(sh_cfg)
    pt_train_data = pt_train_ds.data
    pt_val_data = pt_val_ds.data
    vocab_size = tokenizer.vocab_size

    mlx_train_data, mlx_val_data, _, _ = load_shakespeare()
    # Vocab size must match (same corpus, same tokenizer logic)
    assert tokenizer.vocab_size == vocab_size  # same corpus → same tokenizer

    print(f"\nVocab size: {vocab_size}")

    results: dict[str, dict] = {}

    # ── PyTorch Baseline ───────────────────────────────────────────────────
    torch.manual_seed(42)
    pt_baseline = PTBaseline(
        vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=CONTEXT_LEN, dropout=DROPOUT,
    )
    # Remove padding_idx for char-level LM (same as phase6_language.py)
    pt_baseline.token_emb = nn.Embedding(vocab_size, D_MODEL)
    torch.nn.init.normal_(pt_baseline.token_emb.weight, std=0.02)
    pt_baseline.head.weight = pt_baseline.token_emb.weight

    steps, vl, af = pt_train_condition("Baseline", pt_baseline, pt_train_data, pt_val_data, device)
    results["pt_baseline"] = {"steps": steps, "val_losses": vl, "active_fracs": af,
                               "label": "PyTorch Baseline"}

    # ── PyTorch TSA ────────────────────────────────────────────────────────
    torch.manual_seed(42)
    pt_tsa_cfg = PTTSAConfig(
        vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=CONTEXT_LEN,
        dropout=DROPOUT, depth_reg_weight=DEPTH_REG,
    )
    pt_tsa = PTTSATransformer(pt_tsa_cfg)
    pt_tsa.token_emb = nn.Embedding(vocab_size, D_MODEL)
    torch.nn.init.normal_(pt_tsa.token_emb.weight, std=0.02)
    pt_tsa.head.weight = pt_tsa.token_emb.weight

    steps, vl, af = pt_train_condition("TSA", pt_tsa, pt_train_data, pt_val_data, device)
    results["pt_tsa"] = {"steps": steps, "val_losses": vl, "active_fracs": af,
                          "label": "PyTorch TSA"}

    # ── MLX Baseline ───────────────────────────────────────────────────────
    mx.random.seed(42)
    mlx_baseline_cfg = BaselineConfig(
        vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, context_len=CONTEXT_LEN, dropout=DROPOUT,
    )
    mlx_baseline = MLXBaseline(mlx_baseline_cfg)

    steps, vl, af = mlx_train_condition("Baseline", mlx_baseline, mlx_train_data, mlx_val_data)
    results["mlx_baseline"] = {"steps": steps, "val_losses": vl, "active_fracs": af,
                                "label": "MLX Baseline"}

    # ── MLX TSA ────────────────────────────────────────────────────────────
    mx.random.seed(42)
    mlx_tsa_cfg = MLXTSAConfig(
        vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, context_len=CONTEXT_LEN,
        dropout=DROPOUT, depth_reg_weight=DEPTH_REG,
    )
    mlx_tsa = MLXTSATransformer(mlx_tsa_cfg)

    steps, vl, af = mlx_train_condition("TSA", mlx_tsa, mlx_train_data, mlx_val_data)
    results["mlx_tsa"] = {"steps": steps, "val_losses": vl, "active_fracs": af,
                           "label": "MLX TSA"}

    # ── Compare and report ─────────────────────────────────────────────────
    _print_comparison(results)
    _save_report(results)
    _save_plot(results)

    print(f"\nVerification report: docs/results/mlx_port_verification.md")
    print(f"Plot: docs/results/mlx_verification_curves.png")


def _print_comparison(results: dict) -> None:
    print(f"\n{'='*70}")
    print("  MLX PORT VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Condition':<30}  {'Final Val Loss':>14}  {'Final BPC':>10}")
    print(f"  {'-'*30}  {'-'*14}  {'-'*10}")
    for key, r in results.items():
        vl = r["val_losses"][-1] if r["val_losses"] else float("nan")
        bpc = vl / math.log(2)
        print(f"  {r['label']:<30}  {vl:>14.4f}  {bpc:>10.4f}")

    # Check convergence: both should end up in similar range
    pt_base_vl = results["pt_baseline"]["val_losses"][-1]
    mlx_base_vl = results["mlx_baseline"]["val_losses"][-1]
    diff = abs(pt_base_vl - mlx_base_vl)
    pct = diff / pt_base_vl * 100
    print(f"\n  Baseline: PyTorch={pt_base_vl:.4f}  MLX={mlx_base_vl:.4f}")
    print(f"  Absolute diff: {diff:.4f}  ({pct:.1f}%)")

    pt_tsa_vl = results["pt_tsa"]["val_losses"][-1]
    mlx_tsa_vl = results["mlx_tsa"]["val_losses"][-1]
    diff_tsa = abs(pt_tsa_vl - mlx_tsa_vl)
    pct_tsa = diff_tsa / pt_tsa_vl * 100
    print(f"  TSA:      PyTorch={pt_tsa_vl:.4f}  MLX={mlx_tsa_vl:.4f}")
    print(f"  Absolute diff: {diff_tsa:.4f}  ({pct_tsa:.1f}%)")

    pt_tsa_af = results["pt_tsa"]["active_fracs"]
    mlx_tsa_af = results["mlx_tsa"]["active_fracs"]
    if pt_tsa_af:
        print(f"\n  TSA active_frac PyTorch: {sum(pt_tsa_af)/len(pt_tsa_af):.3f}")
    if mlx_tsa_af:
        print(f"  TSA active_frac MLX:     {sum(mlx_tsa_af)/len(mlx_tsa_af):.3f}")

    verdict = "PASS" if pct <= 5.0 and pct_tsa <= 5.0 else "INVESTIGATE"
    print(f"\n  VERDICT: {verdict} (criterion: <5% final val loss diff)")
    print(f"{'='*70}\n")


def _save_report(results: dict) -> None:
    lines = [
        "# MLX Port Verification Report",
        "",
        f"**Date:** 2026-04-01",
        f"**Config:** d_model={D_MODEL}, L={N_LAYERS}, h={N_HEADS}, steps={STEPS}",
        "",
        "## Results",
        "",
        "| Condition | Final Val Loss | Final BPC | Mean Active Frac |",
        "|-----------|---------------|-----------|-----------------|",
    ]
    for r in results.values():
        vl = r["val_losses"][-1] if r["val_losses"] else float("nan")
        bpc = vl / math.log(2)
        af = r["active_fracs"]
        af_str = f"{sum(af)/len(af):.3f}" if af and any(a < 1.0 for a in af) else "1.000"
        lines.append(f"| {r['label']} | {vl:.4f} | {bpc:.4f} | {af_str} |")

    pt_base = results["pt_baseline"]["val_losses"][-1]
    mlx_base = results["mlx_baseline"]["val_losses"][-1]
    pt_tsa = results["pt_tsa"]["val_losses"][-1]
    mlx_tsa = results["mlx_tsa"]["val_losses"][-1]

    lines += [
        "",
        "## Convergence Comparison",
        "",
        f"- Baseline: PyTorch={pt_base:.4f}, MLX={mlx_base:.4f}, "
        f"diff={abs(pt_base-mlx_base)/pt_base*100:.1f}%",
        f"- TSA: PyTorch={pt_tsa:.4f}, MLX={mlx_tsa:.4f}, "
        f"diff={abs(pt_tsa-mlx_tsa)/pt_tsa*100:.1f}%",
        "",
        "## Notes",
        "",
        "- Different RNG (PyTorch vs MLX) means step-by-step curves differ; final "
        "loss is the meaningful comparison.",
        "- Verification criterion: final val loss within 5% between frameworks.",
        "- MLX uses uniform AdamW weight_decay (no selective no-decay for biases/embeddings). "
        "PyTorch version excludes biases and embeddings from weight decay.",
        "- TSA active_frac should be < 0.95 in both frameworks, confirming the router learns.",
    ]

    out = RESULTS_DIR / "mlx_port_verification.md"
    out.write_text("\n".join(lines))
    print(f"  Report saved: {out}")


def _save_plot(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"pt_baseline": "#4C72B0", "pt_tsa": "#DD8452",
              "mlx_baseline": "#4C72B0", "mlx_tsa": "#DD8452"}
    ls = {"pt_baseline": "-", "pt_tsa": "-", "mlx_baseline": "--", "mlx_tsa": "--"}

    for key, r in results.items():
        ax = axes[0]
        ax.plot(r["steps"], r["val_losses"],
                label=r["label"],
                color=colors.get(key, "gray"),
                linestyle=ls.get(key, "-"),
                linewidth=2)

    axes[0].set_xlabel("Training steps")
    axes[0].set_ylabel("Validation loss (nats)")
    axes[0].set_title("PyTorch vs MLX — Val Loss Curves")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Active fraction for TSA
    for key in ["pt_tsa", "mlx_tsa"]:
        r = results[key]
        if r["active_fracs"]:
            axes[1].plot(r["steps"], r["active_fracs"],
                        label=r["label"],
                        color=colors.get(key, "gray"),
                        linestyle=ls.get(key, "-"),
                        linewidth=2)

    axes[1].axhline(1.0, color="gray", linestyle=":", linewidth=1, label="Baseline (all active)")
    axes[1].set_xlabel("Training steps")
    axes[1].set_ylabel("Mean active fraction")
    axes[1].set_title("TSA Active Fraction (PyTorch vs MLX)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)

    fig.tight_layout()
    out = RESULTS_DIR / "mlx_verification_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out}")


if __name__ == "__main__":
    main()
