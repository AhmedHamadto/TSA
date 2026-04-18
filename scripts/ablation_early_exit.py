"""
Ablation Study — Early Exit baseline comparison for TSA.

Trains an EarlyExitTransformer on Shakespeare (identical config to Phase 6)
and evaluates it at multiple confidence thresholds to build a quality-efficiency
Pareto curve. Compares against Phase 6 TSA and Baseline results.

Research question:
  "Why not just use early exit instead of TSA?"

Key differences studied:
  1. Training cost — EE is identical to Baseline (all layers always run);
     TSA reduces effective compute at BOTH train and inference.
  2. Routing mechanism — EE uses max-softmax confidence (no learned gate);
     TSA learns a per-token halt_prob from context.
  3. Quality at matched active fraction — is TSA val_loss lower than EE's?

Decision D049: see docs/plans/decision_log.md.

Architecture: d_model=256, n_layers=6, n_heads=8, d_ff=1024, vocab=65, ctx=128
Training: 5,000 steps, batch=64, AdamW lr=3e-4 (matches Phase 6)

Runtime estimate: ~25 min on M1 Pro (1 condition × 5k steps, then threshold sweep).
Run foreground in terminal. Close GPU-heavy apps.

Output:
  docs/results/ablation_early_exit.md
  docs/paper/figures/early_exit_comparison.pdf / .png
"""
from __future__ import annotations

import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


from tsa.mlx.data import load_shakespeare, random_batch
from tsa.mlx.evaluate import evaluate, bpc, active_fraction_savings
from tsa.mlx.early_exit import (
    EarlyExitConfig,
    EarlyExitTransformer,
    early_exit_loss,
    evaluate_early_exit,
)
from tsa.mlx.train import TrainConfig, _cosine_lr

def _detect_hardware() -> str:
    """Auto-detect hardware description for reproducibility reports."""
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            mem_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip())
            mem_gb = mem_bytes // (1024 ** 3)
            return f"Apple {chip}, {mem_gb} GB unified memory"
        except Exception:
            pass
    return f"{platform.processor() or platform.machine()}, {platform.system()}"


RESULTS_DIR = Path(__file__).parent.parent / "docs" / "results"
FIGURES_DIR = Path(__file__).parent.parent / "docs" / "paper" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Phase 6 reference results (from docs/results/ablation_lambda.md + CLAUDE.md) ─

PHASE6_BASELINE_LOSS  = 1.4422   # Baseline, no routing
PHASE6_BASELINE_AF    = 1.000
PHASE6_TSA_LOSS       = 1.4482   # TSA-only, λ=0.001, active_frac=0.726
PHASE6_TSA_AF         = 0.726    # 27.4% ops saved

# ── Config — matches Phase 6 exactly ─────────────────────────────────────────

TRAIN_CFG = TrainConfig(
    train_steps=5_000,
    batch_size=64,
    context_len=128,
    lr=3e-4,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=200,
    eval_interval=250,
    seed=42,
    loss_thresholds=[3.0, 2.5, 2.0],
)

EE_CFG = EarlyExitConfig(
    vocab_size=65,       # filled in after loading tokenizer
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    context_len=128,
    dropout=0.1,
)

# Thresholds to sweep post-training (ascending → more exits as threshold drops)
THRESHOLDS = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.40, 0.30]


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class EETrainResult:
    val_losses: list[float]
    eval_steps: list[int]
    final_val_loss: float
    final_bpc: float
    elapsed: float
    n_params: int


def train_early_exit(
    model: EarlyExitTransformer,
    train_data: mx.array,
    val_data: mx.array,
    cfg: TrainConfig,
) -> EETrainResult:
    """Train EarlyExitTransformer with uniform multi-exit CE loss."""
    print(f"\n{'='*65}")
    print("  EarlyExitTransformer — training")
    print(f"{'='*65}")
    n_params = model.num_params()
    print(f"  Params: {n_params:,}")
    print(f"  Loss: uniform mean CE across all {model.config.n_layers} exit points")
    print(f"  Note: ALL layers run every step — training cost = Baseline\n")

    optimizer = optim.AdamW(
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    def loss_fn(m: EarlyExitTransformer, x: mx.array, y: mx.array) -> mx.array:
        return early_exit_loss(m, x, y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    val_losses: list[float] = []
    eval_steps: list[int] = []

    mx.random.seed(cfg.seed)
    model.train()
    t0 = time.time()

    for step in range(1, cfg.train_steps + 1):
        # LR schedule: linear warmup then cosine decay
        if step < cfg.warmup_steps:
            lr = cfg.lr * step / cfg.warmup_steps
        else:
            lr = _cosine_lr(step, cfg.train_steps, cfg.lr, cfg.lr * 0.1)
        optimizer.learning_rate = lr

        x, y = random_batch(train_data, cfg.context_len, cfg.batch_size)
        loss, grads = loss_and_grad(model, x, y)

        grads, _ = optim.clip_grad_norm(grads, max_norm=cfg.grad_clip)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % cfg.eval_interval == 0:
            # Evaluate using the full model (no early exit — all layers)
            val_loss = evaluate(model, val_data, cfg.context_len, cfg.batch_size)
            val_losses.append(val_loss)
            eval_steps.append(step)
            b = val_loss / math.log(2)
            elapsed_so_far = time.time() - t0
            pct = step / cfg.train_steps * 100
            print(f"  step {step:5d} ({pct:4.1f}%)  val_loss={val_loss:.4f}  bpc={b:.4f}"
                  f"  elapsed={elapsed_so_far:.0f}s")

    elapsed = time.time() - t0
    final_val_loss = val_losses[-1] if val_losses else float("nan")
    final_bpc_val = final_val_loss / math.log(2)

    print(f"\n  Final val loss: {final_val_loss:.4f}  BPC: {final_bpc_val:.4f}")
    print(f"  Wall time: {elapsed:.1f}s  ({elapsed/cfg.train_steps*1000:.1f}ms/step)")

    return EETrainResult(
        val_losses=val_losses,
        eval_steps=eval_steps,
        final_val_loss=final_val_loss,
        final_bpc=final_bpc_val,
        elapsed=elapsed,
        n_params=n_params,
    )


# ── Threshold sweep ───────────────────────────────────────────────────────────

def sweep_thresholds(
    model: EarlyExitTransformer,
    val_data: mx.array,
    context_len: int,
    batch_size: int,
    thresholds: list[float],
    n_layers: int,
    n_batches: int = 30,
) -> list[dict]:
    """
    Sweep confidence thresholds and record (threshold, val_loss, active_frac, ops_saved).
    """
    print(f"\n  Sweeping {len(thresholds)} confidence thresholds …\n")
    rows = []
    for thresh in thresholds:
        loss, af = evaluate_early_exit(
            model, val_data, context_len, batch_size,
            threshold=thresh, n_batches=n_batches,
        )
        ops_saved = active_fraction_savings(af, n_layers)
        b = loss / math.log(2)
        print(f"  threshold={thresh:.2f}  val_loss={loss:.4f}  bpc={b:.4f}"
              f"  active_frac={af:.3f}  ops_saved={ops_saved:.1%}")
        rows.append({
            "threshold": thresh,
            "val_loss": loss,
            "bpc": b,
            "active_frac": af,
            "ops_saved": ops_saved,
        })
    return rows


# ── Find row closest to target active fraction ────────────────────────────────

def closest_to_af(rows: list[dict], target_af: float) -> dict:
    return min(rows, key=lambda r: abs(r["active_frac"] - target_af))


# ── Reporting ─────────────────────────────────────────────────────────────────

def write_report(
    train_result: EETrainResult,
    threshold_rows: list[dict],
    matched: dict,
) -> None:
    n_layers = EE_CFG.n_layers
    baseline_bpc = PHASE6_BASELINE_LOSS / math.log(2)
    tsa_bpc = PHASE6_TSA_LOSS / math.log(2)

    # Quality delta at matched active fraction vs TSA
    ee_matched_loss = matched["val_loss"]
    tsa_delta = (ee_matched_loss - PHASE6_TSA_LOSS) / PHASE6_TSA_LOSS * 100
    baseline_delta = (ee_matched_loss - PHASE6_BASELINE_LOSS) / PHASE6_BASELINE_LOSS * 100

    report = f"""# Ablation Study — Early Exit vs TSA

**Date:** 2026-04-02
**Hardware:** {_detect_hardware()}
**Framework:** MLX (Metal GPU)
**Model:** d_model=256, n_layers=6, n_heads=8, d_ff=1024, vocab=65 (Shakespeare)
**Config:** 5,000 steps, batch=64, ctx=128, lr=0.0003
**Training time:** {train_result.elapsed:.0f}s ({train_result.elapsed/60:.1f} min)

---

## Summary

Early Exit (EE) is the standard per-token inference-time baseline for early-stopping
models. It trains a multi-head transformer with N auxiliary exit classifiers (one per
layer), using the **same dense training cost as a standard baseline**. Compute savings
arise only at inference via a per-token confidence threshold.

This ablation answers: **"why not just use early exit instead of TSA?"**

---

## Training Result (full model, no early exit)

| Metric | EarlyExit | Baseline (P6) |
|---|---|---|
| Val Loss | {train_result.final_val_loss:.4f} | {PHASE6_BASELINE_LOSS:.4f} |
| BPC | {train_result.final_bpc:.4f} | {baseline_bpc:.4f} |
| Active Frac (train) | 1.000 | 1.000 |
| Params | {train_result.n_params:,} | {train_result.n_params:,} |

*Note: EarlyExit has N extra exit LayerNorms but reuses tied embeddings — near-identical
parameter count to Baseline.*

---

## Threshold Sweep (inference active fraction)

| Threshold | Val Loss | BPC | Active Frac | Ops Saved |
|---|---|---|---|---|
"""
    for r in threshold_rows:
        marker = " ← matched" if abs(r["active_frac"] - PHASE6_TSA_AF) < 0.04 else ""
        report += (
            f"| {r['threshold']:.2f} | {r['val_loss']:.4f} | {r['bpc']:.4f} "
            f"| {r['active_frac']:.3f} | {r['ops_saved']:.1%} |{marker}\n"
        )

    report += f"""
---

## Quality Comparison at Matched Active Fraction (α ≈ {PHASE6_TSA_AF:.3f})

| Condition | Val Loss | BPC | Active Frac | Ops Saved | Δ vs Baseline |
|---|---|---|---|---|---|
| Baseline | {PHASE6_BASELINE_LOSS:.4f} | {baseline_bpc:.4f} | 1.000 | 0.0% | 0.0% |
| TSA (Phase 6) | {PHASE6_TSA_LOSS:.4f} | {tsa_bpc:.4f} | {PHASE6_TSA_AF:.3f} | {active_fraction_savings(PHASE6_TSA_AF, n_layers):.1%} | +{(PHASE6_TSA_LOSS-PHASE6_BASELINE_LOSS)/PHASE6_BASELINE_LOSS*100:.2f}% |
| EarlyExit (threshold={matched['threshold']:.2f}) | {matched['val_loss']:.4f} | {matched['bpc']:.4f} | {matched['active_frac']:.3f} | {matched['ops_saved']:.1%} | +{baseline_delta:.2f}% |

TSA quality advantage over EarlyExit at matched α: **{tsa_delta:+.2f}%** val loss
(negative = TSA better, positive = EarlyExit better).

---

## Structural Analysis

### 1. Training compute

| Method | Train compute (vs Baseline) |
|---|---|
| Baseline | 1.00× |
| **EarlyExit** | **1.00× — identical (all layers always run)** |
| **TSA** | **{PHASE6_TSA_AF:.2f}× — {(1-PHASE6_TSA_AF)*100:.0f}% ops saved at train AND inference** |

Early Exit saves compute only at inference. TSA's soft gating acts during training too:
`h += (1 - halt_prob) × δ` means the gradient signal is weighted by the gate, which
acts as implicit regularisation and improves sample efficiency.

### 2. Routing mechanism

- **EarlyExit:** exits when `max_softmax(logits) > threshold`. This is a *post-hoc*
  confidence measure that ignores the token's representational needs. A syntactically
  complex token with certain next-word prediction (e.g., deterministic punctuation)
  will exit early regardless of whether deeper processing would improve other positions'
  predictions.

- **TSA:** the router is a learned 2-layer MLP that sees the full hidden state h.
  It can route based on contextual complexity, not just output confidence. The gate
  `halt_prob ∈ [0,1]` is differentiable and trained jointly with the model weights.

### 3. Attention scope

Both methods run dense attention over all token positions (all-token KV). Neither can
reduce attention FLOPs — only FFN and residual update costs differ. EarlyExit skips
subsequent FFN+attn *computation for that token's layer*, but still holds positions
in the residual stream for downstream KV correctness.

### 4. Training stability

EarlyExit training is equivalent to Baseline + auxiliary CE heads — straightforward
gradient flow. TSA introduces a learned gate that affects the gradient path but
remains stable across λ ∈ [0, 0.1] (confirmed in λ sweep ablation).

---

## Figure

Pareto curve: active_frac (x, inverted left=efficient) vs val_loss (y).
EarlyExit threshold sweep (blue), TSA Phase 6 (red star), Baseline (grey star).

![early_exit_comparison](../paper/figures/early_exit_comparison.png)

---

## Decisions Logged

- D049: Early exit ablation methodology — see docs/plans/decision_log.md
"""
    out_path = RESULTS_DIR / "ablation_early_exit.md"
    out_path.write_text(report)
    print(f"\n  Report written → {out_path}")


def plot_pareto(
    threshold_rows: list[dict],
    matched: dict,
) -> None:
    n_layers = EE_CFG.n_layers
    baseline_bpc = PHASE6_BASELINE_LOSS / math.log(2)
    tsa_bpc = PHASE6_TSA_LOSS / math.log(2)

    afs = [r["active_frac"] for r in threshold_rows]
    losses = [r["val_loss"] for r in threshold_rows]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Early exit Pareto curve
    ax.plot(afs, losses, "bo-", markersize=5, linewidth=1.5, label="Early Exit (threshold sweep)")
    ax.annotate(
        f"threshold={matched['threshold']:.2f}\n(α≈{PHASE6_TSA_AF:.3f})",
        xy=(matched["active_frac"], matched["val_loss"]),
        xytext=(matched["active_frac"] + 0.06, matched["val_loss"] + 0.005),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="blue", lw=0.8),
        color="blue",
    )

    # TSA Phase 6 reference point
    tsa_ops_saved = active_fraction_savings(PHASE6_TSA_AF, n_layers)
    ax.scatter(
        [PHASE6_TSA_AF], [PHASE6_TSA_LOSS],
        marker="*", s=200, color="red", zorder=5,
        label=f"TSA (α={PHASE6_TSA_AF:.3f})",
    )

    # Baseline reference point
    ax.scatter(
        [PHASE6_BASELINE_AF], [PHASE6_BASELINE_LOSS],
        marker="*", s=200, color="gray", zorder=5,
        label="Baseline",
    )

    ax.set_xlabel("Active Fraction (← more efficient)", fontsize=11)
    ax.set_ylabel("Val Loss (lower is better)", fontsize=11)
    ax.set_title("Early Exit vs TSA — Quality-Efficiency Pareto", fontsize=12)
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    for fmt in ("pdf", "png"):
        out = FIGURES_DIR / f"early_exit_comparison.{fmt}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"  Figure saved → {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()
    print("\n" + "="*65)
    print("  Early Exit Ablation — Phase 6 Shakespeare")
    print("="*65)
    print(f"  Phase 6 TSA reference: val_loss={PHASE6_TSA_LOSS:.4f}  af={PHASE6_TSA_AF:.3f}")
    print(f"  Phase 6 Baseline ref:  val_loss={PHASE6_BASELINE_LOSS:.4f}  af=1.000")

    # Load data
    print("\n  Loading Shakespeare …")
    train_data, val_data, _, tokenizer = load_shakespeare()
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Update vocab size to match loaded data
    cfg = EarlyExitConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=EE_CFG.d_model,
        n_heads=EE_CFG.n_heads,
        n_layers=EE_CFG.n_layers,
        d_ff=EE_CFG.d_ff,
        context_len=EE_CFG.context_len,
        dropout=EE_CFG.dropout,
    )

    # Train
    mx.random.seed(42)
    model = EarlyExitTransformer(cfg)
    train_result = train_early_exit(model, train_data, val_data, TRAIN_CFG)

    # Threshold sweep
    threshold_rows = sweep_thresholds(
        model, val_data,
        context_len=TRAIN_CFG.context_len,
        batch_size=TRAIN_CFG.batch_size,
        thresholds=THRESHOLDS,
        n_layers=cfg.n_layers,
        n_batches=30,
    )

    # Find threshold closest to TSA operating point
    matched = closest_to_af(threshold_rows, PHASE6_TSA_AF)

    print(f"\n  ── Matched TSA operating point (α≈{PHASE6_TSA_AF:.3f}) ─────────────────")
    print(f"  EarlyExit threshold={matched['threshold']:.2f}  val_loss={matched['val_loss']:.4f}"
          f"  af={matched['active_frac']:.3f}  ops_saved={matched['ops_saved']:.1%}")
    print(f"  TSA Phase 6          val_loss={PHASE6_TSA_LOSS:.4f}"
          f"  af={PHASE6_TSA_AF:.3f}  ops_saved={active_fraction_savings(PHASE6_TSA_AF, cfg.n_layers):.1%}")
    delta = (matched["val_loss"] - PHASE6_TSA_LOSS) / PHASE6_TSA_LOSS * 100
    print(f"  Delta (EE vs TSA):   {delta:+.2f}% val loss  ({'TSA better' if delta > 0 else 'EE better'})")

    # Plot + report
    plot_pareto(threshold_rows, matched)
    write_report(train_result, threshold_rows, matched)

    total_elapsed = time.time() - t_total
    print(f"\n  Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("\n  Done.")


if __name__ == "__main__":
    main()
