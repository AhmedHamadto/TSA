"""
Phase 7: Character-level language modeling on enwik8 — MLX implementation.

Validates TSA compute savings on enwik8 (first 10^8 bytes of English Wikipedia)
using native MLX on Apple Silicon. This is the second dataset validation for the
TSA paper — checking that the routing mechanism generalises beyond Shakespeare.

2 conditions (SGC excluded — Phase 6 showed it does not transfer to language):
  1. Baseline Transformer — standard fixed-depth
  2. TSA-only             — same architecture + learned token routing

Architecture: same as Phase 6 Shakespeare (d_model=256, n_layers=6, n_heads=8,
d_ff=1024) with context_len=256 (2× Phase 6) to handle longer Wikipedia sentences.

Decision D045: 5,000 training steps (not 10,000 from the PyTorch Phase 7 plan).
  At 600ms/step on M1 Pro, 10,000 steps would be ~100 min/condition = 200 min total.
  5,000 steps still covers ~82% of the 90M training characters in a single pass and
  gives enough convergence to measure TSA routing behaviour. Step budget matches
  Phase 6 exactly for cleaner comparison.
  See also D033 (10K was PyTorch plan; MLX enables faster iteration).

All other hyperparameters match Phase 6:
  d_model=256, n_layers=6, n_heads=8, d_ff=1024, context_len=256
  lr=3e-4, batch_size=64, dropout=0.1, depth_reg_weight=0.001
  optimizer=AdamW(β=(0.9,0.95), wd=0.1), cosine LR + warmup
  grad_clip=1.0, seed=42

Hardware: Apple MacBook Pro, M1 Pro, 16 GB unified memory.
"""
from __future__ import annotations

import math
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx


from tsa.mlx.data import load_enwik8
from tsa.mlx.model import BaselineConfig, BaselineTransformer
from tsa.mlx.tsa import TSAConfig, TSATransformer
from tsa.mlx.train import TrainConfig, train_condition, ConditionResult
from tsa.mlx.evaluate import active_fraction_savings

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


RESULTS_DIR = Path(__file__).parent.parent / "docs" / "results" / "phase7"
FIGURES_DIR = Path(__file__).parent.parent / "docs" / "paper" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Experiment config ─────────────────────────────────────────────────────────

TRAIN_CFG = TrainConfig(
    train_steps=5_000,
    batch_size=64,
    context_len=256,    # D034: 2× Phase 6's 128
    lr=3e-4,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=200,
    eval_interval=250,  # every 5% of training, same ratio as Phase 6
    seed=42,
    loss_thresholds=[2.5, 2.0, 1.8],  # D036: enwik8 difficulty range
)

MODEL_DIMS = dict(
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    context_len=256,
    dropout=0.1,
)
DEPTH_REG = 0.001

HARDWARE = _detect_hardware()


# ── Model factories ───────────────────────────────────────────────────────────

def make_baseline(vocab_size: int) -> BaselineTransformer:
    cfg = BaselineConfig(vocab_size=vocab_size, **MODEL_DIMS)
    return BaselineTransformer(cfg)


def make_tsa(vocab_size: int) -> TSATransformer:
    cfg = TSAConfig(vocab_size=vocab_size, depth_reg_weight=DEPTH_REG, **MODEL_DIMS)
    return TSATransformer(cfg)


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {"Baseline": "#4C72B0", "TSA": "#DD8452"}
LINESTYLES = {"Baseline": "-", "TSA": "--"}


def _plot_val_loss_vs_steps(results: list[ConditionResult], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        key = "Baseline" if "Baseline" in r.label else "TSA"
        ax.plot(
            r.eval_steps, r.val_losses,
            label=f"{r.label}  (final={r.final_val_loss:.3f})",
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=2,
        )
    ax.set_xlabel("Training steps", fontsize=12)
    ax.set_ylabel("Validation loss (nats)", fontsize=12)
    ax.set_title("Phase 7 — enwik8 char-level LM: Val Loss vs Steps", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        p = out_path.with_suffix(ext)
        fig.savefig(p, dpi=150)
        print(f"  Saved: {p}")
    plt.close(fig)


def _plot_val_loss_vs_compute(results: list[ConditionResult], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        key = "Baseline" if "Baseline" in r.label else "TSA"
        ops_billions = [o / 1e9 for o in r.token_layer_ops]
        ax.plot(
            ops_billions, r.val_losses,
            label=r.label,
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=2,
        )
    ax.set_xlabel("Cumulative token-layer ops (billions)", fontsize=12)
    ax.set_ylabel("Validation loss (nats)", fontsize=12)
    ax.set_title("Phase 7 — enwik8 char-level LM: Val Loss vs Compute", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        p = out_path.with_suffix(ext)
        fig.savefig(p, dpi=150)
        print(f"  Saved: {p}")
    plt.close(fig)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list[ConditionResult]) -> dict:
    sep = "=" * 75
    print(f"\n\n{sep}")
    print("  PHASE 7 RESULTS — enwik8 Character-Level LM (MLX, M1 Pro)")
    print(sep)
    n = MODEL_DIMS
    print(f"\n  Model: d={n['d_model']}, L={n['n_layers']}, h={n['n_heads']}, "
          f"d_ff={n['d_ff']}, ctx={n['context_len']}")
    print(f"  Steps: {TRAIN_CFG.train_steps:,}  Batch: {TRAIN_CFG.batch_size}  "
          f"LR: {TRAIN_CFG.lr}")

    baseline = next((r for r in results if "Baseline" in r.label), None)
    tsa = next((r for r in results if "TSA" in r.label), None)

    print(f"\n  {'Condition':<32}  {'Params':>8}  {'Val Loss':>9}  {'BPC':>7}", end="")
    for t in TRAIN_CFG.loss_thresholds:
        print(f"  {'→'+str(t):>7}", end="")
    print()
    print(f"  {'-'*32}  {'-'*8}  {'-'*9}  {'-'*7}", end="")
    for _ in TRAIN_CFG.loss_thresholds:
        print(f"  {'-'*7}", end="")
    print()

    for r in results:
        print(f"  {r.label:<32}  {r.n_params:>8,}  {r.final_val_loss:>9.4f}  "
              f"{r.final_bpc:>7.4f}", end="")
        for t in TRAIN_CFG.loss_thresholds:
            s = r.steps_to_threshold.get(t)
            print(f"  {str(s) if s else 'MISS':>7}", end="")
        print()

    print("\n  COMPUTE EFFICIENCY (TSA):")
    verdict = "UNKNOWN"
    tsa_saves = False
    if tsa and tsa.active_fracs:
        mean_af = sum(tsa.active_fracs) / len(tsa.active_fracs)
        ops_saved_pct = active_fraction_savings(mean_af, MODEL_DIMS["n_layers"]) * 100
        print(f"    Mean active fraction: {mean_af:.4f}")
        print(f"    Token-layer ops saved: {ops_saved_pct:.1f}%  "
              f"(formula: Δ = 1 − (1 + (L−1)α)/L)")
        # Compare to Phase 6 Shakespeare (0.726 active → 22.8% ops saved)
        print(f"    Phase 6 Shakespeare comparison: 0.726 active → 22.8% ops saved")
        tsa_saves = ops_saved_pct >= 15.0

    if baseline and tsa:
        loss_diff_pct = (tsa.final_val_loss - baseline.final_val_loss) / baseline.final_val_loss * 100
        print(f"\n  QUALITY (TSA vs Baseline):")
        print(f"    Val loss diff: {loss_diff_pct:+.2f}%  "
              f"(Baseline: {baseline.final_val_loss:.4f}  TSA: {tsa.final_val_loss:.4f})")
        quality_ok = abs(loss_diff_pct) <= 1.0

        if tsa_saves and quality_ok:
            verdict = "✓ TSA GENERALISES TO ENWIK8 — compute savings at comparable quality"
        elif tsa_saves:
            verdict = "~ PARTIAL — TSA saves compute but quality gap > 1%"
        elif quality_ok:
            verdict = "~ ROUTING ACTIVE but savings below 15% threshold"
        else:
            verdict = "✗ TSA did not generalise as expected"

    print(f"\n  VERDICT: {verdict}")
    print(f"\n  WALL TIME:")
    for r in results:
        ms_per_step = r.elapsed / TRAIN_CFG.train_steps * 1000
        print(f"    {r.label:<32}: {r.elapsed:.0f}s  ({ms_per_step:.0f}ms/step)")
    print(f"\n{sep}\n")

    return {"verdict": verdict, "tsa_saves": tsa_saves}


# ── Results saving ─────────────────────────────────────────────────────────────

def save_results_md(
    results: list[ConditionResult],
    vocab_size: int,
    total_elapsed: float,
    verdict_dict: dict,
) -> None:
    baseline = next((r for r in results if "Baseline" in r.label), None)
    tsa = next((r for r in results if "TSA" in r.label), None)

    mean_af = ""
    ops_saved = ""
    if tsa and tsa.active_fracs:
        af = sum(tsa.active_fracs) / len(tsa.active_fracs)
        ops_pct = active_fraction_savings(af, MODEL_DIMS["n_layers"]) * 100
        mean_af = f"{af:.4f}"
        ops_saved = f"{ops_pct:.1f}%"

    lines = [
        "# Phase 7 Results — enwik8 Character-Level LM (MLX)",
        "",
        f"**Date:** 2026-04-01",
        f"**Hardware:** {HARDWARE}",
        f"**MLX version:** 0.31.1",
        f"**Corpus:** enwik8 (first 10^8 bytes of English Wikipedia, raw XML)",
        f"**Vocab size:** {vocab_size}",
        f"**Scale:** {results[0].n_params:,} params (d={MODEL_DIMS['d_model']}, "
        f"L={MODEL_DIMS['n_layers']}, h={MODEL_DIMS['n_heads']}, "
        f"ctx={MODEL_DIMS['context_len']})",
        f"**Steps:** {TRAIN_CFG.train_steps:,} per condition",
        f"**Total wall time:** {total_elapsed/60:.1f} min",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Condition | Params | Val Loss | BPC | →2.5 | →2.0 | →1.8 | "
        "Active Frac | ms/step | Time |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        af_str = (f"{sum(r.active_fracs)/len(r.active_fracs):.3f}"
                  if r.active_fracs else "1.000")
        ms_s = f"{r.elapsed / TRAIN_CFG.train_steps * 1000:.0f}"
        t_str = f"{r.elapsed/60:.1f}m"
        thresholds = [str(r.steps_to_threshold.get(t, "MISS")) for t in TRAIN_CFG.loss_thresholds]
        lines.append(
            f"| {r.label} | {r.n_params:,} | **{r.final_val_loss:.4f}** | "
            f"**{r.final_bpc:.4f}** | {' | '.join(thresholds)} | "
            f"{af_str} | {ms_s} | {t_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### Finding 1: TSA on enwik8",
    ]

    if tsa and tsa.active_fracs:
        af = sum(tsa.active_fracs) / len(tsa.active_fracs)
        ops_pct = active_fraction_savings(af, MODEL_DIMS["n_layers"]) * 100
        loss_diff = ((tsa.final_val_loss - baseline.final_val_loss) / baseline.final_val_loss * 100
                     if baseline else float("nan"))
        lines += [
            f"TSA mean active fraction = {af:.4f} → "
            f"**{ops_pct:.1f}% of token-layer computations skipped**.",
            f"Quality: val loss diff = {loss_diff:+.2f}% vs Baseline.",
            "",
            "**Phase 6 Shakespeare comparison:**",
            "- Shakespeare: 0.726 active → 22.8% ops saved, +0.4% quality loss",
            f"- enwik8: {af:.3f} active → {ops_pct:.1f}% ops saved, {loss_diff:+.1f}% quality diff",
        ]

    lines += [
        "",
        "### Finding 2: Cross-dataset generalisation",
        "",
        verdict_dict.get("verdict", ""),
        "",
        "---",
        "",
        "## Compute Efficiency Detail",
        "",
        "```",
        f"Formula: Δ = 1 − (1 + (L−1) × α) / L  (all-layers, includes mandatory stem)",
        f"Where α = mean active fraction, L = {MODEL_DIMS['n_layers']} layers",
        "",
    ]

    for r in results:
        if r.active_fracs:
            af = sum(r.active_fracs) / len(r.active_fracs)
            eff = 1 + (MODEL_DIMS["n_layers"] - 1) * af
            delta = 1 - eff / MODEL_DIMS["n_layers"]
        else:
            eff = MODEL_DIMS["n_layers"]
            delta = 0.0
        lines.append(
            f"  {r.label:<32}  α={af if r.active_fracs else 1.000:.3f}  "
            f"eff_layers={eff:.2f}  ops_saved={delta*100:.1f}%"
        )

    lines += [
        "```",
        "",
        "---",
        "",
        "## Notes on enwik8 vs Shakespeare",
        "",
        "- Vocab size: enwik8 has Unicode chars from non-English articles → "
        "larger vocab than Shakespeare's 65",
        "- Context length: 256 (vs 128 for Shakespeare) to handle longer Wikipedia sentences",
        "- Training: 5,000 steps × batch 64 × context 256 = 81.9M tokens seen "
        "(covers ~91% of 90M training chars once)",
        "- Raw XML kept (Decision D032): standard enwik8 benchmark, "
        "stripping would invalidate BPC comparison",
        "",
        "## Decisions Logged",
        "",
        "- D032: Raw XML kept (standard enwik8)",
        "- D033: Originally 10K steps for PyTorch; MLX Phase 7 uses 5K (D045)",
        "- D034: context_len=256",
        "- D035: eval_interval=250 (5% frequency)",
        "- D036: loss_thresholds=[2.5, 2.0, 1.8]",
        "- D045: Use MLX for Phase 7; 5K steps based on 600ms/step timing on M1 Pro",
        "",
        "## Plots",
        "",
        "- `docs/paper/figures/phase7_val_loss_vs_steps.pdf` — val loss vs training step",
        "- `docs/paper/figures/phase7_val_loss_vs_compute.pdf` — val loss vs cumulative token-layer ops",
    ]

    out = RESULTS_DIR / "phase7_enwik8_mlx.md"
    out.write_text("\n".join(lines))
    print(f"  Results saved: {out}")

    # Also save numeric results.txt
    nums_out = RESULTS_DIR / "results_mlx.txt"
    num_lines = [f"Phase 7 enwik8 MLX — {HARDWARE}"]
    for r in results:
        ms_s = r.elapsed / TRAIN_CFG.train_steps * 1000
        num_lines.append(f"\n{r.label}")
        num_lines.append(f"  params={r.n_params:,}  val_loss={r.final_val_loss:.4f}  "
                         f"bpc={r.final_bpc:.4f}  time={r.elapsed:.0f}s  "
                         f"ms_per_step={ms_s:.0f}")
        if r.active_fracs:
            af = sum(r.active_fracs) / len(r.active_fracs)
            num_lines.append(f"  mean_active_frac={af:.4f}  "
                             f"ops_saved={active_fraction_savings(af, MODEL_DIMS['n_layers'])*100:.1f}%")
        for t, s in r.steps_to_threshold.items():
            num_lines.append(f"  steps_to_{t}={s if s is not None else 'MISS'}")
    nums_out.write_text("\n".join(num_lines))
    print(f"  Numeric results saved: {nums_out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'='*75}")
    print("  PHASE 7 — enwik8 Character-Level Language Modeling (MLX)")
    print(f"{'='*75}")
    print(f"  Hardware: {HARDWARE}")
    print(f"  MLX device: {mx.default_device()}")
    print(f"  Steps: {TRAIN_CFG.train_steps:,}  Batch: {TRAIN_CFG.batch_size}  "
          f"Context: {TRAIN_CFG.context_len}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading enwik8 …")
    t_load = time.time()
    train_data, val_data, test_data, tokenizer = load_enwik8()
    vocab_size = tokenizer.vocab_size
    print(f"  Load time: {time.time() - t_load:.1f}s  |  Vocab: {vocab_size}")

    total_t0 = time.time()
    results: list[ConditionResult] = []

    # ── Condition 1: Baseline ──────────────────────────────────────────────
    mx.random.seed(TRAIN_CFG.seed)
    baseline_model = make_baseline(vocab_size)
    r_baseline = train_condition(
        "Baseline",
        baseline_model,
        train_data,
        val_data,
        TRAIN_CFG,
    )
    results.append(r_baseline)

    # ── Condition 2: TSA ───────────────────────────────────────────────────
    mx.random.seed(TRAIN_CFG.seed)
    tsa_model = make_tsa(vocab_size)
    r_tsa = train_condition(
        "TSA",
        tsa_model,
        train_data,
        val_data,
        TRAIN_CFG,
    )
    results.append(r_tsa)

    total_elapsed = time.time() - total_t0

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    _plot_val_loss_vs_steps(
        results, FIGURES_DIR / "phase7_val_loss_vs_steps"
    )
    _plot_val_loss_vs_compute(
        results, FIGURES_DIR / "phase7_val_loss_vs_compute"
    )

    # ── Summary + save ────────────────────────────────────────────────────
    verdict_dict = print_summary(results)
    save_results_md(results, vocab_size, total_elapsed, verdict_dict)

    print(f"\nTotal elapsed: {total_elapsed/60:.1f} min")
    print("Phase 7 complete.\n")


if __name__ == "__main__":
    main()
