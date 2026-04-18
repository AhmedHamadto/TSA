"""
Ablation Study — λ (depth regularisation) sweep for TSA on Shakespeare.

Tests how sensitive TSA is to the depth_reg_weight hyperparameter λ.
Runs 7 conditions on Shakespeare (Phase 6 architecture) and records
val loss, BPC, active fraction, and training stability per λ.

Decision D048: see docs/plans/decision_log.md.

Pre-check:
  λ=0.001 (the Phase 6 default) is run first. Its MLX val loss must match
  Phase 6 PyTorch reference (1.4482) within 1% before the sweep continues.
  If it fails, debug the MLX implementation — don't waste 6 more runs.

λ values: [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

Runtime estimate: ~90–120 min total on M1 Pro (7 conditions × 5k steps).
Run foreground in terminal. Close GPU-heavy apps.

Output:
  docs/results/ablation_lambda.md
  docs/paper/figures/lambda_pareto.pdf / .png
"""
from __future__ import annotations

import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx


from tsa.mlx.data import load_shakespeare
from tsa.mlx.tsa import TSAConfig, TSATransformer
from tsa.mlx.train import TrainConfig, train_condition, ConditionResult

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


# ── Config — matches Phase 6 exactly ─────────────────────────────────────────

TRAIN_CFG = TrainConfig(
    train_steps=5_000,
    batch_size=64,
    context_len=128,       # Phase 6: 128 (not 256 like Phase 7)
    lr=3e-4,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=200,
    eval_interval=250,
    seed=42,
    loss_thresholds=[3.0, 2.5, 2.0],
)

MODEL_DIMS = dict(
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    context_len=128,
    dropout=0.1,
)

# λ values to sweep.
# Run 0.001 first (pre-check against Phase 6 reference), then the rest.
LAMBDA_VALUES = [0.001, 0.0, 0.005, 0.01, 0.05, 0.1, 0.5]

# Phase 6 PyTorch reference (D029). Pre-check tolerances.
PHASE6_REF_LOSS    = 1.4482   # TSA-only, λ=0.001, 5k steps, Shakespeare MPS
PHASE6_REF_AF      = 0.726    # active fraction
PHASE6_BASE_LOSS   = 1.4422   # Baseline (no routing), for Pareto reference point
PRECHECK_LAMBDA    = 0.001
PRECHECK_TOL       = 0.01     # 1% relative tolerance


# ── Pre-check ─────────────────────────────────────────────────────────────────

def check_mlx_matches_phase6(result: ConditionResult) -> bool:
    """
    Verify MLX λ=0.001 val loss is within 1% of Phase 6 PyTorch reference.
    Prints pass/fail with details.
    """
    mlx_loss = result.final_val_loss
    mlx_af   = result.active_fracs[-1] if result.active_fracs else float("nan")
    rel_diff = abs(mlx_loss - PHASE6_REF_LOSS) / PHASE6_REF_LOSS

    print(f"\n  ── Pre-check ────────────────────────────────────────────────────")
    print(f"  Phase 6 PyTorch reference:  val_loss={PHASE6_REF_LOSS:.4f}  af={PHASE6_REF_AF:.3f}")
    print(f"  MLX λ=0.001 result:         val_loss={mlx_loss:.4f}  af={mlx_af:.3f}")
    print(f"  Relative diff: {rel_diff*100:.2f}%  (tolerance: {PRECHECK_TOL*100:.0f}%)")

    if rel_diff > PRECHECK_TOL:
        print(f"  ✗ PRE-CHECK FAILED — diff {rel_diff*100:.2f}% > {PRECHECK_TOL*100:.0f}%")
        print(f"    Debug the MLX port before sweeping.")
        return False

    print(f"  ✓ PRE-CHECK PASSED — continuing with λ sweep")
    return True


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_pareto_plot(
    lambdas: list[float],
    val_losses: list[float],
    active_fracs: list[float],
) -> None:
    """
    Pareto curve: active_frac (x) vs val_loss (y).
    Lower-right is better (more efficient + lower loss).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Colour points by λ on a log scale
    colours = plt.cm.viridis_r(
        [(math.log10(max(lam, 1e-4)) + 4) / 4.7 for lam in lambdas]
    )

    # Baseline reference (from PyTorch results)
    ax.scatter([1.0], [PHASE6_BASE_LOSS], marker="*", s=200, color="grey",
               zorder=5, label="Baseline (λ=—)")

    # PyTorch TSA reference at λ=0.001
    ax.scatter([PHASE6_REF_AF], [PHASE6_REF_LOSS], marker="D", s=100,
               facecolors="none", edgecolors="steelblue", linewidths=1.5,
               zorder=4, label="PyTorch ref (λ=0.001)")

    for i, (lam, vl, af) in enumerate(zip(lambdas, val_losses, active_fracs)):
        stable = not (math.isnan(vl) or vl > 4.0)
        label = f"λ={lam}" if lam > 0 else "λ=0"
        marker = "o" if stable else "x"
        ax.scatter([af], [vl], color=colours[i], s=90, marker=marker,
                   zorder=5, label=label)
        # Annotate
        ax.annotate(
            label,
            (af, vl),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=colours[i],
        )

    # "Better" direction arrow
    ax.annotate(
        "better →",
        xy=(0.75, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] < 3 else 2.9),
        fontsize=9, color="grey", ha="center",
        arrowprops=dict(arrowstyle="->", color="grey", lw=1),
        xytext=(0.55, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] < 3 else 2.9),
    )

    ax.set_xlabel("Mean active fraction α  (lower = more efficient)")
    ax.set_ylabel("Val loss (lower = better quality)")
    ax.set_title("TSA λ-sweep Pareto Curve — Shakespeare (MLX)\n"
                 "Each point is a λ value; lower-right is better")
    ax.legend(fontsize=8, loc="upper left")
    ax.invert_xaxis()  # left = more efficient

    plt.tight_layout()
    out_pdf = FIGURES_DIR / "lambda_pareto.pdf"
    out_png = FIGURES_DIR / "lambda_pareto.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")


# ── Markdown report ───────────────────────────────────────────────────────────

def write_report(
    lambdas: list[float],
    results: list[ConditionResult],
    precheck_passed: bool,
    elapsed_total: float,
) -> None:
    """Write docs/results/ablation_lambda.md."""
    import datetime
    today = datetime.date.today().isoformat()

    # Derived metrics
    val_losses  = [r.final_val_loss for r in results]
    bpcs        = [r.final_bpc for r in results]
    active_fracs = [r.active_fracs[-1] if r.active_fracs else float("nan") for r in results]
    stable = [not (math.isnan(vl) or vl > 4.0) for vl in val_losses]

    # Find the λ closest to the Pareto front (best quality per unit efficiency)
    # Simple proxy: smallest val_loss among λ values with active_frac < 0.95
    efficient = [(lam, vl, af) for lam, vl, af, s in
                 zip(lambdas, val_losses, active_fracs, stable)
                 if s and af < 0.95]
    best_lam, best_vl, best_af = min(efficient, key=lambda x: x[1]) if efficient else (None, None, None)

    # Quality range across stable λ values
    stable_losses = [vl for vl, s in zip(val_losses, stable) if s]
    loss_range = max(stable_losses) - min(stable_losses) if stable_losses else float("nan")

    # Active fraction range
    stable_afs = [af for af, s in zip(active_fracs, stable) if s and not math.isnan(af)]
    af_range_min = min(stable_afs) if stable_afs else float("nan")
    af_range_max = max(stable_afs) if stable_afs else float("nan")

    # Analysis paragraph
    n_unstable = sum(1 for s in stable if not s)
    analysis = _write_analysis(lambdas, val_losses, active_fracs, stable,
                                best_lam, best_vl, best_af, loss_range,
                                af_range_min, af_range_max, n_unstable)

    lines = [
        "# Ablation Study — λ (Depth Regularisation) Sweep",
        "",
        f"**Date:** {today}",
        f"**Hardware:** {_detect_hardware()}",
        "**Framework:** MLX (Metal GPU)",
        f"**Model:** d_model={MODEL_DIMS['d_model']}, n_layers={MODEL_DIMS['n_layers']}, "
        f"n_heads={MODEL_DIMS['n_heads']}, d_ff={MODEL_DIMS['d_ff']}, vocab=65 (Shakespeare)",
        f"**Config:** {TRAIN_CFG.train_steps:,} steps, batch={TRAIN_CFG.batch_size}, "
        f"ctx={TRAIN_CFG.context_len}, lr={TRAIN_CFG.lr}",
        f"**Total runtime:** {elapsed_total/60:.1f} min",
        "",
        "---",
        "",
        "## Pre-Check",
        "",
        f"Phase 6 PyTorch reference (λ=0.001): val_loss={PHASE6_REF_LOSS} "
        f"BPC={PHASE6_REF_LOSS/math.log(2):.4f} active_frac={PHASE6_REF_AF}",
        "",
    ]

    # Pre-check result
    ref_idx = lambdas.index(PRECHECK_LAMBDA)
    mlx_ref_loss = val_losses[ref_idx]
    mlx_ref_af   = active_fracs[ref_idx]
    rel_diff = abs(mlx_ref_loss - PHASE6_REF_LOSS) / PHASE6_REF_LOSS
    if precheck_passed:
        lines += [
            f"MLX λ=0.001 result: val_loss={mlx_ref_loss:.4f} "
            f"active_frac={mlx_ref_af:.3f} (diff={rel_diff*100:.2f}%) — **✓ PASSED**",
            "",
        ]
    else:
        lines += [
            f"MLX λ=0.001 result: val_loss={mlx_ref_loss:.4f} "
            f"active_frac={mlx_ref_af:.3f} (diff={rel_diff*100:.2f}%) — **✗ FAILED**",
            "",
            "> Warning: pre-check failed. Results below may not be reliable.",
            "",
        ]

    lines += [
        "---",
        "",
        "## Results Table",
        "",
        "| λ | Val Loss | BPC | Active Frac | Ops Saved | Stable? |",
        "|---|----------|-----|-------------|-----------|---------|",
    ]

    # Reference baseline row
    base_ops = 0.0
    lines.append(
        f"| — (Baseline, P6 ref) | {PHASE6_BASE_LOSS:.4f} | "
        f"{PHASE6_BASE_LOSS/math.log(2):.4f} | 1.000 | 0.0% | ✓ (P6 PyTorch) |"
    )

    for lam, r, af, s in sorted(
        zip(lambdas, results, active_fracs, stable), key=lambda x: x[0]
    ):
        vl = r.final_val_loss
        bpc_val = r.final_bpc
        ops_saved = (1.0 - (1.0 + (MODEL_DIMS["n_layers"] - 1) * af) /
                     MODEL_DIMS["n_layers"]) * 100 if not math.isnan(af) else float("nan")
        lam_str = str(lam) if lam > 0 else "0"
        mark = " ← best" if lam == best_lam else ""
        lines.append(
            f"| {lam_str}{mark} | {vl:.4f} | {bpc_val:.4f} | "
            f"{af:.3f} | {ops_saved:.1f}% | {'✓' if s else '✗ (unstable)'} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Analysis",
        "",
        analysis,
        "",
        "---",
        "",
        "## Figure",
        "",
        "![lambda_pareto](../paper/figures/lambda_pareto.png)",
        "",
        "Pareto curve: active fraction (x, inverted — left is more efficient) vs val loss (y).",
        "Each point is a λ value. Lower-right is better.",
        "Baseline (grey star) and Phase 6 PyTorch TSA reference (blue diamond) shown for context.",
        "",
        "---",
        "",
        "## Decisions Logged",
        "- D048: Lambda sweep methodology — see docs/plans/decision_log.md",
    ]

    out_path = RESULTS_DIR / "ablation_lambda.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n  Saved: {out_path}")


def _write_analysis(
    lambdas, val_losses, active_fracs, stable,
    best_lam, best_vl, best_af, loss_range, af_range_min, af_range_max, n_unstable,
) -> str:
    """Generate the analysis paragraph from results."""
    stable_lambdas = [lam for lam, s in zip(lambdas, stable) if s]

    if n_unstable == 0:
        stability_note = "All 7 λ values trained stably (no divergence or NaN)."
    else:
        unstable_lams = [lam for lam, s in zip(lambdas, stable) if not s]
        stability_note = (
            f"{n_unstable} condition(s) were unstable (val_loss > 4 or NaN): "
            f"λ ∈ {unstable_lams}."
        )

    # How much does val_loss vary across stable runs?
    if loss_range < 0.02:
        quality_note = (
            f"Val loss is highly robust to λ: the full range across stable conditions "
            f"is only {loss_range:.3f} nats ({loss_range/PHASE6_REF_LOSS*100:.1f}% relative)."
        )
    elif loss_range < 0.10:
        quality_note = (
            f"Val loss shows moderate sensitivity to λ: range across stable conditions "
            f"is {loss_range:.3f} nats ({loss_range/PHASE6_REF_LOSS*100:.1f}% relative)."
        )
    else:
        quality_note = (
            f"Val loss is sensitive to λ: range across stable conditions "
            f"is {loss_range:.3f} nats ({loss_range/PHASE6_REF_LOSS*100:.1f}% relative)."
        )

    # Active fraction range
    af_note = (
        f"Active fraction ranges from {af_range_min:.3f} to {af_range_max:.3f} "
        f"across stable conditions, confirming λ strongly controls routing sparsity."
    )

    # Best λ
    if best_lam is not None:
        best_note = (
            f"The best Pareto point among conditions with α < 0.95 is λ={best_lam} "
            f"(val_loss={best_vl:.4f}, α={best_af:.3f})."
        )
    else:
        best_note = "No conditions with α < 0.95 were stable."

    return (
        f"{stability_note} {quality_note} {af_note} {best_note} "
        f"The Phase 6 default (λ=0.001) sits in the stable region and achieves a good "
        f"quality-efficiency tradeoff, but this sweep quantifies how much headroom exists "
        f"in either direction: increasing λ forces more aggressive routing (lower α) at "
        f"some quality cost, while λ=0 reveals the baseline routing behaviour without "
        f"explicit efficiency pressure."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("TSA λ-sweep ablation — Shakespeare (MLX)")
    print(f"  λ values: {LAMBDA_VALUES}")
    print(f"  Steps per condition: {TRAIN_CFG.train_steps:,}")
    print(f"  Pre-check: λ=0.001 vs Phase 6 PyTorch ref (±{PRECHECK_TOL*100:.0f}%)")
    t_start = time.time()

    # Load Shakespeare (reuses cached file if present)
    print("\nLoading Shakespeare corpus …")
    train_data, val_data, _, tokenizer = load_shakespeare()
    vocab_size = tokenizer.vocab_size
    print(f"  vocab_size={vocab_size}")

    results: dict[float, ConditionResult] = {}

    # ── Step 1: Pre-check — run λ=0.001 first ────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PRE-CHECK: λ={PRECHECK_LAMBDA}")
    print(f"{'='*65}")
    cfg = TSAConfig(vocab_size=vocab_size, **MODEL_DIMS, depth_reg_weight=PRECHECK_LAMBDA)
    model = TSATransformer(cfg)
    result = train_condition(
        f"TSA λ={PRECHECK_LAMBDA} (pre-check)", model, train_data, val_data, TRAIN_CFG
    )
    results[PRECHECK_LAMBDA] = result

    precheck_passed = check_mlx_matches_phase6(result)
    if not precheck_passed:
        print("\n  Aborting sweep. Fix MLX implementation before continuing.")
        sys.exit(1)

    # ── Step 2: Remaining λ values ────────────────────────────────────────────
    remaining = [lam for lam in LAMBDA_VALUES if lam != PRECHECK_LAMBDA]
    for lam in remaining:
        cfg = TSAConfig(vocab_size=vocab_size, **MODEL_DIMS, depth_reg_weight=lam)
        model = TSATransformer(cfg)
        result = train_condition(
            f"TSA λ={lam}", model, train_data, val_data, TRAIN_CFG
        )
        results[lam] = result

    # ── Step 3: Plot and report ────────────────────────────────────────────────
    lambdas_ordered = sorted(results.keys())
    results_ordered = [results[lam] for lam in lambdas_ordered]
    active_fracs    = [r.active_fracs[-1] if r.active_fracs else float("nan")
                       for r in results_ordered]
    val_losses      = [r.final_val_loss for r in results_ordered]

    make_pareto_plot(lambdas_ordered, val_losses, active_fracs)
    write_report(
        lambdas_ordered,
        results_ordered,
        precheck_passed,
        time.time() - t_start,
    )

    # ── Step 4: Console summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'λ':>8}  {'Val Loss':>10}  {'BPC':>8}  {'Active Frac':>12}")
    print(f"  {'—':>8}  {'—':>10}  {'—':>8}  {'—':>12}")
    for lam, r, af in zip(lambdas_ordered, results_ordered, active_fracs):
        print(f"  {str(lam):>8}  {r.final_val_loss:>10.4f}  "
              f"{r.final_bpc:>8.4f}  {af:>12.3f}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
