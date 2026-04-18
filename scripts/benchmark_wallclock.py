"""
Wall-clock inference benchmark: Baseline vs soft-TSA vs sparse-TSA on Metal GPU.

Measures whether TSA's token-layer-op savings translate to actual throughput
gains on Apple Silicon. Three conditions:
  Baseline    — standard dense transformer (reference)
  Soft-TSA    — continuous (1-halt_prob) gate, all tokens fully processed
  Sparse-TSA  — hard threshold, FFN skipped for halted tokens via gather/scatter

Router output is controlled synthetically (FixedRouter / BimodalRouter), so
no training is required. This isolates the hardware throughput question from
the optimisation dynamics question.

Decision D046: synthetic routers for benchmark.
Decision D047: sparse FFN only (dense attention) — see sparse_infer.py.

Run from terminal as a foreground process. Close GPU-heavy apps (Safari, Slack,
video players) before running. Expected runtime: 10–15 minutes on M1 Pro.

Output:
  docs/results/ablation_wallclock.md   — full result tables + paper sentence
  docs/paper/figures/wallclock_speedup.pdf / .png
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
import numpy as np
import mlx.core as mx


from tsa.mlx.model import BaselineConfig, BaselineTransformer
from tsa.mlx.tsa import TSAConfig, TSATransformer
from tsa.mlx.sparse_infer import make_soft_tsa, make_sparse_tsa
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


RESULTS_DIR = Path(__file__).parent.parent / "docs" / "results"
FIGURES_DIR = Path(__file__).parent.parent / "docs" / "paper" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Benchmark hyperparameters ─────────────────────────────────────────────────

N_WARMUP = 30   # Metal JIT compiles on warmup steps
N_TIMED  = 200  # timed forward passes per configuration

# Primary config matches Phase 7 training
PRIMARY_BATCH  = 64
PRIMARY_SEQLEN = 256

# Phase-observed active fractions (for annotation)
PHASE6_AF = 0.726  # Shakespeare
PHASE7_AF = 0.833  # enwik8

# Active fraction sweep
ACTIVE_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.726, 0.8, 0.833, 0.9, 1.0]

# Secondary: batch size scaling at Phase 7 active fraction
BATCH_SIZES = [1, 8, 32, 64, 128]

# Model config — matches Phase 7
VOCAB_SIZE = 6064
MODEL_DIMS = dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024,
                  context_len=PRIMARY_SEQLEN, dropout=0.0)
# dropout=0 for inference — avoids stochastic noise in timing


# ── Benchmark primitive ───────────────────────────────────────────────────────

def benchmark_forward(
    model: BaselineTransformer | TSATransformer,
    batch_size: int,
    seq_len: int,
    n_warmup: int = N_WARMUP,
    n_timed: int = N_TIMED,
) -> tuple[float, float, float]:
    """
    Time model forward pass on random token IDs.

    Returns:
        mean_ms   — mean ms per forward pass
        std_ms    — std dev ms per forward pass
        tok_per_s — tokens per second (batch_size * seq_len / mean_ms * 1000)

    MLX timing notes:
      - mx.eval(x) before timing: pre-materialise input so it doesn't count
      - mx.eval(logits) after model(x): force GPU execution to complete
      - time.perf_counter() wraps both model(x) + mx.eval(logits)
      - Warmup ensures Metal JIT compilation is done before measurements
    """
    x = mx.random.randint(0, VOCAB_SIZE, shape=(batch_size, seq_len), dtype=mx.int32)
    mx.eval(x)

    # Warmup — Metal compiles shaders on first runs
    for _ in range(n_warmup):
        logits = model(x)
        mx.eval(logits)

    # Timed loop
    times_ms: list[float] = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        logits = model(x)
        mx.eval(logits)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    std_ms  = float(np.std(times_ms))
    tok_per_s = (batch_size * seq_len) / (mean_ms / 1000.0)
    return mean_ms, std_ms, tok_per_s


def make_baseline() -> BaselineTransformer:
    cfg = BaselineConfig(vocab_size=VOCAB_SIZE, **MODEL_DIMS)
    m = BaselineTransformer(cfg)
    m.eval()
    return m


def tsa_config() -> TSAConfig:
    return TSAConfig(vocab_size=VOCAB_SIZE, **MODEL_DIMS, depth_reg_weight=0.001)


# ── Phase 1: active_frac sweep at primary config ──────────────────────────────

def run_active_frac_sweep() -> dict:
    """
    Sweep active_frac ∈ ACTIVE_FRACS at (batch=PRIMARY_BATCH, seq=PRIMARY_SEQLEN).
    Measures: Baseline, soft-TSA, sparse-TSA.
    Returns dict of results.
    """
    print("\n" + "="*65)
    print("  Phase 1: Active-fraction sweep")
    print(f"  batch={PRIMARY_BATCH}  seq={PRIMARY_SEQLEN}  "
          f"warmup={N_WARMUP}  timed={N_TIMED}")
    print("="*65)

    results: dict[str, list] = {
        "active_frac": [],
        "ops_saved_pct": [],
        "baseline_ms": [],
        "soft_ms": [],
        "soft_std": [],
        "soft_speedup": [],
        "sparse_ms": [],
        "sparse_std": [],
        "sparse_speedup": [],
    }

    # Baseline (single measurement, reused for all speedup ratios)
    print("\n  [Baseline] ...", end="", flush=True)
    b_ms, b_std, b_tps = benchmark_forward(make_baseline(), PRIMARY_BATCH, PRIMARY_SEQLEN)
    print(f" {b_ms:.1f} ± {b_std:.1f} ms  ({b_tps/1000:.1f}k tok/s)")
    baseline_ms = b_ms

    cfg = tsa_config()

    for af in ACTIVE_FRACS:
        ops_saved = active_fraction_savings(af, MODEL_DIMS["n_layers"]) * 100
        halt_prob = 1.0 - af

        print(f"\n  active_frac={af:.3f}  ops_saved={ops_saved:.1f}%  halt_prob={halt_prob:.3f}")

        # Soft-TSA (FixedRouter — continuous gate, all tokens processed)
        print(f"    [Soft-TSA]   ...", end="", flush=True)
        soft_model = make_soft_tsa(cfg, af)
        s_ms, s_std, s_tps = benchmark_forward(soft_model, PRIMARY_BATCH, PRIMARY_SEQLEN)
        soft_speedup = baseline_ms / s_ms
        print(f" {s_ms:.1f} ± {s_std:.1f} ms  speedup={soft_speedup:.3f}x  ({s_tps/1000:.1f}k tok/s)")
        del soft_model

        # Sparse-TSA (BimodalRouter + GatherSparseBlocks)
        print(f"    [Sparse-TSA] ...", end="", flush=True)
        sparse_model = make_sparse_tsa(cfg, af, threshold=0.5)
        p_ms, p_std, p_tps = benchmark_forward(sparse_model, PRIMARY_BATCH, PRIMARY_SEQLEN)
        sparse_speedup = baseline_ms / p_ms
        print(f" {p_ms:.1f} ± {p_std:.1f} ms  speedup={sparse_speedup:.3f}x  ({p_tps/1000:.1f}k tok/s)")
        del sparse_model

        results["active_frac"].append(af)
        results["ops_saved_pct"].append(ops_saved)
        results["baseline_ms"].append(baseline_ms)
        results["soft_ms"].append(s_ms)
        results["soft_std"].append(s_std)
        results["soft_speedup"].append(soft_speedup)
        results["sparse_ms"].append(p_ms)
        results["sparse_std"].append(p_std)
        results["sparse_speedup"].append(sparse_speedup)

    return results, baseline_ms, b_std, b_tps


# ── Phase 2: batch-size scaling at Phase 7 active fraction ───────────────────

def run_batch_scaling(target_af: float = PHASE7_AF) -> dict:
    """
    Sweep batch sizes ∈ BATCH_SIZES at active_frac=target_af.
    Shows how sparse-TSA speedup scales with batch size.
    """
    print("\n" + "="*65)
    print(f"  Phase 2: Batch-size scaling  (active_frac={target_af})")
    print(f"  seq={PRIMARY_SEQLEN}  warmup={N_WARMUP}  timed={N_TIMED}")
    print("="*65)

    results: dict[str, list] = {
        "batch_size": [],
        "baseline_ms": [],
        "soft_ms": [],
        "soft_speedup": [],
        "sparse_ms": [],
        "sparse_speedup": [],
        "baseline_tps": [],
        "soft_tps": [],
        "sparse_tps": [],
    }

    cfg = tsa_config()

    for bs in BATCH_SIZES:
        print(f"\n  batch_size={bs}")

        print(f"    [Baseline]   ...", end="", flush=True)
        b_ms, b_std, b_tps = benchmark_forward(make_baseline(), bs, PRIMARY_SEQLEN)
        print(f" {b_ms:.1f} ± {b_std:.1f} ms  ({b_tps/1000:.1f}k tok/s)")

        print(f"    [Soft-TSA]   ...", end="", flush=True)
        soft_model = make_soft_tsa(cfg, target_af)
        s_ms, s_std, s_tps = benchmark_forward(soft_model, bs, PRIMARY_SEQLEN)
        soft_speedup = b_ms / s_ms
        print(f" {s_ms:.1f} ± {s_std:.1f} ms  speedup={soft_speedup:.3f}x")
        del soft_model

        print(f"    [Sparse-TSA] ...", end="", flush=True)
        sparse_model = make_sparse_tsa(cfg, target_af, threshold=0.5)
        p_ms, p_std, p_tps = benchmark_forward(sparse_model, bs, PRIMARY_SEQLEN)
        sparse_speedup = b_ms / p_ms
        print(f" {p_ms:.1f} ± {p_std:.1f} ms  speedup={sparse_speedup:.3f}x")
        del sparse_model

        results["batch_size"].append(bs)
        results["baseline_ms"].append(b_ms)
        results["soft_ms"].append(s_ms)
        results["soft_speedup"].append(soft_speedup)
        results["sparse_ms"].append(p_ms)
        results["sparse_speedup"].append(sparse_speedup)
        results["baseline_tps"].append(b_tps)
        results["soft_tps"].append(s_tps)
        results["sparse_tps"].append(p_tps)

    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(sweep_results: dict, batch_results: dict) -> None:
    """
    Two-panel figure: break-even curve + batch scaling.
    Saved as PDF and PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("TSA Wall-Clock Efficiency on Apple Silicon (M1 Pro)", fontsize=13)

    # ── Panel A: break-even curve ─────────────────────────────────────────────
    ax = axes[0]
    afs = sweep_results["active_frac"]
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.8, label="Break-even (1×)")
    ax.plot(afs, sweep_results["soft_speedup"],
            "o-", color="#2196F3", label="Soft-TSA (continuous gate)")
    ax.plot(afs, sweep_results["sparse_speedup"],
            "s-", color="#FF5722", label="Sparse-TSA (gather/scatter FFN)")

    # Annotate observed active fractions
    for phase, phase_af, color in [("Phase 6\nShakespeare", PHASE6_AF, "#9C27B0"),
                                    ("Phase 7\nenwik8", PHASE7_AF, "#009688")]:
        ax.axvline(x=phase_af, color=color, linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(phase_af + 0.005, ax.get_ylim()[0] + 0.02, phase,
                color=color, fontsize=8, va="bottom")

    ax.set_xlabel("Mean active fraction α")
    ax.set_ylabel("Speedup ratio vs Baseline")
    ax.set_title(f"Break-even curve  (batch={PRIMARY_BATCH}, seq={PRIMARY_SEQLEN})")
    ax.legend(fontsize=8)
    ax.set_xlim(0.05, 1.05)

    # ── Panel B: batch scaling at Phase 7 active fraction ────────────────────
    ax2 = axes[1]
    bs_arr = batch_results["batch_size"]
    ax2.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.8, label="Break-even (1×)")
    ax2.plot(bs_arr, batch_results["soft_speedup"],
             "o-", color="#2196F3", label=f"Soft-TSA (α={PHASE7_AF})")
    ax2.plot(bs_arr, batch_results["sparse_speedup"],
             "s-", color="#FF5722", label=f"Sparse-TSA (α={PHASE7_AF})")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(bs_arr)
    ax2.set_xticklabels([str(b) for b in bs_arr])
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Speedup ratio vs Baseline")
    ax2.set_title(f"Batch-size scaling  (α={PHASE7_AF}, seq={PRIMARY_SEQLEN})")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_pdf = FIGURES_DIR / "wallclock_speedup.pdf"
    out_png = FIGURES_DIR / "wallclock_speedup.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")


# ── Markdown report ───────────────────────────────────────────────────────────

def write_report(
    sweep_results: dict,
    batch_results: dict,
    baseline_ms: float,
    baseline_std: float,
    baseline_tps: float,
) -> None:
    """Write docs/results/ablation_wallclock.md with full tables and paper sentence."""
    import datetime
    today = datetime.date.today().isoformat()

    lines = [
        "# Wall-Clock Benchmark — TSA Inference on Apple Silicon",
        "",
        f"**Date:** {today}",
        f"**Hardware:** {_detect_hardware()}",
        "**Framework:** MLX (Metal GPU)",
        f"**Model:** d_model={MODEL_DIMS['d_model']}, n_layers={MODEL_DIMS['n_layers']}, "
        f"n_heads={MODEL_DIMS['n_heads']}, d_ff={MODEL_DIMS['d_ff']}, vocab={VOCAB_SIZE}",
        f"**Benchmark:** {N_WARMUP} warmup + {N_TIMED} timed forward passes per config",
        f"**Primary config:** batch={PRIMARY_BATCH}, seq={PRIMARY_SEQLEN}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    # Find break-even points.
    # "Break-even" = the MAXIMUM active_frac at which speedup >= 1.0.
    # This is the upper operating limit: above it, overhead exceeds savings.
    # (The curve is monotonically decreasing: high sparsity → big savings → fast;
    #  low sparsity → little savings but full overhead → slow.)
    soft_breakeven = None
    sparse_breakeven = None
    for i, af in enumerate(sweep_results["active_frac"]):
        if sweep_results["soft_speedup"][i] >= 1.0:
            soft_breakeven = af   # keep updating → ends at maximum af with speedup >= 1
        if sweep_results["sparse_speedup"][i] >= 1.0:
            sparse_breakeven = af

    # Paper sentence
    p6_idx = sweep_results["active_frac"].index(PHASE6_AF)
    p7_idx = sweep_results["active_frac"].index(PHASE7_AF)
    p6_sparse_speedup = sweep_results["sparse_speedup"][p6_idx]
    p7_sparse_speedup = sweep_results["sparse_speedup"][p7_idx]
    p6_soft_overhead  = (sweep_results["soft_ms"][p6_idx] / baseline_ms - 1.0) * 100
    p7_soft_overhead  = (sweep_results["soft_ms"][p7_idx] / baseline_ms - 1.0) * 100

    p6_sparse_speedup_pct = (sweep_results["sparse_speedup"][p6_idx] - 1.0) * 100

    if sparse_breakeven is not None and sparse_breakeven >= PHASE6_AF:
        # Sparse-TSA is faster at Phase 6 (and possibly Phase 7) active fracs
        paper_sentence = (
            f"On Apple M1 Pro (batch={PRIMARY_BATCH}, seq={PRIMARY_SEQLEN}), "
            f"gather/scatter sparse-TSA (FFN-only, dense attention) achieves wall-clock "
            f"speedup for α ≤ {sparse_breakeven:.2f}. "
            f"At Phase 6's α={PHASE6_AF} ({sweep_results['ops_saved_pct'][p6_idx]:.1f}% ops saved), "
            f"sparse-TSA is {sweep_results['sparse_speedup'][p6_idx]:.3f}× faster than Baseline. "
            f"At Phase 7's α={PHASE7_AF} ({sweep_results['ops_saved_pct'][p7_idx]:.1f}% ops saved), "
            f"sparse-TSA is {p7_sparse_speedup:.3f}×. "
            f"Soft-gating adds {p7_soft_overhead:+.1f}% overhead at all active fractions."
        )
    elif sparse_breakeven is not None:
        paper_sentence = (
            f"On Apple M1 Pro (batch={PRIMARY_BATCH}, seq={PRIMARY_SEQLEN}), "
            f"gather/scatter sparse-TSA breaks even at α≤{sparse_breakeven:.2f}. "
            f"At Phase 7's α={PHASE7_AF} ({sweep_results['ops_saved_pct'][p7_idx]:.1f}% ops saved), "
            f"sparse-TSA is {p7_sparse_speedup:.3f}× vs Baseline. "
            f"Soft-gating adds {p7_soft_overhead:+.1f}% overhead. "
            f"Hardware-aware block-sparse kernels are needed to realise the theoretical "
            f"{sweep_results['ops_saved_pct'][p7_idx]:.0f}% ops savings as wall-clock gains."
        )
    else:
        paper_sentence = (
            f"On Apple M1 Pro, neither soft-gating ({p7_soft_overhead:+.1f}% overhead) "
            f"nor gather/scatter sparse-TSA ({p7_sparse_speedup:.3f}× at α={PHASE7_AF}) "
            f"achieves wall-clock speedup at the observed active fractions. "
            f"Hardware-aware block-sparse kernels are needed to realise the theoretical "
            f"{sweep_results['ops_saved_pct'][p7_idx]:.0f}% ops savings."
        )

    lines += [
        "**Paper sentence:**",
        f"> {paper_sentence}",
        "",
        f"- Baseline: **{baseline_ms:.1f} ms/step** ({baseline_tps/1000:.1f}k tok/s)",
        f"- Soft-TSA overhead at α={PHASE7_AF}: **{p7_soft_overhead:+.1f}%**",
        f"- Sparse-TSA speedup at α={PHASE7_AF}: **{p7_sparse_speedup:.3f}×**",
        f"- Soft-TSA break-even: **{'α=' + str(soft_breakeven) if soft_breakeven else 'NOT REACHED'}**",
        f"- Sparse-TSA break-even: **{'α=' + str(sparse_breakeven) if sparse_breakeven else 'NOT REACHED'}**",
        "",
        "---",
        "",
        "## Table 1: Active-fraction sweep",
        f"(batch={PRIMARY_BATCH}, seq={PRIMARY_SEQLEN})",
        "",
        "| α | Ops Saved | Baseline (ms) | Soft-TSA (ms) | Soft Speedup | Sparse-TSA (ms) | Sparse Speedup |",
        "|---|-----------|---------------|---------------|--------------|-----------------|----------------|",
    ]

    for i in range(len(sweep_results["active_frac"])):
        af  = sweep_results["active_frac"][i]
        ops = sweep_results["ops_saved_pct"][i]
        bms = sweep_results["baseline_ms"][i]
        sms = sweep_results["soft_ms"][i]
        sstd = sweep_results["soft_std"][i]
        ss  = sweep_results["soft_speedup"][i]
        pms = sweep_results["sparse_ms"][i]
        pstd = sweep_results["sparse_std"][i]
        ps  = sweep_results["sparse_speedup"][i]
        mark_p6 = " *(P6)*" if abs(af - PHASE6_AF) < 0.001 else ""
        mark_p7 = " *(P7)*" if abs(af - PHASE7_AF) < 0.001 else ""
        mark = mark_p6 + mark_p7
        lines.append(
            f"| {af:.3f}{mark} | {ops:.1f}% | {bms:.1f} | "
            f"{sms:.1f}±{sstd:.1f} | **{ss:.3f}×** | "
            f"{pms:.1f}±{pstd:.1f} | **{ps:.3f}×** |"
        )

    lines += [
        "",
        "---",
        "",
        "## Table 2: Batch-size scaling",
        f"(α={PHASE7_AF}, seq={PRIMARY_SEQLEN})",
        "",
        "| Batch | Baseline (ms) | Soft-TSA (ms) | Soft Speedup | Sparse-TSA (ms) | Sparse Speedup |",
        "|-------|---------------|---------------|--------------|-----------------|----------------|",
    ]

    for i in range(len(batch_results["batch_size"])):
        bs  = batch_results["batch_size"][i]
        bms = batch_results["baseline_ms"][i]
        sms = batch_results["soft_ms"][i]
        ss  = batch_results["soft_speedup"][i]
        pms = batch_results["sparse_ms"][i]
        ps  = batch_results["sparse_speedup"][i]
        lines.append(
            f"| {bs} | {bms:.1f} | {sms:.1f} | **{ss:.3f}×** | {pms:.1f} | **{ps:.3f}×** |"
        )

    lines += [
        "",
        "---",
        "",
        "## Methodology",
        "",
        "### Router",
        "- **Soft-TSA**: FixedRouter returns constant halt_prob=(1-α) for all tokens.",
        "  The continuous gate (1-halt_prob) = α scales all residuals. All tokens processed.",
        "- **Sparse-TSA**: BimodalRouter returns halt_prob=0 for the first α×T tokens,",
        "  halt_prob=1 for the rest. GatherSparseBlock uses threshold=0.5.",
        "  Active tokens (halt_prob<0.5) are gathered, run through FFN, scattered back.",
        "",
        "### Sparse execution overhead",
        "The GatherSparseBlock forward pass includes:",
        "1. Dense attention (all tokens) with hard binary gate on residual",
        "2. `mx.eval(active_mask)` — CPU-GPU sync to materialise boolean mask",
        "3. `np.where(active_np)` — CPU: get active indices",
        "4. `h_flat[active_idx]` — Metal gather kernel",
        "5. FFN on gathered tokens — sparse computation",
        "6. `h_flat.at[active_idx].add(ffn_out)` — Metal scatter kernel",
        "7. `mx.eval(h_flat)` — force scatter completion before next block",
        "",
        "The CPU-GPU sync (step 2) is the dominant overhead at typical batch sizes.",
        "With n_layers=6, there are 5 such syncs per forward pass.",
        "",
        "### Architecture note",
        "Attention remains dense in all variants. Skipping individual tokens from",
        "attention changes the all-to-all attention pattern relative to the trained model.",
        "Dense attention preserves exact semantic equivalence to the trained TSA.",
        "",
        "---",
        "",
        "## Figure",
        "",
        "![wallclock_speedup](../../docs/paper/figures/wallclock_speedup.png)",
        "",
        "Left: break-even curve (speedup vs active fraction).",
        "Right: batch-size scaling at Phase 7 observed active fraction.",
        "Vertical dashed lines mark Phase 6 (α=0.726) and Phase 7 (α=0.833).",
    ]

    out_path = RESULTS_DIR / "ablation_wallclock.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n  Saved: {out_path}")
    print(f"\n  Paper sentence:")
    print(f"  {paper_sentence}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Wall-clock TSA inference benchmark")
    print(f"  Model: d={MODEL_DIMS['d_model']} L={MODEL_DIMS['n_layers']} "
          f"h={MODEL_DIMS['n_heads']} vocab={VOCAB_SIZE}")
    print(f"  Primary: batch={PRIMARY_BATCH} seq={PRIMARY_SEQLEN}")
    print(f"  Warmup={N_WARMUP} timed={N_TIMED} per config")
    t_start = time.time()

    sweep_results, baseline_ms, baseline_std, baseline_tps = run_active_frac_sweep()
    batch_results = run_batch_scaling(PHASE7_AF)

    make_plots(sweep_results, batch_results)
    write_report(sweep_results, batch_results, baseline_ms, baseline_std, baseline_tps)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
