"""
Phase 7: Character-level language modeling on enwik8.

Validates TSA compute savings on a second dataset (first 10^8 bytes of English
Wikipedia). Replicates the Phase 6 TSA vs Baseline comparison to check that
the routing mechanism generalises beyond Shakespeare.

2 conditions (TSA validation only — SGC excluded, already shown not to
transfer to undifferentiated language data in Phase 6):
  1. Baseline Transformer — standard fixed-depth
  2. TSA-only             — same architecture + learned token routing

Decision D032: raw XML kept (standard enwik8 benchmark, see enwik8.py).
Decision D033: 10,000 training steps (2× Phase 6 — enwik8 is 90× larger,
  so we need more steps for meaningful convergence while keeping runtime
  under 4 hours per condition).
Decision D034: context_len=256 (2× Phase 6's 128 — Wikipedia sentences
  are longer than Shakespeare lines; longer context better exercises the router).
Decision D035: eval_interval=500 steps (same 5% ratio as Phase 6's 250/5000).
Decision D036: loss_thresholds=[2.5, 2.0, 1.8] — enwik8 is harder than
  Shakespeare; 3.0 will be reached very early and 1.5 may not be reached
  in 10K steps. Added 1.8 as an intermediate threshold.

All other hyperparameters match Phase 6 exactly:
  d_model=256, n_layers=6, n_heads=8, d_ff=1024
  lr=3e-4, batch_size=64, dropout=0.1, depth_reg_weight=0.001
  optimizer=AdamW(β=(0.9,0.95), wd=0.1), cosine LR schedule
  grad_clip=1.0, seed=42
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


from tsa.benchmarks.baseline_transformer import BaselineTransformer
from tsa.benchmarks.enwik8 import Enwik8Config, build_enwik8_datasets
from tsa.modules.topological_attention.variable_depth import TSAConfig, TSATransformer

RESULTS_DIR = Path(__file__).parent.parent / "docs" / "results" / "phase7"
FIGURES_DIR = Path(__file__).parent.parent / "docs" / "paper" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Hyperparameters ──────────────────────────────────────────────────────────

@dataclass
class Enwik8ExpConfig:
    # Model dims — identical to Phase 6
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    context_len: int = 256        # D034: 2× Phase 6
    dropout: float = 0.1
    depth_reg_weight: float = 0.001

    # Training — D033: 2× Phase 6 steps
    train_steps: int = 10_000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500      # D035: every 5% of training
    seed: int = 42

    # D036: loss thresholds for enwik8 difficulty range
    loss_thresholds: list[float] = field(default_factory=lambda: [2.5, 2.0, 1.8])


# ── Device ───────────────────────────────────────────────────────────────────

def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Model factory ────────────────────────────────────────────────────────────

def make_baseline(cfg: Enwik8ExpConfig, vocab_size: int) -> BaselineTransformer:
    model = BaselineTransformer(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.context_len,
        dropout=cfg.dropout,
    )
    # Remove padding_idx: enwik8 index 0 is a real character
    model.token_emb = nn.Embedding(vocab_size, cfg.d_model)
    nn.init.normal_(model.token_emb.weight, std=0.02)
    model.head.weight = model.token_emb.weight
    return model


def make_tsa(cfg: Enwik8ExpConfig, vocab_size: int) -> TSATransformer:
    tsa_cfg = TSAConfig(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.context_len,
        dropout=cfg.dropout,
        depth_reg_weight=cfg.depth_reg_weight,
    )
    model = TSATransformer(tsa_cfg)
    model.token_emb = nn.Embedding(vocab_size, cfg.d_model)
    nn.init.normal_(model.token_emb.weight, std=0.02)
    model.head.weight = model.token_emb.weight
    return model


# ── Batch sampler ─────────────────────────────────────────────────────────────

def random_batch(
    data: torch.Tensor,
    context_len: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample batch_size random overlapping context windows."""
    max_start = len(data) - context_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s : s + context_len] for s in starts]).to(device)
    y = torch.stack([data[s + 1 : s + context_len + 1] for s in starts]).to(device)
    return x, y


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: torch.Tensor,
    context_len: int,
    batch_size: int,
    device: str,
    n_batches: int = 20,
) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = random_batch(data, context_len, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        total_loss += loss.item()
    model.train()
    return total_loss / n_batches


# ── Training loop ─────────────────────────────────────────────────────────────

@dataclass
class ConditionResult:
    label: str
    val_losses: list[float]
    eval_steps: list[int]
    token_layer_ops: list[float]
    active_fracs: list[float]
    steps_to_threshold: dict[float, int | None]
    final_val_loss: float
    final_bpc: float
    elapsed: float
    n_params: int


def train_condition(
    label: str,
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: Enwik8ExpConfig,
    device: str,
) -> ConditionResult:
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    model = model.to(device)
    opt = model.configure_optimizers(cfg.lr, cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.train_steps, eta_min=cfg.lr * 0.1
    )

    val_losses, eval_steps, tl_ops_log, active_frac_log = [], [], [], []
    steps_to_threshold = {t: None for t in cfg.loss_thresholds}
    cumulative_ops = 0.0

    model.train()
    t0 = time.time()

    for step in range(1, cfg.train_steps + 1):
        x, y = random_batch(train_data, cfg.context_len, cfg.batch_size, device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if hasattr(model, "aux_loss"):
            loss = loss + model.aux_loss().to(loss.device)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        scheduler.step()

        # Track token-layer ops
        af = getattr(model, "_last_mean_active_frac", 1.0)
        effective_layers = 1 + (cfg.n_layers - 1) * af
        cumulative_ops += cfg.batch_size * cfg.context_len * effective_layers

        if step % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_data, cfg.context_len, cfg.batch_size, device)
            val_losses.append(val_loss)
            eval_steps.append(step)
            tl_ops_log.append(cumulative_ops)
            active_frac_log.append(af)

            for thresh in cfg.loss_thresholds:
                if steps_to_threshold[thresh] is None and val_loss <= thresh:
                    steps_to_threshold[thresh] = step

            bpc = val_loss / math.log(2)
            af_str = f"  active={af:.3f}" if hasattr(model, "_last_mean_active_frac") else ""
            elapsed_min = (time.time() - t0) / 60
            print(f"  step {step:5d}  val_loss={val_loss:.4f}  bpc={bpc:.4f}{af_str}  [{elapsed_min:.1f}m]")

    elapsed = time.time() - t0
    final_val_loss = val_losses[-1] if val_losses else float("nan")
    final_bpc = final_val_loss / math.log(2)

    print(f"\n  Final val loss: {final_val_loss:.4f}  BPC: {final_bpc:.4f}")
    print(f"  Wall time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
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


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: list[ConditionResult]) -> None:
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })

    BASELINE_COLOR = "#2d6a9f"
    TSA_COLOR      = "#c0392b"

    def color_for(label: str) -> str:
        return TSA_COLOR if "TSA" in label else BASELINE_COLOR

    # Figure 1: val loss vs steps
    fig1, ax1 = plt.subplots(figsize=(4.5, 3.0))
    for r in results:
        ax1.plot(r.eval_steps, r.val_losses,
                 color=color_for(r.label), label=r.label,
                 linestyle="--" if "TSA" in r.label else "-")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Validation loss (nats)")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.legend(loc="upper right", frameon=False)
    fig1.tight_layout()
    out1 = FIGURES_DIR / "enwik8_val_loss_vs_steps.pdf"
    fig1.savefig(out1, format="pdf", bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {out1}")

    # Figure 2: val loss vs cumulative token-layer ops
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.0))
    for r in results:
        ops_b = [o / 1e9 for o in r.token_layer_ops]
        ax2.plot(ops_b, r.val_losses,
                 color=color_for(r.label), label=r.label,
                 linestyle="--" if "TSA" in r.label else "-")
    ax2.set_xlabel("Cumulative token-layer ops (×10⁹)")
    ax2.set_ylabel("Validation loss (nats)")
    ax2.legend(loc="upper right", frameon=False)
    fig2.tight_layout()
    out2 = FIGURES_DIR / "enwik8_val_loss_vs_compute.pdf"
    fig2.savefig(out2, format="pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2}")


# ── Summary + results file ────────────────────────────────────────────────────

def print_and_save_summary(
    results: list[ConditionResult],
    cfg: Enwik8ExpConfig,
    vocab_size: int,
) -> None:
    sep = "=" * 80
    print(f"\n\n{sep}")
    print("  PHASE 7 RESULTS — enwik8 Character-Level Language Modeling")
    print(sep)
    print(f"\n  Model: d={cfg.d_model}, L={cfg.n_layers}, h={cfg.n_heads}, "
          f"d_ff={cfg.d_ff}, vocab={vocab_size}")
    print(f"  Context: {cfg.context_len} chars  |  Steps: {cfg.train_steps:,}  |"
          f"  Batch: {cfg.batch_size}\n")

    for r in results:
        print(f"  {r.label}")
        print(f"    params={r.n_params:,}  val_loss={r.final_val_loss:.4f}  "
              f"bpc={r.final_bpc:.4f}  time={r.elapsed:.0f}s")
        if r.active_fracs:
            af = sum(r.active_fracs) / len(r.active_fracs)
            n_layers = cfg.n_layers
            tlops_saved = 1.0 - (1 + (n_layers - 1) * af) / n_layers
            print(f"    mean_active_frac={af:.4f}  TLOps_saved={tlops_saved*100:.1f}%")
        for t, s in r.steps_to_threshold.items():
            print(f"    steps_to_{t}={s if s else 'NOT REACHED'}")
        print()

    # Compute efficiency comparison
    baseline = next((r for r in results if "Baseline" in r.label), None)
    tsa      = next((r for r in results if "TSA" in r.label), None)

    if baseline and tsa:
        loss_delta = tsa.final_val_loss - baseline.final_val_loss
        loss_pct   = loss_delta / baseline.final_val_loss * 100
        mean_af    = sum(tsa.active_fracs) / len(tsa.active_fracs) if tsa.active_fracs else 1.0
        tlops_saved = 1.0 - (1 + (cfg.n_layers - 1) * mean_af) / cfg.n_layers
        print(f"  TSA vs Baseline:")
        print(f"    TLOps saved:       {tlops_saved*100:.1f}%")
        print(f"    Val loss delta:    {loss_delta:+.4f} nats  ({loss_pct:+.1f}%)")
        success = tlops_saved >= 0.15 and abs(loss_pct) < 1.0
        print(f"    Success criterion: {'PASS ✓' if success else 'CHECK ✗'} "
              f"(≥15% ops saved AND <1% quality loss)")
    print(f"\n{sep}\n")

    # Write results to file
    lines = [
        f"Phase 7 enwik8 Results",
        f"d_model={cfg.d_model}, n_layers={cfg.n_layers}, context={cfg.context_len}, vocab={vocab_size}",
        f"steps={cfg.train_steps}, batch={cfg.batch_size}",
        "",
    ]
    for r in results:
        lines.append(f"{r.label}")
        lines.append(f"  params={r.n_params:,}  val_loss={r.final_val_loss:.4f}  "
                     f"bpc={r.final_bpc:.4f}  time={r.elapsed:.0f}s")
        if r.active_fracs:
            af = sum(r.active_fracs) / len(r.active_fracs)
            tlops = 1.0 - (1 + (cfg.n_layers - 1) * af) / cfg.n_layers
            lines.append(f"  mean_active_frac={af:.4f}  tlops_saved={tlops*100:.1f}%")
        for t, s in r.steps_to_threshold.items():
            lines.append(f"  steps_to_{t}={s if s else 'NOT REACHED'}")
        for step, vl, af in zip(r.eval_steps, r.val_losses, r.active_fracs or [0]*len(r.eval_steps)):
            af_s = f"  active={af:.3f}" if r.active_fracs else ""
            lines.append(f"    step={step:5d}  val_loss={vl:.4f}  bpc={vl/math.log(2):.4f}{af_s}")
        lines.append("")

    out = RESULTS_DIR / "results.txt"
    out.write_text("\n".join(lines))
    print(f"  Numeric results saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)
    device = _resolve_device()
    print(f"\nDevice: {device}")
    print(f"Phase 7: enwik8 character-level language modeling")
    print(f"  context_len=256, train_steps=10,000, conditions=2 (Baseline + TSA)")

    cfg = Enwik8ExpConfig()

    print("\nLoading enwik8 corpus...")
    enwik8_cfg = Enwik8Config(context_len=cfg.context_len)
    train_ds, val_ds, test_ds, tokenizer = build_enwik8_datasets(enwik8_cfg)
    vocab_size = tokenizer.vocab_size

    train_data = train_ds.data
    val_data   = val_ds.data

    results = []

    # Condition 1: Baseline
    torch.manual_seed(cfg.seed)
    r1 = train_condition(
        "Baseline (static, random sampling)",
        make_baseline(cfg, vocab_size),
        train_data, val_data, cfg, device,
    )
    results.append(r1)

    # Condition 2: TSA-only
    torch.manual_seed(cfg.seed)
    r2 = train_condition(
        "TSA-only (static, random sampling)",
        make_tsa(cfg, vocab_size),
        train_data, val_data, cfg, device,
    )
    results.append(r2)

    # Plots + summary
    plot_results(results)
    print_and_save_summary(results, cfg, vocab_size)


if __name__ == "__main__":
    main()
