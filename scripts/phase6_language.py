"""
Phase 6: Character-level language modeling on Shakespeare.

Validates whether Genesis efficiency gains (TSA + SGC) survive on real text.
This is the experiment that turns toy results into paper-ready results.

4 conditions:
  1. Baseline         — standard transformer, static (random) sampling
  2. TSA-only         — same architecture + topological sparse attention
  3. SGC + Baseline   — loss-weighted adaptive curriculum, standard transformer
  4. Genesis (TSA+SGC) — TSA routing + loss-weighted adaptive curriculum

CKR parked: language has no natural discrete task_ids (D025).
SGC simplified: loss-weighted chunk sampling replaces REINFORCE (D028).

PRIMARY metric: val loss (cross-entropy, nats) vs training steps
SECONDARY metric: val loss vs cumulative token-layer ops (compute efficiency)

FAILURE condition (pre-defined):
  Phase 6 FAILS if:
    (a) TSA shows no compute reduction at equivalent val loss, AND
    (b) SGC shows no convergence speedup vs static baseline
  Failure is a valid result — documents that gains don't transfer to language.

Stretch: 10M-param scaling teaser (run with --scale 10m flag).
"""
from __future__ import annotations

import argparse
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
from tsa.benchmarks.shakespeare import ShakespeareConfig, build_shakespeare_datasets
from tsa.modules.curriculum_gen.language_sgc import (
    LossWeightedSampler,
    compute_per_sample_loss,
)
from tsa.modules.topological_attention.variable_depth import TSAConfig, TSATransformer

RESULTS_DIR = Path(__file__).parent.parent / "docs" / "results" / "phase6"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Hyperparameters ──────────────────────────────────────────────────────────

@dataclass
class LangExpConfig:
    # Model dims — same for all conditions
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    context_len: int = 128
    dropout: float = 0.1
    depth_reg_weight: float = 0.001   # small λ for language — don't sacrifice accuracy

    # Training
    train_steps: int = 5_000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    seed: int = 42

    # SGC curriculum
    sgc_alpha: float = 0.6       # loss prioritization temperature
    sgc_ema_decay: float = 0.9   # EMA decay for chunk loss estimates

    # Val loss thresholds for steps-to-threshold metric
    loss_thresholds: list[float] = field(default_factory=lambda: [3.0, 2.5, 2.0])


# ── Device ───────────────────────────────────────────────────────────────────

def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Model factory (removes padding_idx=0 for char-level LM) ─────────────────

def make_baseline(cfg: LangExpConfig, vocab_size: int) -> BaselineTransformer:
    """
    Build BaselineTransformer for language.

    Reinitialises token_emb without padding_idx=0 — for char-level LM every
    character token is meaningful, including index 0 (typically '\n').
    """
    model = BaselineTransformer(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.context_len,
        dropout=cfg.dropout,
    )
    # Remove padding_idx: all chars are meaningful
    model.token_emb = nn.Embedding(vocab_size, cfg.d_model)
    nn.init.normal_(model.token_emb.weight, std=0.02)
    model.head.weight = model.token_emb.weight  # re-tie weights
    return model


def make_tsa(cfg: LangExpConfig, vocab_size: int) -> TSATransformer:
    """Build TSATransformer for language (same dims as baseline)."""
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
    # Remove padding_idx for same reason as above
    model.token_emb = nn.Embedding(vocab_size, cfg.d_model)
    nn.init.normal_(model.token_emb.weight, std=0.02)
    model.head.weight = model.token_emb.weight
    return model


# ── Batch samplers ────────────────────────────────────────────────────────────

def random_batch(
    data: torch.Tensor,
    context_len: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample batch_size random context windows from token data."""
    max_start = len(data) - context_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s : s + context_len] for s in starts]).to(device)
    y = torch.stack([data[s + 1 : s + context_len + 1] for s in starts]).to(device)
    return x, y


def sgc_batch(
    data: torch.Tensor,
    context_len: int,
    batch_size: int,
    device: str,
    sampler: LossWeightedSampler,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a batch using loss-weighted chunk indices.

    Returns (x, y, chunk_ids) where chunk_ids are needed to update the sampler.
    Chunks are non-overlapping (stride = context_len + 1) to match ShakespeareDataset.
    """
    chunk_ids = sampler.sample(batch_size)
    stride = context_len + 1
    x_list, y_list = [], []
    for cid in chunk_ids.tolist():
        start = int(cid) * stride
        # Safety: clip to data bounds
        start = min(start, len(data) - context_len - 1)
        x_list.append(data[start : start + context_len])
        y_list.append(data[start + 1 : start + context_len + 1])
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y, chunk_ids


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
    """Estimate val loss over n_batches random windows."""
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
    val_losses: list[float]           # val loss at each eval step
    eval_steps: list[int]             # training step at each eval
    token_layer_ops: list[float]      # cumulative token-layer ops at each eval
    active_fracs: list[float]         # mean_active_frac at each eval (TSA only)
    steps_to_threshold: dict[float, int | None]   # threshold → step
    final_val_loss: float
    final_bpc: float
    elapsed: float
    n_params: int


def train_condition(
    label: str,
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: LangExpConfig,
    device: str,
    use_sgc: bool = False,
) -> ConditionResult:
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    model = model.to(device)
    opt = model.configure_optimizers(cfg.lr, cfg.weight_decay)

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.train_steps, eta_min=cfg.lr * 0.1
    )

    # SGC curriculum sampler (if requested)
    stride = cfg.context_len + 1
    n_chunks = len(train_data) // stride
    sampler = LossWeightedSampler(n_chunks, alpha=cfg.sgc_alpha, ema_decay=cfg.sgc_ema_decay) \
        if use_sgc else None

    val_losses, eval_steps, tl_ops_log, active_frac_log = [], [], [], []
    steps_to_threshold = {t: None for t in cfg.loss_thresholds}

    # Cumulative token-layer ops tracker
    # One "token-layer op" = one token processed by one transformer block
    # For baseline: ops += batch * context * n_layers each step
    # For TSA: ops += batch * context * n_layers * mean_active_frac
    n_layers = cfg.n_layers
    cumulative_ops = 0.0

    model.train()
    t0 = time.time()

    for step in range(1, cfg.train_steps + 1):
        # ── Batch sampling ──────────────────────────────────────────────────
        if use_sgc and sampler is not None:
            x, y, chunk_ids = sgc_batch(
                train_data, cfg.context_len, cfg.batch_size, device, sampler
            )
        else:
            x, y = random_batch(train_data, cfg.context_len, cfg.batch_size, device)
            chunk_ids = None

        # ── Forward ─────────────────────────────────────────────────────────
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        # Add TSA depth regularisation
        if hasattr(model, "aux_loss"):
            loss = loss + model.aux_loss().to(loss.device)

        # ── Backward ────────────────────────────────────────────────────────
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        scheduler.step()

        # ── SGC curriculum update ────────────────────────────────────────────
        if use_sgc and sampler is not None and chunk_ids is not None:
            with torch.no_grad():
                per_sample = compute_per_sample_loss(logits.detach(), y)
            sampler.update(chunk_ids.cpu(), per_sample.cpu())

        # ── Token-layer ops tracking ─────────────────────────────────────────
        if hasattr(model, "_last_mean_active_frac"):
            af = model._last_mean_active_frac
        else:
            af = 1.0
        # Block 0 always runs (active=1), then n_layers-1 gated blocks at frac af
        # Effective layers per token = 1 + (n_layers - 1) * af
        effective_layers = 1 + (n_layers - 1) * af
        cumulative_ops += cfg.batch_size * cfg.context_len * effective_layers

        # ── Evaluate ─────────────────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            val_loss = evaluate(
                model, val_data, cfg.context_len, cfg.batch_size, device
            )
            val_losses.append(val_loss)
            eval_steps.append(step)
            tl_ops_log.append(cumulative_ops)
            active_frac_log.append(af)

            for thresh in cfg.loss_thresholds:
                if steps_to_threshold[thresh] is None and val_loss <= thresh:
                    steps_to_threshold[thresh] = step

            bpc = val_loss / math.log(2)
            af_str = f"  active={af:.3f}" if hasattr(model, "_last_mean_active_frac") else ""
            print(f"  step {step:5d}  val_loss={val_loss:.4f}  bpc={bpc:.4f}{af_str}")

    elapsed = time.time() - t0
    final_val_loss = val_losses[-1] if val_losses else float("nan")
    final_bpc = final_val_loss / math.log(2)

    print(f"\n  Final val loss: {final_val_loss:.4f}  BPC: {final_bpc:.4f}")
    print(f"  Wall time: {elapsed:.1f}s")
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

COLORS = {
    "Baseline":    "#4C72B0",
    "TSA-only":    "#DD8452",
    "SGC+Baseline":"#55A868",
    "Genesis":     "#C44E52",
}
LINE_STYLES = {
    "Baseline":    "-",
    "TSA-only":    "--",
    "SGC+Baseline":"-.",
    "Genesis":     "-",
}


def plot_val_loss_vs_steps(results: list[ConditionResult], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        key = r.label.split("(")[0].strip()
        ax.plot(
            r.eval_steps, r.val_losses,
            label=f"{r.label}  (final={r.final_val_loss:.3f})",
            color=COLORS.get(key, None),
            linestyle=LINE_STYLES.get(key, "-"),
            linewidth=2,
        )
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Validation loss (nats)")
    ax.set_title("Phase 6 — Val loss vs Training steps (Shakespeare char-level LM)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_val_loss_vs_compute(results: list[ConditionResult], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        key = r.label.split("(")[0].strip()
        ops_billions = [o / 1e9 for o in r.token_layer_ops]
        ax.plot(
            ops_billions, r.val_losses,
            label=f"{r.label}",
            color=COLORS.get(key, None),
            linestyle=LINE_STYLES.get(key, "-"),
            linewidth=2,
        )
    ax.set_xlabel("Cumulative token-layer ops (billions)")
    ax.set_ylabel("Validation loss (nats)")
    ax.set_title("Phase 6 — Val loss vs Compute (token-layer ops)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: list[ConditionResult], cfg: LangExpConfig) -> dict:
    sep = "=" * 80
    print(f"\n\n{sep}")
    print("  GENESIS PHASE 6 RESULTS — Character-Level Language Modeling")
    print(sep)
    print(f"\n  Model: d={cfg.d_model}, L={cfg.n_layers}, h={cfg.n_heads}, d_ff={cfg.d_ff}")
    print(f"  Context: {cfg.context_len} chars  |  Steps: {cfg.train_steps:,}  |"
          f"  Batch: {cfg.batch_size}\n")

    # Header
    print(f"  {'Condition':<28}  {'Params':>8}  {'Val Loss':>9}  {'BPC':>7}", end="")
    for t in cfg.loss_thresholds:
        print(f"  {'→'+str(t):>8}", end="")
    print()
    print(f"  {'-'*28}  {'-'*8}  {'-'*9}  {'-'*7}", end="")
    for _ in cfg.loss_thresholds:
        print(f"  {'-'*8}", end="")
    print()

    for r in results:
        print(f"  {r.label:<28}  {r.n_params:>8,}  {r.final_val_loss:>9.4f}  {r.final_bpc:>7.4f}", end="")
        for t in cfg.loss_thresholds:
            s = r.steps_to_threshold.get(t)
            print(f"  {str(s) if s else 'MISS':>8}", end="")
        print()

    # Compute savings for TSA
    baseline = next((r for r in results if r.label.startswith("Baseline")), None)
    tsa = next((r for r in results if r.label.startswith("TSA")), None)
    tsa = next((r for r in results if r.label.startswith("Genesis")), None)
    sgc_base = next((r for r in results if r.label.startswith("SGC")), None)

    print(f"\n  COMPUTE EFFICIENCY (TSA):")
    if tsa and tsa.active_fracs:
        mean_af = sum(tsa.active_fracs) / len(tsa.active_fracs)
        ops_saved = (1 - mean_af) * (cfg.n_layers - 1) / cfg.n_layers
        print(f"    TSA mean active frac: {mean_af:.4f}")
        print(f"    Approx token-layer ops saved: {ops_saved*100:.1f}%")
    if tsa and tsa.active_fracs:
        mean_af_g = sum(tsa.active_fracs) / len(tsa.active_fracs)
        ops_saved_g = (1 - mean_af_g) * (cfg.n_layers - 1) / cfg.n_layers
        print(f"    Genesis mean active frac: {mean_af_g:.4f}")
        print(f"    Approx token-layer ops saved: {ops_saved_g*100:.1f}%")

    print(f"\n  CONVERGENCE SPEEDUP (SGC):")
    for thresh in cfg.loss_thresholds:
        if baseline and sgc_base:
            s_b = baseline.steps_to_threshold.get(thresh)
            s_s = sgc_base.steps_to_threshold.get(thresh)
            if s_b and s_s:
                speedup = (s_b - s_s) / s_b * 100
                print(f"    val_loss≤{thresh}: Baseline={s_b}  SGC+Baseline={s_s}  speedup={speedup:+.1f}%")
        if baseline and tsa:
            s_b = baseline.steps_to_threshold.get(thresh)
            s_g = tsa.steps_to_threshold.get(thresh)
            if s_b and s_g:
                speedup = (s_b - s_g) / s_b * 100
                print(f"    val_loss≤{thresh}: Baseline={s_b}  Genesis={s_g}      speedup={speedup:+.1f}%")

    # Verdict
    print(f"\n  VERDICT:")
    tsa_saves_compute = (tsa is not None and tsa.active_fracs and
                         sum(tsa.active_fracs) / len(tsa.active_fracs) < 0.95)
    sgc_speeds_up = any(
        sgc_base.steps_to_threshold.get(t) is not None and
        baseline is not None and baseline.steps_to_threshold.get(t) is not None and
        sgc_base.steps_to_threshold[t] < baseline.steps_to_threshold[t]
        for t in cfg.loss_thresholds
    ) if sgc_base and baseline else False
    genesis_best = (
        tsa is not None and baseline is not None and
        tsa.final_val_loss < baseline.final_val_loss
    )

    if tsa_saves_compute and sgc_speeds_up and genesis_best:
        verdict = "✓✓ FULL TRANSFER — TSA saves compute, SGC speeds convergence, Genesis best overall"
    elif tsa_saves_compute and sgc_speeds_up:
        verdict = "✓  TRANSFER CONFIRMED — both mechanisms work on language"
    elif tsa_saves_compute:
        verdict = "~  PARTIAL — TSA saves compute but SGC speedup unclear"
    elif sgc_speeds_up:
        verdict = "~  PARTIAL — SGC speeds convergence but TSA routing minimal"
    else:
        verdict = "✗  GAINS DO NOT TRANSFER at this scale — valuable negative result"

    print(f"    {verdict}")
    print(f"\n{sep}\n")

    return {
        "tsa_saves_compute": tsa_saves_compute,
        "sgc_speeds_up": sgc_speeds_up,
        "genesis_best": genesis_best,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(scale: str = "5m") -> None:
    torch.manual_seed(42)
    device = _resolve_device()
    print(f"\nDevice: {device}")

    # Scale config
    cfg = LangExpConfig()
    if scale == "10m":
        cfg.n_layers = 12   # double depth → ~9.6M params
        cfg.train_steps = 7_500
        print("Scale: 10M (n_layers=12)")
    else:
        print(f"Scale: ~{sum(256*256*3 + 256*256 + 256*1024*2 for _ in range(6)) // 1_000_000}M params")

    # Dataset
    print("\nLoading Shakespeare corpus...")
    sh_cfg = ShakespeareConfig(context_len=cfg.context_len)
    train_ds, val_ds, test_ds, tokenizer = build_shakespeare_datasets(sh_cfg)
    vocab_size = tokenizer.vocab_size

    # Use raw token tensors for efficient random-window batching
    # (avoids DataLoader overhead for fixed-size random access)
    train_data = train_ds.data
    val_data = val_ds.data

    results = []

    # ── Condition 1: Baseline + Static ──────────────────────────────────────
    torch.manual_seed(cfg.seed)
    r1 = train_condition(
        "Baseline (static, random sampling)",
        make_baseline(cfg, vocab_size),
        train_data, val_data, cfg, device, use_sgc=False,
    )
    results.append(r1)

    # ── Condition 2: TSA-only + Static ──────────────────────────────────────
    torch.manual_seed(cfg.seed)
    r2 = train_condition(
        "TSA-only (static, random sampling)",
        make_tsa(cfg, vocab_size),
        train_data, val_data, cfg, device, use_sgc=False,
    )
    results.append(r2)

    # ── Condition 3: SGC + Baseline ─────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    r3 = train_condition(
        "SGC+Baseline (loss-weighted curriculum)",
        make_baseline(cfg, vocab_size),
        train_data, val_data, cfg, device, use_sgc=True,
    )
    results.append(r3)

    # ── Condition 4: Genesis (TSA + SGC) ────────────────────────────────────
    torch.manual_seed(cfg.seed)
    r4 = train_condition(
        "Genesis (TSA + SGC, loss-weighted)",
        make_tsa(cfg, vocab_size),
        train_data, val_data, cfg, device, use_sgc=True,
    )
    results.append(r4)

    # ── Summary + Plots ──────────────────────────────────────────────────────
    suffix = f"_{scale}" if scale != "5m" else ""
    plot_val_loss_vs_steps(
        results, RESULTS_DIR / f"val_loss_vs_steps{suffix}.png"
    )
    plot_val_loss_vs_compute(
        results, RESULTS_DIR / f"val_loss_vs_compute{suffix}.png"
    )
    verdict = print_summary(results, cfg)

    # Save numeric results for docs
    _save_results_md(results, cfg, verdict, scale)


def _save_results_md(
    results: list[ConditionResult],
    cfg: LangExpConfig,
    verdict: dict,
    scale: str,
) -> None:
    suffix = f"_{scale}" if scale != "5m" else ""
    out = RESULTS_DIR / f"results{suffix}.txt"
    lines = []
    lines.append(f"Phase 6 Language Results — scale={scale}")
    lines.append(f"d_model={cfg.d_model}, n_layers={cfg.n_layers}, context={cfg.context_len}")
    lines.append("")
    for r in results:
        lines.append(f"{r.label}")
        lines.append(f"  params={r.n_params:,}  val_loss={r.final_val_loss:.4f}  "
                     f"bpc={r.final_bpc:.4f}  time={r.elapsed:.0f}s")
        if r.active_fracs:
            af = sum(r.active_fracs) / len(r.active_fracs)
            lines.append(f"  mean_active_frac={af:.4f}")
        for t, s in r.steps_to_threshold.items():
            lines.append(f"  steps_to_{t}={s}")
        lines.append("")
    out.write_text("\n".join(lines))
    print(f"  Numeric results saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["5m", "10m"], default="5m")
    args = parser.parse_args()
    main(scale=args.scale)
