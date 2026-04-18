"""
Generate publication-quality PDF figures for the TSA paper.

Figure 1: Val loss vs training steps (Baseline vs TSA-only)
Figure 2: Val loss vs cumulative token-layer ops (Baseline vs TSA-only)

Data sourced from docs/results/phase6/results.txt experiment output.
TSA paper covers TSA only — SGC and Genesis conditions are excluded.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Experimental data (from scripts/phase6_language.py output)
# ---------------------------------------------------------------------------

STEPS = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500,
         2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000]

BASELINE_LOSS = [2.2421, 1.8855, 1.7268, 1.6202, 1.5633, 1.5266, 1.5086, 1.4962,
                 1.4783, 1.4612, 1.4585, 1.4531, 1.4555, 1.4498, 1.4380, 1.4509,
                 1.4294, 1.4379, 1.4369, 1.4422]

TSA_LOSS = [2.2375, 1.9021, 1.7114, 1.6123, 1.5632, 1.5293, 1.5155, 1.4832,
            1.4813, 1.4753, 1.4603, 1.4522, 1.4543, 1.4642, 1.4520, 1.4523,
            1.4456, 1.4343, 1.4426, 1.4482]

# Active fractions per eval step for TSA (from experiment logs)
TSA_ACTIVE_FRAC = [0.741, 0.804, 0.785, 0.763, 0.743, 0.736, 0.726, 0.718,
                   0.716, 0.716, 0.716, 0.714, 0.706, 0.706, 0.708, 0.709,
                   0.704, 0.705, 0.706, 0.702]

# Model config
N_LAYERS = 6
BATCH_SIZE = 64
CONTEXT_LEN = 128

# ---------------------------------------------------------------------------
# Cumulative TLOps computation
# TLOps per step = B * T * effective_layers
# effective_layers = 1 (stem) + (L-1) * active_frac
# Baseline: effective_layers = L = 6 always
# TSA: effective_layers = 1 + 5 * active_frac (varies per eval checkpoint)
# We approximate cumulative ops by linear interpolation between eval points
# ---------------------------------------------------------------------------

def cumulative_tlops(steps_list, active_fracs, batch=BATCH_SIZE, context=CONTEXT_LEN, n_layers=N_LAYERS):
    """Compute cumulative token-layer ops at each eval checkpoint."""
    tokens_per_step = batch * context
    cumops = []
    total = 0.0
    prev_step = 0
    for i, step in enumerate(steps_list):
        interval = step - prev_step
        eff_layers = 1 + (n_layers - 1) * active_fracs[i]
        total += interval * tokens_per_step * eff_layers
        cumops.append(total)
        prev_step = step
    return cumops

baseline_active = [1.0] * len(STEPS)
baseline_ops = cumulative_tlops(STEPS, baseline_active)
tsa_ops = cumulative_tlops(STEPS, TSA_ACTIVE_FRAC)

# Scale to billions for axis readability
baseline_ops_b = [x / 1e9 for x in baseline_ops]
tsa_ops_b = [x / 1e9 for x in tsa_ops]

# ---------------------------------------------------------------------------
# Style configuration — clean, serif, no grid
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
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
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BASELINE_COLOR = "#2d6a9f"   # steel blue
TSA_COLOR      = "#c0392b"   # deep red

# ---------------------------------------------------------------------------
# Figure 1: Val loss vs training steps
# ---------------------------------------------------------------------------

fig1, ax1 = plt.subplots(figsize=(4.5, 3.0))

ax1.plot(STEPS, BASELINE_LOSS, color=BASELINE_COLOR, linestyle="-",
         label="Baseline ($\\alpha=1.00$)", zorder=3)
ax1.plot(STEPS, TSA_LOSS, color=TSA_COLOR, linestyle="-",
         label="TSA ($\\alpha=0.73$)", zorder=3)

ax1.set_xlabel("Training step")
ax1.set_ylabel("Validation loss (nats)")
ax1.set_xlim(0, 5200)
ax1.set_ylim(1.38, 2.35)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

ax1.legend(loc="upper right", frameon=False)

# Annotate final values
ax1.annotate(f"1.4422", xy=(5000, 1.4422), xytext=(4600, 1.410),
             fontsize=9, color=BASELINE_COLOR,
             arrowprops=dict(arrowstyle="-", color=BASELINE_COLOR, lw=0.8))
ax1.annotate(f"1.4482", xy=(5000, 1.4482), xytext=(4600, 1.465),
             fontsize=9, color=TSA_COLOR,
             arrowprops=dict(arrowstyle="-", color=TSA_COLOR, lw=0.8))

fig1.tight_layout()
out1 = os.path.join(FIGURES_DIR, "val_loss_vs_steps.pdf")
fig1.savefig(out1, format="pdf")
plt.close(fig1)
print(f"Saved: {out1}")

# ---------------------------------------------------------------------------
# Figure 2: Val loss vs cumulative token-layer ops
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(4.5, 3.0))

ax2.plot(baseline_ops_b, BASELINE_LOSS, color=BASELINE_COLOR, linestyle="-",
         label="Baseline (1.00 layers/token)", zorder=3)
ax2.plot(tsa_ops_b, TSA_LOSS, color=TSA_COLOR, linestyle="-",
         label="TSA (0.73 layers/token)", zorder=3)

ax2.set_xlabel("Cumulative token-layer ops ($\\times 10^9$)")
ax2.set_ylabel("Validation loss (nats)")
ax2.set_ylim(1.38, 2.35)

# Mark the TSA savings at convergence with an arrow
final_baseline_ops = baseline_ops_b[-1]
final_tsa_ops = tsa_ops_b[-1]
ax2.annotate(
    "",
    xy=(final_tsa_ops, 1.45),
    xytext=(final_baseline_ops, 1.45),
    arrowprops=dict(arrowstyle="<->", color="gray", lw=1.0),
)
ax2.text((final_tsa_ops + final_baseline_ops) / 2, 1.44,
         "22.8\\% fewer ops",
         ha="center", va="top", fontsize=8.5, color="gray")

ax2.legend(loc="upper right", frameon=False)

fig2.tight_layout()
out2 = os.path.join(FIGURES_DIR, "val_loss_vs_compute.pdf")
fig2.savefig(out2, format="pdf")
plt.close(fig2)
print(f"Saved: {out2}")

print("Figures generated successfully.")
