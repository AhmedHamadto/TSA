"""
Generate two new figures for the TSA paper:
  1. Architecture diagram showing the TSA gating mechanism
  2. Routing heatmap from a trained TSA model on Shakespeare text
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os, sys

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def draw_architecture():
    fig, ax = plt.subplots(figsize=(4.8, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 11.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colors
    BLOCK_C = "#d4e6f1"
    ROUTER_C = "#f9e79f"
    GATE_C = "#d5f5e3"
    STEM_C = "#e8daef"
    ARROW_C = "#444444"
    RED = "#c0392b"

    bw, bh = 3.2, 0.85
    rw, rh = 2.2, 0.75
    cx = 4.2  # main column center

    def rounded_box(x, y, w, h, color, label, fontsize=9.5):
        r = mpatches.FancyBboxPatch((x - w/2, y), w, h, boxstyle="round,pad=0.08",
                                     facecolor=color, edgecolor="#333", linewidth=1.1)
        ax.add_patch(r)
        ax.text(x, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")

    def arrow(x1, y1, x2, y2, color=ARROW_C, lw=1.3):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                     shrinkA=2, shrinkB=2))

    # ── Bottom: input ────────────────────────────────────────────────────
    y = 0.0
    ax.text(cx, y, "Token embeddings $h$", ha="center", va="center",
            fontsize=9, style="italic", color="#555")
    arrow(cx, y + 0.25, cx, y + 0.65)

    # ── Stem block ───────────────────────────────────────────────────────
    stem_y = 0.9
    rounded_box(cx, stem_y, bw, bh, STEM_C, "Block $f_0$  (stem)", fontsize=10)
    ax.text(cx - bw/2 - 0.15, stem_y + bh/2, "always active",
            ha="right", va="center", fontsize=7.5, color="#7d3c98", style="italic")

    arrow(cx, stem_y + bh + 0.05, cx, stem_y + bh + 0.65)

    # ── Dashed bracket for repeated section ──────────────────────────────
    rep_y0 = stem_y + bh + 0.55
    rep_y1 = 10.0
    # Left bracket line
    bx = cx - bw/2 - 0.55
    ax.plot([bx, bx], [rep_y0, rep_y1], color="#999", lw=1.0, ls="--")
    ax.plot([bx, bx + 0.2], [rep_y0, rep_y0], color="#999", lw=1.0, ls="--")
    ax.plot([bx, bx + 0.2], [rep_y1, rep_y1], color="#999", lw=1.0, ls="--")
    ax.text(bx - 0.15, (rep_y0 + rep_y1) / 2, "repeat\n$L{-}1$\ntimes",
            ha="right", va="center", fontsize=8, color="#888",
            fontweight="bold", linespacing=1.4)

    # ── h state label ────────────────────────────────────────────────────
    h_y = rep_y0 + 0.25
    ax.text(cx - bw/2 - 0.15, h_y + 0.1, "$h$", ha="right", va="center",
            fontsize=11, fontweight="bold")

    # ── Router (offset to the right) ────────────────────────────────────
    router_cx = cx + bw/2 + 0.2 + rw/2
    router_y = h_y - 0.15
    rounded_box(router_cx, router_y, rw, rh, ROUTER_C,
                "Router $r_l$", fontsize=9)
    ax.text(router_cx, router_y + 0.12, "$\\sigma$(MLP($h$))",
            ha="center", va="center", fontsize=8, color="#555")

    # Arrow h -> router
    arrow(cx + bw/2 * 0.6, h_y + 0.1, router_cx - rw/2 - 0.05, router_y + rh/2,
          color=RED)
    # p_l label
    ax.text(router_cx + rw/2 + 0.2, router_y + rh/2, "$p_l$",
            ha="left", va="center", fontsize=11, fontweight="bold", color=RED)

    # ── Block f_{l+1} ───────────────────────────────────────────────────
    block_y = h_y + 0.9
    rounded_box(cx, block_y, bw, bh, BLOCK_C, "Block $f_{l+1}$", fontsize=10)
    arrow(cx, h_y + 0.3, cx, block_y - 0.05)

    # ── Delta label ─────────────────────────────────────────────────────
    delta_y = block_y + bh + 0.35
    ax.text(cx, delta_y, "$\\Delta^{\\mathrm{attn}},\\; \\Delta^{\\mathrm{ffn}}$",
            ha="center", va="center", fontsize=10)
    arrow(cx, block_y + bh + 0.05, cx, delta_y - 0.18)

    # ── Gate box ────────────────────────────────────────────────────────
    gate_y = delta_y + 0.55
    rounded_box(cx, gate_y, bw + 0.3, bh, GATE_C,
                "$h + (1{-}p_l) \\cdot \\Delta$", fontsize=10)
    arrow(cx, delta_y + 0.18, cx, gate_y - 0.05)

    # Curved arrow: router p_l -> gate
    ax.annotate("", xy=(cx + bw/2 + 0.05, gate_y + bh/2),
                xytext=(router_cx - 0.1, router_y + 0.02),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2,
                                connectionstyle="arc3,rad=-0.3",
                                shrinkA=3, shrinkB=3))

    # ── Output ──────────────────────────────────────────────────────────
    out_y = gate_y + bh + 0.65
    arrow(cx, gate_y + bh + 0.05, cx, out_y - 0.25)
    ax.text(cx, out_y, "Updated $h$", ha="center", va="center",
            fontsize=9, style="italic", color="#555")

    # ── Legend box (bottom-right) ────────────────────────────────────────
    props = dict(boxstyle="round,pad=0.35", facecolor="#fafafa",
                 edgecolor="#bbb", alpha=0.95)
    ax.text(9.8, 0.5,
            "$p_l = 0$: full update\n(standard transformer)\n\n"
            "$p_l = 1$: block skipped\n(state unchanged)",
            fontsize=7.5, va="bottom", ha="right", bbox=props,
            family="serif", linespacing=1.3)

    out = os.path.join(FIGURES_DIR, "architecture_diagram.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Routing Heatmap from trained model
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_extract_routing():
    """Train a TSA model for 2000 steps and extract routing probabilities."""
    # Add project src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from tsa.mlx.data import load_shakespeare, random_batch
    from tsa.mlx.tsa import TSAConfig, TSATransformer

    print("\nTraining TSA model for routing heatmap...")
    train_data, val_data, _test_data, tokenizer = load_shakespeare()
    vocab_size = tokenizer.vocab_size

    cfg = TSAConfig(
        vocab_size=vocab_size, d_model=256, n_heads=8, n_layers=6,
        d_ff=1024, context_len=128, dropout=0.0, depth_reg_weight=0.001,
    )
    mx.random.seed(42)
    model = TSATransformer(cfg)
    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.1, betas=(0.9, 0.95))

    def loss_fn(model, x, y):
        logits = model(x)
        task_loss = mx.mean(
            nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        )
        return task_loss + cfg.depth_reg_weight * model._depth_reg_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Train for 2000 steps
    for step in range(1, 2001):
        x, y = random_batch(train_data, batch_size=64, context_len=128)
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if step % 500 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")

    # Extract routing on a short, readable passage
    sample_text = "To be, or not to be, that is the question."
    token_ids = tokenizer.encode(sample_text)
    chars = list(sample_text)
    seq_len = len(token_ids)

    x = mx.array([token_ids])  # (1, T)
    model.eval()

    # Forward pass, collecting halt probs from each router
    positions = mx.arange(seq_len, dtype=mx.int32)
    h = model.token_emb(x) + model.pos_emb(positions)

    # Stem block (always active)
    zero_halt = mx.zeros((1, seq_len))
    h = model.blocks[0](h, halt_prob=zero_halt)

    halt_probs = []
    for router, block in zip(model.routers, model.blocks[1:]):
        halt_prob = router(h)  # (1, T)
        halt_probs.append(halt_prob[0].tolist())
        h = block(h, halt_prob=halt_prob)

    mx.eval(h)
    halt_probs_np = np.array(halt_probs)  # (n_routers, T)

    return chars, halt_probs_np


def draw_heatmap(chars, halt_probs):
    """Draw routing heatmap: x=token position, y=routing decision (layer gap)."""
    n_routers, seq_len = halt_probs.shape
    active_frac = 1.0 - halt_probs

    # Size proportional to character count
    fig_w = max(4.5, seq_len * 0.18 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, 2.2))

    im = ax.imshow(active_frac, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="nearest")

    # Y axis
    ax.set_yticks(range(n_routers))
    ax.set_yticklabels([f"$r_{l}$" for l in range(n_routers)], fontsize=9)
    ax.set_ylabel("Router", fontsize=10)

    # X axis: large readable characters
    ax.set_xticks(range(seq_len))
    xlabels = []
    for c in chars:
        if c == '\n':
            xlabels.append('\\n')
        elif c == ' ':
            xlabels.append('\u2423')
        else:
            xlabels.append(c)
    ax.set_xticklabels(xlabels, fontsize=10, rotation=0, ha="center",
                       family="monospace")
    ax.tick_params(axis='x', length=0, pad=3)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=12)
    cbar.set_label("Active fraction $(1{-}p_l)$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "routing_heatmap.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating architecture diagram...")
    draw_architecture()

    print("Training model and generating routing heatmap...")
    chars, halt_probs = train_and_extract_routing()
    draw_heatmap(chars, halt_probs)

    print("\nDone.")
