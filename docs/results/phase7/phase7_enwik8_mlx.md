# Phase 7 Results — enwik8 Character-Level LM (MLX)

**Date:** 2026-04-01
**Hardware:** Apple MacBook Pro (M1 Pro, 16 GB)
**MLX version:** 0.31.1
**Corpus:** enwik8 (first 10^8 bytes of English Wikipedia, raw XML)
**Vocab size:** 6064
**Scale:** 6,350,848 params (d=256, L=6, h=8, ctx=256)
**Steps:** 5,000 per condition
**Total wall time:** 98.6 min

---

## Summary Table

| Condition | Params | Val Loss | BPC | →2.5 | →2.0 | →1.8 | Active Frac | ms/step | Time |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 6,350,848 | **1.2826** | **1.8504** | 500 | 750 | 1000 | 1.000 | 585 | 48.7m |
| TSA | 6,433,413 | **1.2774** | **1.8429** | 500 | 750 | 1000 | 0.833 | 599 | 49.9m |

---

## Key Findings

### Finding 1: TSA on enwik8
TSA mean active fraction = 0.8326 → **14.0% of token-layer computations skipped**.
Quality: val loss diff = -0.41% vs Baseline.

**Phase 6 Shakespeare comparison:**
- Shakespeare: 0.726 active → 22.8% ops saved, +0.4% quality loss
- enwik8: 0.833 active → 14.0% ops saved, -0.4% quality diff

### Finding 2: Cross-dataset generalisation

~ ROUTING ACTIVE but savings below 15% threshold

---

## Compute Efficiency Detail

```
Formula: Δ = 1 − (1 + (L−1) × α) / L  (all-layers, includes mandatory stem)
Where α = mean active fraction, L = 6 layers

  Baseline                          α=1.000  eff_layers=6.00  ops_saved=0.0%
  TSA                               α=0.833  eff_layers=5.16  ops_saved=14.0%
```

---

## Notes on enwik8 vs Shakespeare

- Vocab size: enwik8 has Unicode chars from non-English articles → larger vocab than Shakespeare's 65
- Context length: 256 (vs 128 for Shakespeare) to handle longer Wikipedia sentences
- Training: 5,000 steps × batch 64 × context 256 = 81.9M tokens seen (covers ~91% of 90M training chars once)
- Raw XML kept (Decision D032): standard enwik8 benchmark, stripping would invalidate BPC comparison

## Decisions Logged

- D032: Raw XML kept (standard enwik8)
- D033: Originally 10K steps for PyTorch; MLX Phase 7 uses 5K (D045)
- D034: context_len=256
- D035: eval_interval=250 (5% frequency)
- D036: loss_thresholds=[2.5, 2.0, 1.8]
- D045: Use MLX for Phase 7; 5K steps based on 600ms/step timing on M1 Pro

## Plots

- `docs/paper/figures/phase7_val_loss_vs_steps.pdf` — val loss vs training step
- `docs/paper/figures/phase7_val_loss_vs_compute.pdf` — val loss vs cumulative token-layer ops