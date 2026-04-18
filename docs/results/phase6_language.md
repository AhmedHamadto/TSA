# Phase 6 Results — Character-Level Language Modeling

**Date:** 2026-04-01
**Scale:** ~5M params (d_model=256, n_layers=6, context_len=128)
**Corpus:** Tiny-Shakespeare (~1M chars, char-level, vocab=65)
**Steps:** 5,000 per condition
**Hardware:** Apple MPS

---

## Summary Table

| Condition | Params | Val Loss | BPC | →3.0 | →2.5 | →2.0 | Active Frac | Time |
|---|---|---|---|---|---|---|---|---|
| Baseline (static) | 4,782,336 | **1.4422** | **2.0806** | 250 | 250 | 500 | 1.000 | 983s |
| TSA-only (static) | 4,864,901 | 1.4482 | 2.0893 | 250 | 250 | 500 | **0.726** | 1070s |
| SGC+Baseline (loss-wt) | 4,782,336 | 1.5111 | 2.1801 | 250 | 250 | 500 | 1.000 | 993s |
| Genesis (TSA+SGC) | 4,864,901 | 1.5044 | 2.1704 | 250 | 250 | 500 | **0.718** | 1076s |

---

## Key Findings

### Finding 1: TSA compute savings confirmed (✓ positive)
TSA-only mean active fraction = 0.726 → **~27.4% of token-layer computations skipped**.
Final val loss difference vs baseline: +0.006 nats (0.4% worse) — effectively neutral.
TSA achieves compute efficiency on language without hurting quality.

### Finding 2: SGC does NOT transfer to language (✗ negative — valuable)
All four conditions hit the same threshold steps (250/250/500).
SGC+Baseline final val_loss = 1.5111 — **worse** than static baseline 1.4422 (+4.8%).
Genesis final val_loss = 1.5044 — worse than TSA-only 1.4482 (+3.9%).

**Why SGC fails on language:**
The toy-task SGC worked because it had discrete, identifiable tasks (copy/reverse/sort) with clear mastery gaps. The Generator learned a policy over a finite, stable action space.

Loss-weighted prioritized replay on language chunks has no such structure:
- Every chunk is its own "task" with different vocabulary, style, and difficulty
- A chunk seen once may never repeat — there's no stable per-chunk difficulty to learn
- High-loss chunks are often ones with rare character sequences, proper nouns, or unusual meter
- Sampling them more often creates a distribution mismatch vs the true corpus distribution
- The model overfits the "hard" chunks at the expense of fluency on the corpus as a whole

The signal used in toy SGC (target_acc − current_acc) was a policy over *which task to train more*. Loss-weighted replay is conceptually similar but the "tasks" are not stable: chunk difficulty is highly non-stationary and doesn't represent a coherent mastery gap.

### Finding 3: TSA + SGC do not compound on language
Unlike Phase 5b where Genesis+LabeledSGC was 40% faster than Baseline+SGC, here Genesis slightly underperforms TSA-only. The TSA compute savings (+27%) remain, but the SGC degradation cancels any benefit.

---

## Compute Efficiency Detail

```
Condition         Effective layers (mean)   Token-layer ops vs Baseline
Baseline          6.0 (1.000 active)        1.00x  (reference)
TSA-only          4.356 (0.726 active)      0.726x (-27.4% ops)
SGC+Baseline      6.0 (1.000 active)        1.00x
Genesis           4.308 (0.718 active)      0.718x (-28.2% ops)
```

With 5M params, TSA's router learns to halt ~27% of token-layer computations while preserving final loss quality within 0.4% of baseline. This is a real inference efficiency win on language, even though it provides no training convergence speedup.

---

## Interpretation

**TSA on language:** The positive story holds. The router generalizes from toy tasks to real language — it learns which token positions need less processing. The 27% ops savings with <0.5% quality loss is practically meaningful. For a deployed model, this translates directly to inference FLOPs reduction.

**SGC on language:** The negative result is more informative than a positive one would have been. It tells us:
1. Curriculum learning requires a meaningful task structure to be exploited
2. Loss-weighted replay without task decomposition hurts distribution coverage
3. The SGC design is fundamentally tied to the discrete-task framing

Future directions for language curriculum (out of scope for this project):
- Skill-slotted document chunks (e.g., dialogue vs narration vs verse)
- BPE-level difficulty from subword entropy
- Online hard example mining with a small priority queue (subset, not all chunks)

**What this means for the Genesis story:**
Genesis = TSA + CKR + SGC. On language, CKR is parked (no task_ids) and SGC hurts. So "Genesis on language" degrades to TSA-only + broken SGC. The compounding result from Phase 5b (where all three mechanisms interact correctly) doesn't carry over to language in this formulation. The architectural paper story changes:

> *TSA generalizes to language and delivers compute savings. SGC is powerful but requires discrete task structure. CKR is the most impactful mechanism when task identity is available. The compounding effect is real but task-structure-dependent.*

---

## Plots

- `docs/results/phase6/val_loss_vs_steps.png` — all 4 conditions, loss vs training step
- `docs/results/phase6/val_loss_vs_compute.png` — loss vs cumulative token-layer ops (TSA curves shift left)

---

## Decisions Logged
- D025: CKR parked for language (no natural task_ids)
- D026: Removed padding_idx=0 for character-level models
- D027: context_len=128 for Phase 6
- D028: Language SGC simplified to loss-weighted prioritized replay
- D029: TSA compute savings confirmed on language (~27%)
- D030: SGC does not transfer to language — negative result

---

## Next: Scaling Teaser (10M params)

Requested in Telegram msg 64. Run `scripts/phase6_language.py --scale 10m`.
Expected: TSA savings stable or growing (more depth → more routing opportunities).
SGC: still negative (structural issue, not a scale issue).
Question: does TSA quality gap widen or shrink at 10M?
