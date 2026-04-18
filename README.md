# Token-Selective Attention (TSA)

[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](https://arxiv.org/abs/TBD)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-3776ab.svg)](https://www.python.org/downloads/)

Code for *"Adaptive Computation Depth via Learned Token Routing in Transformers"*.

Standard transformers apply the same number of layers to every token. TSA is a learned per-token gate on residual updates between consecutive blocks: a lightweight 2-layer MLP router produces a continuous halting probability `p`, and the residual is soft-gated by `h ← h + (1 − p) · Δ`. The mechanism is end-to-end differentiable, adds **1.7% parameter overhead**, and makes no changes to the base architecture. Notably, TSA learns difficulty-proportional routing **without any depth pressure**: even at λ=0, the task-loss gradient alone drives the router to skip 20% of token-layer operations.

## Results Summary

TSA adds **1.7% parameter overhead** and saves **14–23% of token-layer operations (TLOps)** on character-level language modeling at **<0.5% quality loss**.

| Benchmark | Model | Val Loss / Accuracy | α (active frac) | TLOps saved |
|-----------|-------|---------------------|-----------------|-------------|
| Synthetic — Copy (L=6) | TSA | 1.0000 acc | 0.341 | **54.9%** |
| Synthetic — Sort (L=6) | TSA | 0.9878 acc (Baseline 0.9915) | 0.730 | 22.5% |
| Tiny-Shakespeare (d=256, L=6) | TSA | 1.4482 nats (Baseline 1.4422, +0.4%) | 0.726 | 22.8% |
| enwik8 (d=256, L=6, MLX M1 Pro) | TSA | 1.2774 nats (Baseline 1.2826, −0.4%) | 0.833 | 13.9% |

TLOps saved = 1 − (1 + (L−1)α)/L, which conservatively includes the mandatory stem block.

### Central finding (λ=0)

Without any depth regularisation (λ=0), the router still learns to skip ~**20% of TLOps** (α=0.755) from the task-loss gradient alone. The gating multiplication `h ← h + (1−p)·Δ` provides an intrinsic learning signal: when a layer's residual update is noisy or redundant, the gradient favours increasing p to attenuate it. Routing emerges from task loss — not from the regulariser.

### Ablations

- **λ sweep** on Shakespeare: TSA is robust across λ ∈ [0, 0.1]. Best Pareto point at λ=0.05 — 50% TLOps saved, <0.5% val loss degradation.
- **Early exit comparison** at matched α≈0.726: TSA val_loss 1.4482 vs EE 1.4586 — TSA is **0.7% better** at identical efficiency, while also saving training compute (EE uses full compute every step).
- **Wall-clock** (MLX, Apple M1 Pro, batch=64): sparse-TSA reaches 1.023× throughput at α=0.726, break-even at α=0.83, 1.246× at α=0.10. Soft-gating overhead ~1% flat.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install (editable mode)
pip install -e .

# For MLX experiments (Apple Silicon only)
pip install -e ".[mlx]"

# For development (tests)
pip install -e ".[dev]"
```

## Project Structure

```
src/tsa/
  core/             # Base model, trainer, metrics
  benchmarks/       # Baseline transformer, toy tasks, Shakespeare, enwik8
  modules/          # TSA implementation: TokenRouter, SparseTransformerBlock, TSATransformer
  mlx/              # MLX-native port (Apple Silicon)
scripts/            # Experiment scripts (TSA phases + ablations)
tests/              # 120 tests
docs/paper/         # LaTeX source + figures
docs/results/       # Detailed experiment results per phase
```

## Running Experiments

```bash
# Phase 1: TSA vs Baseline on toy tasks
python scripts/compare_baseline_tsa.py

# Phase 6: Character-level language modeling (Shakespeare)
python scripts/phase6_language.py

# Phase 7: enwik8 cross-dataset validation (MLX, Apple Silicon)
python scripts/phase7_enwik8_mlx.py

# Ablations
python scripts/ablation_lambda_sweep.py       # lambda sensitivity (~90 min)
python scripts/benchmark_wallclock.py          # Wall-clock throughput (~15 min)
python scripts/ablation_early_exit.py          # Early exit comparison (~25 min)
```

## Tests

```bash
pytest tests/ -q
```

## Citation

If you use this work, please cite:

```bibtex
@article{abdelmuniem2026tsa,
  title={Adaptive Computation Depth via Learned Token Routing in Transformers},
  author={Abdelmuniem Abdalla Mohammed, Ahmed},
  journal={arXiv preprint arXiv:TBD},
  year={2026}
}
```

See also `CITATION.cff` at the repo root.

## License

MIT
