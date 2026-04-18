# MLX Port Verification Report

**Date:** 2026-04-01
**Config:** d_model=128, L=4, h=4, steps=500

## Results

| Condition | Final Val Loss | Final BPC | Mean Active Frac |
|-----------|---------------|-----------|-----------------|
| PyTorch Baseline | 2.3309 | 3.3628 | 1.000 |
| PyTorch TSA | 2.3385 | 3.3737 | 0.740 |
| MLX Baseline | 2.3339 | 3.3671 | 1.000 |
| MLX TSA | 2.3445 | 3.3824 | 0.734 |

## Convergence Comparison

- Baseline: PyTorch=2.3309, MLX=2.3339, diff=0.1%
- TSA: PyTorch=2.3385, MLX=2.3445, diff=0.3%

## Notes

- Different RNG (PyTorch vs MLX) means step-by-step curves differ; final loss is the meaningful comparison.
- Verification criterion: final val loss within 5% between frameworks.
- MLX uses uniform AdamW weight_decay (no selective no-decay for biases/embeddings). PyTorch version excludes biases and embeddings from weight decay.
- TSA active_frac should be < 0.95 in both frameworks, confirming the router learns.