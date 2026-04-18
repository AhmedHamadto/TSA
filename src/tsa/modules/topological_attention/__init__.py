# Module 1: Topological Sparse Attention (TSA)
# Hypothesis: variable compute per token (content-difficulty routing)
# improves efficiency without accuracy loss vs fixed-depth baseline.

from tsa.modules.topological_attention.graph_router import TokenRouter
from tsa.modules.topological_attention.sparse_attention import SparseTransformerBlock
from tsa.modules.topological_attention.variable_depth import TSAConfig, TSATransformer

__all__ = ["TokenRouter", "SparseTransformerBlock", "TSAConfig", "TSATransformer"]
