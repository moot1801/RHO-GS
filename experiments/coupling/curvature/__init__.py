"""Gauss-Newton block assembly and coupling diagnostics."""

from .assembler import (
    GaussNewtonAssembler,
    coupling_capture_ratio,
    edge_budget_oracle_capture_metrics,
    edge_capture_metrics,
    group_normalized_coupling,
    normalized_pairwise_coupling,
)

__all__ = [
    "GaussNewtonAssembler",
    "normalized_pairwise_coupling",
    "group_normalized_coupling",
    "coupling_capture_ratio",
    "edge_capture_metrics",
    "edge_budget_oracle_capture_metrics",
]
