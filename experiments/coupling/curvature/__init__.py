"""Gauss-Newton block assembly and coupling diagnostics."""

from .assembler import GaussNewtonAssembler, coupling_capture_ratio, group_normalized_coupling, normalized_pairwise_coupling

__all__ = ["GaussNewtonAssembler", "normalized_pairwise_coupling", "group_normalized_coupling", "coupling_capture_ratio"]
