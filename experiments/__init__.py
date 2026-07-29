"""Extensible reference experiments for inter-Gaussian coupling analysis.

The package deliberately does not import NeRFICG or the CUDA rasterizer at
module import time.  This keeps configuration, grouping, solvers, and unit
tests usable on CPU-only machines.
"""

from .registry import REGISTRIES, Registry

__all__ = ["REGISTRIES", "Registry"]
