"""Registered first- and second-order update strategies."""

from .strategies import AdamAdapter, GroupGaussNewton, GroupLevenbergMarquardt, PerGaussianGaussNewton, PerGaussianLevenbergMarquardt

__all__ = ["AdamAdapter", "PerGaussianGaussNewton", "PerGaussianLevenbergMarquardt", "GroupGaussNewton", "GroupLevenbergMarquardt"]
