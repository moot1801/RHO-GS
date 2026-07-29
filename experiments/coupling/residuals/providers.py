"""Reference residual definitions used by Gauss-Newton experiments."""

from __future__ import annotations

from typing import Any

import torch

from ...registry import REGISTRIES
from ...types import ResidualData


def _pixels(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 1:
        return image.reshape(-1, 1)
    if image.ndim == 2:
        return image.reshape(-1, 1)
    if image.ndim != 3:
        raise ValueError(f"expected image with 2 or 3 dimensions, got {tuple(image.shape)}")
    if image.shape[0] in (1, 3, 4):
        return image.permute(1, 2, 0).reshape(-1, image.shape[0])
    return image.reshape(-1, image.shape[-1])


class _RGBL2Base:
    name = "rgb_l2"

    def build(self, prediction: torch.Tensor, target: torch.Tensor, pixel_indices: torch.Tensor | None = None) -> ResidualData:
        prediction_pixels = _pixels(prediction)
        target_pixels = _pixels(target).to(prediction_pixels)
        if prediction_pixels.shape != target_pixels.shape:
            raise ValueError(f"prediction/target shapes differ: {prediction_pixels.shape} vs {target_pixels.shape}")
        if pixel_indices is None:
            pixel_indices = torch.arange(prediction_pixels.shape[0], device=prediction_pixels.device)
        selected_prediction = prediction_pixels.index_select(0, pixel_indices)
        selected_target = target_pixels.index_select(0, pixel_indices)
        residual = (selected_prediction - selected_target).reshape(-1)
        return ResidualData(
            residual=residual,
            prediction=selected_prediction,
            target=selected_target,
            pixel_indices=pixel_indices,
            loss=0.5 * torch.dot(residual, residual),
            metadata={"definition": "0.5_sum_squared_rgb_residual", "channels": selected_prediction.shape[-1]},
        )


@REGISTRIES["residual"].register("rgb_l2")
class RGBL2Residual(_RGBL2Base):
    name = "rgb_l2"


@REGISTRIES["residual"].register("rgb_l2_sampled")
class RGBL2SampledResidual(_RGBL2Base):
    name = "rgb_l2_sampled"

    def build(self, prediction: torch.Tensor, target: torch.Tensor, pixel_indices: torch.Tensor | None = None) -> ResidualData:
        if pixel_indices is None:
            raise ValueError("rgb_l2_sampled requires explicit pixel_indices")
        return super().build(prediction, target, pixel_indices)


class _UnsupportedResidual:
    def __init__(self, **_: Any) -> None:
        raise NotImplementedError(f"residual '{self.name}' is registered as an extension point but is not implemented")


for _name in ("sqrt_l1_dssim", "mse_dssim_diagonal", "robust_rgb_irls"):
    REGISTRIES["residual"].register(_name)(type(f"{_name.title().replace('_', '')}Residual", (_UnsupportedResidual,), {"name": _name}))
