"""Parameter block implementations.  Position is the first supported block."""

from __future__ import annotations

from typing import Any

import torch

from ...registry import REGISTRIES


@REGISTRIES["parameter_block"].register("position")
class PositionBlock:
    name = "position"
    dimension_per_gaussian = 3

    def gather(self, gaussian_model: Any, indices: torch.Tensor) -> torch.Tensor:
        return gaussian_model.means.index_select(0, indices)

    @torch.no_grad()
    def apply_update(self, gaussian_model: Any, indices: torch.Tensor, delta: torch.Tensor) -> None:
        if delta.shape != (indices.numel(), 3):
            raise ValueError(f"position update must have shape ({indices.numel()}, 3), got {tuple(delta.shape)}")
        if not torch.isfinite(delta).all():
            raise FloatingPointError("position update contains NaN or Inf")
        gaussian_model.means.index_add_(0, indices, delta.to(gaussian_model.means))

    def project_or_clamp_update(self, state: torch.Tensor, delta: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
        absolute = config.get("max_step_norm")
        normalized = config.get("max_step_scene_fraction")
        scene_scale = float(config.get("scene_scale", 1.0))
        limits = [float(value) for value in (absolute, None if normalized is None else normalized * scene_scale) if value is not None]
        if not limits:
            return delta
        limit = min(limits)
        norms = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        return delta * (limit / norms.clamp_min(limit)).clamp_max(1.0)

    def snapshot(self, gaussian_model: Any, indices: torch.Tensor) -> torch.Tensor:
        return self.gather(gaussian_model, indices).detach().clone()

    @torch.no_grad()
    def restore(self, gaussian_model: Any, indices: torch.Tensor, snapshot: torch.Tensor) -> None:
        gaussian_model.means.index_copy_(0, indices, snapshot.to(gaussian_model.means))


class _UnsupportedBlock:
    dimension_per_gaussian = 0

    def __init__(self, **_: Any) -> None:
        raise NotImplementedError(f"parameter block '{self.name}' is registered as an extension point but is not implemented")


@REGISTRIES["parameter_block"].register("scale_rotation")
class ScaleRotationBlock(_UnsupportedBlock):
    name = "scale_rotation"


@REGISTRIES["parameter_block"].register("opacity")
class OpacityBlock(_UnsupportedBlock):
    name = "opacity"


@REGISTRIES["parameter_block"].register("dc_color")
class DCColorBlock(_UnsupportedBlock):
    name = "dc_color"


@REGISTRIES["parameter_block"].register("opacity_dc")
class OpacityDCBlock(_UnsupportedBlock):
    name = "opacity_dc"


@REGISTRIES["parameter_block"].register("sh")
class SHBlock(_UnsupportedBlock):
    name = "sh"


@REGISTRIES["parameter_block"].register("composite")
class CompositeParameterBlock(_UnsupportedBlock):
    name = "composite"
