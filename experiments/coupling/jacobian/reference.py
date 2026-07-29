"""Slow, explicit Jacobian implementations for sampled residuals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ...registry import REGISTRIES
from ...types import JacobianData, ResidualData


def _selected_output(render_fn: Callable[[], torch.Tensor], residual_data: ResidualData) -> torch.Tensor:
    rendered = render_fn()
    if rendered.ndim == 3 and rendered.shape[0] in (1, 3, 4):
        rendered = rendered.permute(1, 2, 0)
    if rendered.ndim >= 2:
        rendered = rendered.reshape(-1, rendered.shape[-1])
        rendered = rendered.index_select(0, residual_data.pixel_indices)
    return rendered.reshape(-1)


@REGISTRIES["jacobian"].register("repeated_vjp")
class RepeatedVJPJacobian:
    """Build rows with first-order VJPs; no rasterizer double backward needed."""

    name = "repeated_vjp"

    def compute(
        self,
        render_fn: Callable[[], torch.Tensor],
        parameter: torch.Tensor,
        gaussian_ids: tuple[int, ...],
        residual_data: ResidualData,
        config: dict[str, Any],
    ) -> JacobianData:
        outputs = _selected_output(render_fn, residual_data)
        if outputs.numel() != residual_data.residual.numel():
            raise ValueError(f"rendered sample size {outputs.numel()} differs from residual size {residual_data.residual.numel()}")
        ids = torch.tensor(gaussian_ids, dtype=torch.long, device=parameter.device)
        rows: list[torch.Tensor] = []
        for row_id in range(outputs.numel()):
            gradient = torch.autograd.grad(
                outputs[row_id], parameter, retain_graph=row_id + 1 < outputs.numel(),
                create_graph=False, allow_unused=False,
            )[0]
            rows.append(gradient.index_select(0, ids).reshape(-1))
        jacobian = torch.stack(rows) if rows else parameter.new_zeros((0, ids.numel() * parameter.shape[-1]))
        residual = residual_data.residual.detach().to(jacobian)
        return JacobianData(
            gaussian_ids=gaussian_ids,
            jacobian=jacobian,
            residual=residual,
            gradient=jacobian.T @ residual,
            metadata={"provider": self.name, "vjp_count": outputs.numel()},
        )


@REGISTRIES["jacobian"].register("finite_difference")
class FiniteDifferenceJacobian:
    name = "finite_difference"

    def compute(
        self,
        render_fn: Callable[[], torch.Tensor],
        parameter: torch.Tensor,
        gaussian_ids: tuple[int, ...],
        residual_data: ResidualData,
        config: dict[str, Any],
    ) -> JacobianData:
        epsilon = float(config.get("finite_difference_epsilon", 1.0e-4))
        columns: list[torch.Tensor] = []
        with torch.no_grad():
            for gaussian_id in gaussian_ids:
                for dimension in range(parameter.shape[-1]):
                    original = parameter[gaussian_id, dimension].clone()
                    parameter[gaussian_id, dimension] = original + epsilon
                    plus = _selected_output(render_fn, residual_data).detach()
                    parameter[gaussian_id, dimension] = original - epsilon
                    minus = _selected_output(render_fn, residual_data).detach()
                    parameter[gaussian_id, dimension] = original
                    columns.append((plus - minus) / (2.0 * epsilon))
        jacobian = torch.stack(columns, dim=1) if columns else parameter.new_zeros((residual_data.residual.numel(), 0))
        residual = residual_data.residual.detach().to(jacobian)
        return JacobianData(
            gaussian_ids=gaussian_ids,
            jacobian=jacobian,
            residual=residual,
            gradient=jacobian.T @ residual,
            metadata={"provider": self.name, "epsilon": epsilon, "render_count": 2 * len(columns)},
        )


def jacobian_relative_error(reference: torch.Tensor, candidate: torch.Tensor, epsilon: float = 1.0e-12) -> float:
    numerator = torch.linalg.vector_norm(reference - candidate)
    denominator = torch.linalg.vector_norm(reference).clamp_min(epsilon)
    return float((numerator / denominator).item())
