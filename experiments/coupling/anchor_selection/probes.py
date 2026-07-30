"""Low-cost randomized probes for sampled position-Jacobian block energy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ...types import ResidualData


def _selected_output(render_fn: Callable[[], torch.Tensor], residual_data: ResidualData) -> torch.Tensor:
    rendered = render_fn()
    if rendered.ndim == 3 and rendered.shape[0] in (1, 3, 4):
        rendered = rendered.permute(1, 2, 0)
    if rendered.ndim >= 2:
        rendered = rendered.reshape(-1, rendered.shape[-1])
        rendered = rendered.index_select(0, residual_data.pixel_indices)
    return rendered.reshape(-1)


def estimate_position_jacobian_energy(
    render_fn: Callable[[], torch.Tensor],
    parameter: torch.Tensor,
    residual_data: ResidualData,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate each Gaussian's ``||J_i||_F^2`` with Rademacher VJPs.

    For independent Rademacher vectors ``z``, ``E[||J_i^T z||^2]`` equals
    ``||J_i||_F^2``.  The probe is used only to define an eligible population;
    exact group Jacobians are still computed by the configured provider.
    """

    probe_count = max(1, int(config.get("probe_count", 4)))
    seed = int(config.get("probe_seed", config.get("seed", 0)))
    outputs = _selected_output(render_fn, residual_data)
    if outputs.numel() != residual_data.residual.numel():
        raise ValueError(f"probe output size {outputs.numel()} differs from residual size {residual_data.residual.numel()}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.zeros(parameter.shape[0], dtype=torch.float64, device=parameter.device)
    for probe_id in range(probe_count):
        signs = torch.randint(0, 2, (outputs.numel(),), generator=generator, device="cpu", dtype=torch.int8)
        vector = signs.to(device=outputs.device, dtype=outputs.dtype).mul_(2).sub_(1)
        gradient = torch.autograd.grad(
            outputs,
            parameter,
            grad_outputs=vector,
            retain_graph=probe_id + 1 < probe_count,
            create_graph=False,
            allow_unused=False,
        )[0]
        scores.add_(gradient.detach().to(torch.float64).square().sum(dim=-1))
    scores.div_(probe_count)
    return scores, {
        "estimator": "rademacher_vjp_mean_squared_block_norm",
        "estimated_quantity": "position_jacobian_block_frobenius_norm_squared",
        "probe_count": probe_count,
        "probe_seed": seed,
        "residual_scalar_count": int(outputs.numel()),
    }
