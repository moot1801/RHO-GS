"""Dense reference GN/LM solvers for small sampled Gaussian groups."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from ...registry import REGISTRIES
from ...types import CurvatureData, Group, JacobianData, ResidualData, UpdateProposal


def _dtype(config: dict[str, Any], fallback: torch.dtype) -> torch.dtype:
    name = str(config.get("diagnostic_dtype", "float64"))
    return {"float32": torch.float32, "float64": torch.float64}.get(name, fallback)


def _damping_matrix(hessian: torch.Tensor, mode: str, block_dimension: int) -> torch.Tensor:
    if mode == "identity":
        return torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device)
    if mode == "diagonal":
        return torch.diag(torch.diagonal(hessian).clamp_min(torch.finfo(hessian.dtype).eps))
    if mode == "block_diagonal":
        result = torch.zeros_like(hessian)
        for start in range(0, hessian.shape[0], block_dimension):
            section = slice(start, start + block_dimension)
            result[section, section] = hessian[section, section]
        return result
    raise ValueError(f"unknown damping mode: {mode}")


def _solve(hessian: torch.Tensor, gradient: torch.Tensor, damping: float, config: dict[str, Any], block_dimension: int) -> tuple[torch.Tensor, str, float]:
    dtype = _dtype(config, hessian.dtype)
    hessian = hessian.to(dtype)
    gradient = gradient.to(dtype)
    damping_matrix = _damping_matrix(hessian, str(config.get("damping_mode", "identity")), block_dimension)
    system = 0.5 * (hessian + hessian.T) + damping * damping_matrix
    condition = float(torch.linalg.cond(system).item())
    factor, info = torch.linalg.cholesky_ex(system)
    if int(info.max().item()) == 0:
        delta = torch.cholesky_solve((-gradient).unsqueeze(-1), factor).squeeze(-1)
        status = "cholesky"
    else:
        try:
            delta = torch.linalg.solve(system, -gradient)
            status = "solve_fallback"
        except torch.linalg.LinAlgError:
            delta = torch.linalg.lstsq(system, -gradient.unsqueeze(-1)).solution.squeeze(-1)
            status = "lstsq_fallback"
    return delta.to(hessian.dtype), status, condition


def _proposal(name: str, group: Group, curvature: CurvatureData, damping: float, config: dict[str, Any], block_dimension: int) -> UpdateProposal:
    start = perf_counter()
    delta, status, condition = _solve(curvature.hessian, curvature.gradient, damping, config, block_dimension)
    predicted = -(torch.dot(curvature.gradient.to(delta), delta) + 0.5 * torch.dot(delta, curvature.hessian.to(delta) @ delta))
    threshold = float(config.get("condition_number_threshold", float("inf")))
    finite = bool(torch.isfinite(delta).all()) and bool(torch.isfinite(torch.tensor(condition)))
    return UpdateProposal(
        gaussian_ids=group.member_gaussian_ids,
        delta=delta.reshape(len(group.member_gaussian_ids), block_dimension),
        predicted_reduction=float(predicted.item()),
        damping=damping,
        condition_estimate=condition,
        solver_iterations=1,
        converged=finite and condition <= threshold,
        numerical_status=status if finite else "non_finite",
        solve_time_ms=(perf_counter() - start) * 1000.0,
        diagnostics={"solver": name, "linear_residual_norm": float(torch.linalg.vector_norm((curvature.hessian.to(delta) + damping * _damping_matrix(curvature.hessian.to(delta), str(config.get('damping_mode', 'identity')), block_dimension)) @ delta + curvature.gradient.to(delta)).item())},
    )


class _GroupSolver:
    damping = 0.0
    name = "group_gn"

    def propose_update(self, group: Group, parameter_block: Any, residual_data: ResidualData, jacobian_data: JacobianData, curvature_data: CurvatureData, config: dict[str, Any]) -> UpdateProposal:
        damping = float(config.get("initial_damping", self.damping)) if self.damping else 0.0
        return _proposal(self.name, group, curvature_data, damping, config, parameter_block.dimension_per_gaussian)


@REGISTRIES["solver"].register("group_gn")
class GroupGaussNewton(_GroupSolver):
    name = "group_gn"


@REGISTRIES["solver"].register("group_lm")
class GroupLevenbergMarquardt(_GroupSolver):
    name = "group_lm"
    damping = 1.0e-3


class _PerGaussianSolver(_GroupSolver):
    def propose_update(self, group: Group, parameter_block: Any, residual_data: ResidualData, jacobian_data: JacobianData, curvature_data: CurvatureData, config: dict[str, Any]) -> UpdateProposal:
        dimension = parameter_block.dimension_per_gaussian
        deltas: list[torch.Tensor] = []
        conditions: list[float] = []
        statuses: list[str] = []
        predicted = 0.0
        start_time = perf_counter()
        damping = float(config.get("initial_damping", self.damping)) if self.damping else 0.0
        for block_id in range(len(group.member_gaussian_ids)):
            section = slice(block_id * dimension, (block_id + 1) * dimension)
            hessian = curvature_data.hessian[section, section]
            gradient = curvature_data.gradient[section]
            delta, status, condition = _solve(hessian, gradient, damping, config, dimension)
            deltas.append(delta)
            statuses.append(status)
            conditions.append(condition)
            predicted += float((-(torch.dot(gradient.to(delta), delta) + 0.5 * torch.dot(delta, hessian.to(delta) @ delta))).item())
        combined = torch.stack(deltas) if deltas else curvature_data.gradient.new_zeros((0, dimension))
        finite = bool(torch.isfinite(combined).all())
        return UpdateProposal(
            gaussian_ids=group.member_gaussian_ids,
            delta=combined,
            predicted_reduction=predicted,
            damping=damping,
            condition_estimate=max(conditions, default=0.0),
            solver_iterations=len(deltas),
            converged=finite and max(conditions, default=0.0) <= float(config.get("condition_number_threshold", float("inf"))),
            numerical_status=";".join(statuses) if finite else "non_finite",
            solve_time_ms=(perf_counter() - start_time) * 1000.0,
            diagnostics={"solver": self.name, "block_conditions": conditions},
        )


@REGISTRIES["solver"].register("per_gaussian_gn")
class PerGaussianGaussNewton(_PerGaussianSolver):
    name = "per_gaussian_gn"


@REGISTRIES["solver"].register("per_gaussian_lm")
class PerGaussianLevenbergMarquardt(_PerGaussianSolver):
    name = "per_gaussian_lm"
    damping = 1.0e-3


@REGISTRIES["solver"].register("adam")
class AdamAdapter:
    """Counterfactual Adam proposal using restored PyTorch optimizer state."""

    name = "adam"

    def propose_update(self, group: Group, parameter_block: Any, residual_data: ResidualData, jacobian_data: JacobianData, curvature_data: CurvatureData, config: dict[str, Any]) -> UpdateProposal:
        optimizer = config.get("_optimizer")
        parameter = config.get("_parameter")
        allow_cold = bool(config.get("allow_cold_start", False))
        if optimizer is None or parameter is None:
            if not allow_cold:
                raise RuntimeError("AdamAdapter requires restored _optimizer and _parameter; set allow_cold_start=true to mark a cold-start comparison")
            state: dict[str, Any] = {}
            param_group = {"lr": float(config.get("learning_rate", 1.0e-4)), "betas": (0.9, 0.999), "eps": 1.0e-15, "weight_decay": 0.0}
            parameter = torch.empty((max(group.member_gaussian_ids) + 1, parameter_block.dimension_per_gaussian), device=curvature_data.gradient.device)
            cold = True
        else:
            state = optimizer.state.get(parameter, {})
            param_group = next(item for item in optimizer.param_groups if any(candidate is parameter for candidate in item["params"]))
            cold = not bool(state)
        if cold and not allow_cold:
            raise RuntimeError("Adam optimizer state is empty; use a portable state or explicitly set solver.allow_cold_start=true")
        if float(param_group.get("weight_decay", 0.0)) != 0.0 or bool(param_group.get("amsgrad", False)):
            raise NotImplementedError("AdamAdapter reference currently supports only weight_decay=0 and amsgrad=false")
        ids = torch.tensor(group.member_gaussian_ids, dtype=torch.long, device=curvature_data.gradient.device)
        gradient = curvature_data.gradient.reshape(len(group.member_gaussian_ids), -1)
        beta1, beta2 = param_group.get("betas", (0.9, 0.999))
        step_value = state.get("step", 0)
        step = int(step_value.item() if isinstance(step_value, torch.Tensor) else step_value) + 1
        exp_avg = state.get("exp_avg", torch.zeros_like(parameter)).index_select(0, ids).to(gradient)
        exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(parameter)).index_select(0, ids).to(gradient)
        exp_avg = beta1 * exp_avg + (1.0 - beta1) * gradient
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * gradient.square()
        corrected_avg = exp_avg / (1.0 - beta1 ** step)
        corrected_sq = exp_avg_sq / (1.0 - beta2 ** step)
        delta = -float(param_group["lr"]) * corrected_avg / (corrected_sq.sqrt() + float(param_group.get("eps", 1.0e-8)))
        predicted = -(torch.dot(curvature_data.gradient.to(delta), delta.reshape(-1)) + 0.5 * torch.dot(delta.reshape(-1), curvature_data.hessian.to(delta) @ delta.reshape(-1)))
        return UpdateProposal(
            gaussian_ids=group.member_gaussian_ids,
            delta=delta,
            predicted_reduction=float(predicted.item()),
            damping=0.0,
            condition_estimate=float("nan"),
            solver_iterations=1,
            converged=bool(torch.isfinite(delta).all()),
            numerical_status="cold_start" if cold else "restored_state",
            solve_time_ms=0.0,
            diagnostics={"solver": self.name, "optimizer_step": step, "cold_start": cold},
        )
