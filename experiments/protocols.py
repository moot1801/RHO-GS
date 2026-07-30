"""Protocols for extension points; concrete implementations register by name."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from .types import AnchorSet, CurvatureData, GaussianUniverse, Group, GroupSet, JacobianData, RenderState, ResidualData, UpdateProposal


class UniverseSelectionStrategy(Protocol):
    name: str

    def needs_contribution_scores(self, config: dict[str, Any]) -> bool: ...

    def build(
        self,
        gaussian_state: Any,
        render_state: RenderState,
        view: Any,
        config: dict[str, Any],
        contribution_scores: torch.Tensor | None = None,
    ) -> GaussianUniverse | None: ...


class AnchorSelectionStrategy(Protocol):
    name: str
    requires_contribution_scores: bool

    def select(
        self,
        gaussian_state: Any,
        render_state: RenderState,
        view: Any,
        config: dict[str, Any],
        contribution_scores: torch.Tensor | None = None,
        universe: GaussianUniverse | None = None,
    ) -> AnchorSet: ...


class GroupingStrategy(Protocol):
    name: str

    def build_groups(
        self,
        gaussian_state: Any,
        render_state: RenderState | None,
        view: Any,
        config: dict[str, Any],
        anchor_set: AnchorSet | None = None,
        universe: GaussianUniverse | None = None,
    ) -> GroupSet: ...


class ParameterBlock(Protocol):
    name: str
    dimension_per_gaussian: int

    def gather(self, gaussian_model: Any, indices: torch.Tensor) -> torch.Tensor: ...
    def apply_update(self, gaussian_model: Any, indices: torch.Tensor, delta: torch.Tensor) -> None: ...
    def project_or_clamp_update(self, state: torch.Tensor, delta: torch.Tensor, config: dict[str, Any]) -> torch.Tensor: ...
    def snapshot(self, gaussian_model: Any, indices: torch.Tensor) -> torch.Tensor: ...
    def restore(self, gaussian_model: Any, indices: torch.Tensor, snapshot: torch.Tensor) -> None: ...


class ResidualProvider(Protocol):
    name: str

    def build(self, prediction: torch.Tensor, target: torch.Tensor, pixel_indices: torch.Tensor | None = None) -> ResidualData: ...


class JacobianProvider(Protocol):
    name: str

    def compute(self, render_fn: Any, parameter: torch.Tensor, gaussian_ids: tuple[int, ...], residual_data: ResidualData, config: dict[str, Any]) -> JacobianData: ...


class CurvatureAssembler(Protocol):
    name: str

    def assemble(self, jacobian_data: JacobianData, block_dimension: int) -> CurvatureData: ...


class OptimizationStrategy(Protocol):
    name: str

    def propose_update(self, group: Group, parameter_block: ParameterBlock, residual_data: ResidualData, jacobian_data: JacobianData, curvature_data: CurvatureData, config: dict[str, Any]) -> UpdateProposal: ...
