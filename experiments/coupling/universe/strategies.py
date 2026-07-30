"""Deterministic universe selection before anchor selection and grouping."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from ...registry import REGISTRIES
from ...types import GaussianUniverse, RenderState


def gaussian_id_hash(gaussian_ids: torch.Tensor | tuple[int, ...]) -> str:
    canonical = torch.as_tensor(gaussian_ids).detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(canonical.numpy().tobytes()).hexdigest()[:16]


def _half_extent(config: dict[str, Any], means: torch.Tensor) -> torch.Tensor:
    raw = config.get("half_extent")
    if raw is None:
        raise ValueError("fixed_cube universe requires universe.half_extent")
    values = torch.as_tensor(raw, dtype=means.dtype, device=means.device).flatten()
    if values.numel() == 1:
        values = values.repeat(3)
    if values.numel() != 3 or not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
        raise ValueError("universe.half_extent must contain one or three finite positive values")
    return values


def _center(config: dict[str, Any], means: torch.Tensor) -> tuple[torch.Tensor, int | None]:
    center_id = config.get("center_gaussian_id")
    center_xyz = config.get("center_xyz")
    if center_id is not None and center_xyz is not None:
        raise ValueError("set only one of universe.center_gaussian_id and universe.center_xyz")
    if center_id is not None:
        center_id = int(center_id)
        if center_id < 0 or center_id >= means.shape[0]:
            raise ValueError("universe.center_gaussian_id is outside the checkpoint-local Gaussian range")
        return means[center_id].detach(), center_id
    if center_xyz is None:
        raise ValueError("fixed_cube universe requires universe.center_xyz or universe.center_gaussian_id")
    center = torch.as_tensor(center_xyz, dtype=means.dtype, device=means.device).flatten()
    if center.numel() != 3 or not bool(torch.isfinite(center).all()):
        raise ValueError("universe.center_xyz must contain three finite values")
    return center, None


@REGISTRIES["universe"].register("disabled")
class DisabledUniverse:
    name = "disabled"

    def needs_contribution_scores(self, config: dict[str, Any]) -> bool:
        del config
        return False

    def build(
        self,
        gaussian_state: Any,
        render_state: RenderState,
        view: Any,
        config: dict[str, Any],
        contribution_scores: torch.Tensor | None = None,
    ) -> None:
        del gaussian_state, render_state, view, config, contribution_scores
        return None


@REGISTRIES["universe"].register("fixed_cube")
class FixedCubeUniverse:
    name = "fixed_cube"

    def needs_contribution_scores(self, config: dict[str, Any]) -> bool:
        return bool(config.get("contributor_only", False))

    def build(
        self,
        gaussian_state: Any,
        render_state: RenderState,
        view: Any,
        config: dict[str, Any],
        contribution_scores: torch.Tensor | None = None,
    ) -> GaussianUniverse:
        del view
        means = gaussian_state if isinstance(gaussian_state, torch.Tensor) else gaussian_state.means
        center, center_id = _center(config, means)
        half_extent = _half_extent(config, means)
        mask = ((means.detach() - center).abs() <= half_extent).all(dim=-1)
        cube_count = int(mask.sum().item())

        visible_only = bool(config.get("visible_only", False))
        if visible_only:
            mask &= render_state.visible_mask
        post_visibility_count = int(mask.sum().item())

        contributor_only = bool(config.get("contributor_only", False))
        contributor_threshold = None
        if contributor_only:
            if contribution_scores is None:
                raise ValueError("universe.contributor_only=True requires contribution scores")
            eligible_scores = contribution_scores[mask]
            maximum_score = float(eligible_scores.max().item()) if eligible_scores.numel() else 0.0
            absolute = float(config.get("absolute_score_threshold", 0.0))
            relative = float(config.get("relative_score_threshold", 1.0e-8))
            contributor_threshold = max(absolute, relative * maximum_score)
            mask &= contribution_scores > contributor_threshold

        ids = torch.nonzero(mask, as_tuple=False).flatten()
        minimum = int(config.get("minimum_gaussians", 2))
        maximum = config.get("maximum_gaussians")
        if minimum < 1:
            raise ValueError("universe.minimum_gaussians must be positive")
        if maximum is not None and int(maximum) < minimum:
            raise ValueError("universe.maximum_gaussians must be at least minimum_gaussians")
        if ids.numel() < minimum:
            raise RuntimeError(f"fixed_cube universe contains {ids.numel()} Gaussians, below minimum_gaussians={minimum}")
        pre_overflow_count = int(ids.numel())
        overflow_applied = False
        if maximum is not None and ids.numel() > int(maximum):
            policy = str(config.get("overflow_policy", "error"))
            if policy == "error":
                raise RuntimeError(
                    f"fixed_cube universe contains {ids.numel()} Gaussians, above maximum_gaussians={int(maximum)}"
                )
            if policy != "nearest_center":
                raise ValueError("universe.overflow_policy must be 'error' or 'nearest_center'")
            distances = torch.linalg.vector_norm(means.detach().index_select(0, ids) - center, dim=-1)
            order = torch.argsort(distances, stable=True)[:int(maximum)]
            ids = ids.index_select(0, order)
            overflow_applied = True
        ids = torch.sort(ids).values
        return GaussianUniverse(
            strategy=self.name,
            gaussian_ids=tuple(int(value) for value in ids.detach().cpu().tolist()),
            gaussian_id_hash=gaussian_id_hash(ids),
            metadata={
                "center_xyz": tuple(float(value) for value in center.detach().cpu().tolist()),
                "center_gaussian_id": center_id,
                "half_extent": tuple(float(value) for value in half_extent.detach().cpu().tolist()),
                "visible_only": visible_only,
                "contributor_only": contributor_only,
                "contributor_threshold": contributor_threshold,
                "cube_count": cube_count,
                "post_visibility_count": post_visibility_count,
                "pre_overflow_count": pre_overflow_count,
                "overflow_applied": overflow_applied,
                "minimum_gaussians": minimum,
                "maximum_gaussians": None if maximum is None else int(maximum),
                "overflow_policy": str(config.get("overflow_policy", "error")),
                "selection_order": "cube_then_optional_filters_then_overflow",
                "render_state_scope": render_state.metadata.get("restricted_to_sampled_tiles", "full_view"),
            },
        )
