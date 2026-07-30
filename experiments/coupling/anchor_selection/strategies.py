"""Deterministic anchor selectors with explicit candidate populations."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from ...registry import REGISTRIES
from ...types import AnchorSet, GaussianUniverse, RenderState


def candidate_hash(candidate_ids: torch.Tensor) -> str:
    canonical = candidate_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(canonical.numpy().tobytes()).hexdigest()[:16]


def _geometry_eligible(
    gaussian_state: Any,
    render_state: RenderState,
    config: dict[str, Any],
    universe: GaussianUniverse | None = None,
) -> torch.Tensor:
    eligible = render_state.visible_mask.clone()
    opacity_threshold = float(config.get("opacity_threshold", 0.0))
    if opacity_threshold > 0.0 and hasattr(gaussian_state, "opacities"):
        eligible &= gaussian_state.opacities.detach().flatten() >= opacity_threshold
    if universe is not None:
        universe_mask = torch.zeros_like(eligible)
        universe_ids = torch.tensor(universe.gaussian_ids, dtype=torch.long, device=eligible.device)
        universe_mask[universe_ids] = True
        eligible &= universe_mask
    return eligible


def _take_count(config: dict[str, Any], available: int) -> int:
    requested = int(config.get("sampled_anchor_count", available))
    maximum = int(config.get("maximum_anchors", requested))
    return min(available, requested, maximum)


def _random_subset(candidates: torch.Tensor, take: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(candidates.numel(), generator=generator, device="cpu")[:take].to(candidates.device)
    return candidates.index_select(0, order)


def _scores_for(selected: torch.Tensor, scores: torch.Tensor | None) -> tuple[float, ...]:
    if scores is None:
        return tuple(0.0 for _ in range(selected.numel()))
    return tuple(float(value) for value in scores.index_select(0, selected).detach().cpu().tolist())


def _anchor_set(
    name: str,
    selected: torch.Tensor,
    candidates: torch.Tensor,
    config: dict[str, Any],
    scores: torch.Tensor | None,
    metadata: dict[str, Any],
) -> AnchorSet:
    return AnchorSet(
        strategy=name,
        anchor_gaussian_ids=tuple(int(value) for value in selected.detach().cpu().tolist()),
        scores=_scores_for(selected, scores),
        seed=int(config.get("seed", 0)),
        candidate_gaussian_ids=tuple(int(value) for value in candidates.detach().cpu().tolist()),
        candidate_hash=candidate_hash(candidates),
        filter_group_members=bool(config.get("filter_group_members", False)),
        metadata={
            "sampled_anchor_count": int(selected.numel()),
            "opacity_threshold": float(config.get("opacity_threshold", 0.0)),
            **metadata,
        },
    )


@REGISTRIES["anchor_selection"].register("random_eligible")
class RandomEligible:
    name = "random_eligible"
    requires_contribution_scores = False

    def select(self, gaussian_state: Any, render_state: RenderState, view: Any, config: dict[str, Any], contribution_scores: torch.Tensor | None = None, universe: GaussianUniverse | None = None) -> AnchorSet:
        del view, contribution_scores
        candidates = torch.nonzero(_geometry_eligible(gaussian_state, render_state, config, universe), as_tuple=False).flatten()
        selected = _random_subset(candidates, _take_count(config, candidates.numel()), int(config.get("seed", 0)))
        return _anchor_set(self.name, selected, candidates, config, None, {
            "score_basis": "none",
            "candidate_basis": "geometry_visible" if universe is None else "fixed_universe_and_geometry_visible",
            "universe_hash": None if universe is None else universe.gaussian_id_hash,
        })


@REGISTRIES["anchor_selection"].register("fixed")
class FixedAnchorSelection:
    name = "fixed"
    requires_contribution_scores = False

    def select(self, gaussian_state: Any, render_state: RenderState, view: Any, config: dict[str, Any], contribution_scores: torch.Tensor | None = None, universe: GaussianUniverse | None = None) -> AnchorSet:
        del view
        fixed_ids = tuple(int(value) for value in config.get("fixed_ids", ()))
        if not fixed_ids:
            raise ValueError("fixed anchor selection requires anchor_selection.fixed_ids")
        count = int(gaussian_state.means.shape[0])
        if len(set(fixed_ids)) != len(fixed_ids) or any(value < 0 or value >= count for value in fixed_ids):
            raise ValueError("fixed anchor IDs must be unique valid checkpoint-local Gaussian indices")
        selected = torch.tensor(fixed_ids, dtype=torch.long, device=gaussian_state.means.device)
        geometry_mask = _geometry_eligible(gaussian_state, render_state, config, universe)
        if bool(config.get("require_geometry_eligible", True)) and not bool(geometry_mask.index_select(0, selected).all()):
            raise ValueError("one or more fixed anchor IDs are not eligible in the sampled view/tiles")
        candidates = torch.nonzero(geometry_mask, as_tuple=False).flatten()
        return _anchor_set(self.name, selected, candidates, config, contribution_scores, {
            "score_basis": "provided_or_zero",
            "candidate_basis": "geometry_visible" if universe is None else "fixed_universe_and_geometry_visible",
            "universe_hash": None if universe is None else universe.gaussian_id_hash,
        })


class _ContributorBase:
    requires_contribution_scores = True
    random = True

    def select(self, gaussian_state: Any, render_state: RenderState, view: Any, config: dict[str, Any], contribution_scores: torch.Tensor | None = None, universe: GaussianUniverse | None = None) -> AnchorSet:
        del view
        if contribution_scores is None:
            raise ValueError(f"{self.name} requires contribution scores")
        eligible = _geometry_eligible(gaussian_state, render_state, config, universe)
        eligible_scores = contribution_scores[eligible]
        maximum = float(eligible_scores.max().item()) if eligible_scores.numel() else 0.0
        absolute = float(config.get("absolute_score_threshold", 0.0))
        relative = float(config.get("relative_score_threshold", 1.0e-8))
        threshold = max(absolute, relative * maximum)
        contributor_mask = eligible & (contribution_scores > threshold)
        candidates = torch.nonzero(contributor_mask, as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise RuntimeError(
                f"contribution probe found no eligible anchors (maximum_score={maximum:.6g}, threshold={threshold:.6g})"
            )
        take = _take_count(config, candidates.numel())
        if self.random:
            selected = _random_subset(candidates, take, int(config.get("seed", 0)))
        else:
            local_scores = contribution_scores.index_select(0, candidates)
            order = torch.argsort(local_scores, descending=True, stable=True)[:take]
            selected = candidates.index_select(0, order)
        return _anchor_set(
            self.name,
            selected,
            candidates,
            config,
            contribution_scores,
            {
                "score_basis": "randomized_position_jacobian_energy",
                "candidate_basis": "geometry_visible_and_contributor",
                "maximum_score": maximum,
                "score_threshold": threshold,
                "absolute_score_threshold": absolute,
                "relative_score_threshold": relative,
                "analysis_only": not self.random,
                "universe_hash": None if universe is None else universe.gaussian_id_hash,
            },
        )


@REGISTRIES["anchor_selection"].register("random_contributor")
class RandomContributor(_ContributorBase):
    name = "random_contributor"
    random = True


@REGISTRIES["anchor_selection"].register("contribution_topk")
class ContributionTopK(_ContributorBase):
    name = "contribution_topk"
    random = False
