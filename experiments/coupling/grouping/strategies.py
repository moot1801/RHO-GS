"""Deterministic reference grouping implementations."""

from __future__ import annotations

from typing import Any

import torch

from ...registry import REGISTRIES
from ...types import Group, GroupSet, RenderState


def _means(state: Any) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state
    return state.means


def _anchors(config: dict[str, Any], count: int, device: torch.device, eligible: torch.Tensor | None = None) -> torch.Tensor:
    candidates = torch.arange(count, device=device) if eligible is None else torch.nonzero(eligible, as_tuple=False).flatten()
    requested = int(config.get("sampled_anchor_count", candidates.numel()))
    maximum = int(config.get("maximum_groups", requested))
    take = min(candidates.numel(), requested, maximum)
    generator = torch.Generator(device="cpu").manual_seed(int(config.get("seed", 0)))
    order = torch.randperm(candidates.numel(), generator=generator)[:take].to(candidates.device)
    return candidates[order]


def _group_set(name: str, groups: list[Group], config: dict[str, Any]) -> GroupSet:
    return GroupSet(
        strategy=name,
        groups=groups,
        seed=int(config.get("seed", 0)),
        overlapping=bool(config.get("overlapping", True)),
        directed=bool(config.get("directed", False)),
        metadata={"candidate_pool_size": int(config.get("candidate_pool_size", 0))},
    )


@REGISTRIES["grouping"].register("independent")
class IndependentGrouping:
    name = "independent"

    def build_groups(self, gaussian_state: Any, render_state: RenderState | None, view: Any, config: dict[str, Any]) -> GroupSet:
        means = _means(gaussian_state)
        eligible = render_state.visible_mask if render_state is not None and config.get("visible_only", False) else None
        anchors = _anchors(config, means.shape[0], means.device, eligible)
        groups = [Group(i, int(anchor), (int(anchor),), (0.0,)) for i, anchor in enumerate(anchors.tolist())]
        return _group_set(self.name, groups, config)


class _KNNBase:
    name = "knn_3d"
    require_visible = False
    require_overlap = False

    def build_groups(self, gaussian_state: Any, render_state: RenderState | None, view: Any, config: dict[str, Any]) -> GroupSet:
        means = _means(gaussian_state)
        if (self.require_visible or self.require_overlap) and render_state is None:
            raise ValueError(f"{self.name} requires RenderState")
        eligible = render_state.visible_mask.clone() if self.require_visible else None
        visibility_threshold = float(config.get("visibility_threshold", 0.0))
        if eligible is not None and visibility_threshold > 0.0 and hasattr(gaussian_state, "opacities"):
            eligible &= gaussian_state.opacities.detach().flatten() >= visibility_threshold
        anchors = _anchors(config, means.shape[0], means.device, eligible)
        group_size = max(1, int(config.get("group_size", 1)))
        pool_size = max(group_size, int(config.get("candidate_pool_size", max(group_size * 4, 32))))
        distance_threshold = config.get("distance_threshold")
        groups: list[Group] = []
        used: set[int] = set()
        for group_id, anchor_tensor in enumerate(anchors):
            anchor = int(anchor_tensor.item())
            if not bool(config.get("overlapping", True)) and anchor in used:
                continue
            distances = torch.linalg.vector_norm(means - means[anchor], dim=-1)
            candidate_mask = torch.ones(means.shape[0], dtype=torch.bool, device=means.device)
            if self.require_visible:
                candidate_mask &= eligible
            if not bool(config.get("overlapping", True)) and used:
                used_ids = torch.tensor(sorted(used), dtype=torch.long, device=means.device)
                candidate_mask[used_ids] = False
            if distance_threshold is not None:
                candidate_mask &= distances <= float(distance_threshold)
            candidate_mask[anchor] = True
            candidate_ids = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            if candidate_ids.numel() > pool_size:
                nearest = torch.topk(distances[candidate_ids], k=pool_size, largest=False).indices
                candidate_ids = candidate_ids[nearest]
            shared_tiles: list[int] = []
            overlap_scores: list[float] = []
            shared_map: dict[int, int] = {}
            overlap_map: dict[int, float] = {}
            if self.require_overlap:
                use_tiles = bool(config.get("use_tile_overlap", True))
                source_bounds = render_state.tile_bounds if use_tiles else render_state.projected_bounds
                anchor_bounds = source_bounds[anchor]
                intersection_min = torch.maximum(source_bounds[candidate_ids, :2], anchor_bounds[:2])
                intersection_max = torch.minimum(source_bounds[candidate_ids, 2:], anchor_bounds[2:])
                offset = 1 if use_tiles else 0
                extents = (intersection_max - intersection_min + offset).clamp_min(0)
                shared = extents[:, 0] * extents[:, 1]
                anchor_area = ((anchor_bounds[2:] - anchor_bounds[:2] + offset).clamp_min(0)).prod().clamp_min(1)
                overlap = shared / anchor_area
                keep = shared > 0
                if use_tiles:
                    keep &= shared >= int(config.get("minimum_shared_tiles", 1))
                keep &= overlap >= float(config.get("projected_overlap_threshold", 0.0))
                candidate_ids = candidate_ids[keep]
                shared = shared[keep]
                overlap = overlap[keep]
                shared_tiles = [int(value) for value in shared.tolist()]
                overlap_scores = [float(value) for value in overlap.tolist()]
                shared_map = {int(candidate_id): shared_tiles[i] for i, candidate_id in enumerate(candidate_ids.tolist())}
                overlap_map = {int(candidate_id): overlap_scores[i] for i, candidate_id in enumerate(candidate_ids.tolist())}
            if anchor not in candidate_ids.tolist():
                candidate_ids = torch.cat((torch.tensor([anchor], device=means.device), candidate_ids))
                shared_map[anchor] = 0
                overlap_map[anchor] = 1.0
            order = torch.argsort(distances[candidate_ids])[:group_size]
            members = candidate_ids[order]
            member_list = [int(value) for value in members.tolist()]
            scores = [float(distances[value].detach().item()) for value in member_list]
            if self.require_overlap:
                member_shared = tuple(shared_map.get(value, 0) for value in member_list)
                member_overlap = tuple(overlap_map.get(value, 0.0) for value in member_list)
            else:
                member_shared = ()
                member_overlap = ()
            if not bool(config.get("overlapping", True)):
                used.update(member_list)
            directed = bool(config.get("directed", False))
            edges = tuple((anchor, value) for value in member_list if value != anchor) if directed else tuple(
                (min(anchor, value), max(anchor, value)) for value in member_list if value != anchor
            )
            groups.append(Group(
                group_id=len(groups),
                anchor_gaussian_id=anchor,
                member_gaussian_ids=tuple(member_list),
                construction_scores=tuple(scores),
                visible=tuple(bool(render_state.visible_mask[value]) for value in member_list) if render_state is not None else (),
                projected_overlap=member_overlap,
                shared_tile_counts=member_shared,
                directed_edges=edges,
                metadata={"distance_metric": "euclidean_world", "overlap_basis": "tile" if config.get("use_tile_overlap", True) else "pixel_bbox"},
            ))
        return _group_set(self.name, groups, config)


@REGISTRIES["grouping"].register("knn_3d")
class KNN3DGrouping(_KNNBase):
    name = "knn_3d"


@REGISTRIES["grouping"].register("visible_knn_3d")
class VisibleKNN3DGrouping(_KNNBase):
    name = "visible_knn_3d"
    require_visible = True


@REGISTRIES["grouping"].register("visible_overlap_knn")
class VisibleOverlapKNNGrouping(_KNNBase):
    name = "visible_overlap_knn"
    require_visible = True
    require_overlap = True


@REGISTRIES["grouping"].register("oracle_jtj_topk")
class OracleJTJTopKGrouping:
    name = "oracle_jtj_topk"

    def __init__(self, **_: Any) -> None:
        raise NotImplementedError("oracle_jtj_topk is a Priority 4 analysis extension point and is not implemented in the reference milestone")
