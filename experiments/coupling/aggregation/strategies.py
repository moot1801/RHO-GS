"""Aggregation remains independent from solvers by design."""

from __future__ import annotations

import torch

from ...registry import REGISTRIES
from ...types import Group, UpdateProposal


@REGISTRIES["aggregation"].register("anchor_only")
class AnchorOnlyAggregation:
    name = "anchor_only"

    def aggregate(self, groups: list[Group], proposals: list[UpdateProposal]) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        updates: dict[int, torch.Tensor] = {}
        for group, proposal in zip(groups, proposals, strict=True):
            try:
                anchor_index = proposal.gaussian_ids.index(group.anchor_gaussian_id)
            except ValueError as error:
                raise ValueError(f"anchor {group.anchor_gaussian_id} missing from proposal") from error
            updates[group.anchor_gaussian_id] = proposal.delta[anchor_index]
        ids = sorted(updates)
        if not ids:
            return torch.empty(0, dtype=torch.long), torch.empty((0, 0)), {"proposal_count": 0}
        return (
            torch.tensor(ids, dtype=torch.long, device=next(iter(updates.values())).device),
            torch.stack([updates[index] for index in ids]),
            {"proposal_count": len(proposals), "collision_count": len(proposals) - len(ids)},
        )


@REGISTRIES["aggregation"].register("weighted_average")
class WeightedAverageAggregation:
    name = "weighted_average"

    def aggregate(self, groups: list[Group], proposals: list[UpdateProposal]) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        sums: dict[int, torch.Tensor] = {}
        weights: dict[int, float] = {}
        proposal_count = 0
        for proposal in proposals:
            weight = max(float(proposal.predicted_reduction), 0.0) + 1.0e-12
            for gaussian_id, delta in zip(proposal.gaussian_ids, proposal.delta, strict=True):
                sums[gaussian_id] = sums.get(gaussian_id, torch.zeros_like(delta)) + weight * delta
                weights[gaussian_id] = weights.get(gaussian_id, 0.0) + weight
                proposal_count += 1
        ids = sorted(sums)
        if not ids:
            return torch.empty(0, dtype=torch.long), torch.empty((0, 0)), {"proposal_count": 0}
        updates = torch.stack([sums[index] / weights[index] for index in ids])
        return (
            torch.tensor(ids, dtype=torch.long, device=updates.device),
            updates,
            {"proposal_count": proposal_count, "collision_count": proposal_count - len(ids)},
        )
