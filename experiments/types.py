"""Typed, serializable records shared by coupling experiment modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class AnchorSet:
    strategy: str
    anchor_gaussian_ids: tuple[int, ...]
    scores: tuple[float, ...]
    seed: int
    candidate_gaussian_ids: tuple[int, ...]
    candidate_hash: str
    filter_group_members: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_gaussian_ids)

    def to_record(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "anchor_gaussian_ids": self.anchor_gaussian_ids,
            "scores": self.scores,
            "seed": self.seed,
            "candidate_count": self.candidate_count,
            "candidate_hash": self.candidate_hash,
            "filter_group_members": self.filter_group_members,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class Group:
    group_id: int
    anchor_gaussian_id: int
    member_gaussian_ids: tuple[int, ...]
    construction_scores: tuple[float, ...] = ()
    visible: tuple[bool, ...] = ()
    projected_overlap: tuple[float, ...] = ()
    shared_tile_counts: tuple[int, ...] = ()
    directed_edges: tuple[tuple[int, int], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.member_gaussian_ids)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["group_size"] = self.size
        return record


@dataclass(slots=True)
class GroupSet:
    strategy: str
    groups: list[Group]
    seed: int
    overlapping: bool
    directed: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RenderState:
    visible_mask: torch.Tensor
    projected_xy: torch.Tensor
    depth: torch.Tensor
    projected_radius: torch.Tensor
    projected_bounds: torch.Tensor
    tile_bounds: torch.Tensor
    image_width: int
    image_height: int
    tile_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResidualData:
    residual: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    pixel_indices: torch.Tensor
    loss: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JacobianData:
    gaussian_ids: tuple[int, ...]
    jacobian: torch.Tensor
    residual: torch.Tensor
    gradient: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CurvatureData:
    gaussian_ids: tuple[int, ...]
    hessian: torch.Tensor
    gradient: torch.Tensor
    diagonal_blocks: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UpdateProposal:
    gaussian_ids: tuple[int, ...]
    delta: torch.Tensor
    predicted_reduction: float
    damping: float
    condition_estimate: float
    solver_iterations: int
    converged: bool
    numerical_status: str
    solve_time_ms: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AcceptanceDecision:
    accepted: bool
    reason: str
    gain_ratio: float | None = None
    next_damping: float | None = None


@dataclass(slots=True)
class TimingRecord:
    phase: str
    milliseconds: float
    device: str
    warmup: bool = False
    cpu_milliseconds: float | None = None


@dataclass(slots=True)
class MemoryRecord:
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
