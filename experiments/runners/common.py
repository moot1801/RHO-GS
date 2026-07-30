"""Shared reference benchmark orchestration."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from ..config import compose_config
from ..coupling import aggregation, acceptance, anchor_selection, curvature, grouping, jacobian, parameter_blocks, residuals, solvers  # noqa: F401
from ..coupling.anchor_selection import estimate_position_jacobian_energy
from ..coupling.curvature import GaussNewtonAssembler
from ..evaluation import select_audit_views
from ..evaluation import PhaseTimer
from ..registry import REGISTRIES
from ..sampling import build_render_state, sample_top_tiles_and_pixels
from ..types import AnchorSet, CurvatureData, Group, GroupSet, JacobianData, RenderState, ResidualData


@dataclass(slots=True)
class PreparedExperiment:
    config: dict[str, Any]
    runtime: Any
    audit_views: dict[str, list[tuple[str, int, object]]]
    primary_view: Any
    target: torch.Tensor
    render_state: RenderState
    pixel_indices: torch.Tensor
    anchor_set: AnchorSet
    group_set: GroupSet
    residual_data: ResidualData
    timing_records: list[Any]


def parse_config(argv: list[str]) -> dict[str, Any]:
    config = compose_config(argv)
    if not config.get("checkpoint"):
        raise ValueError("checkpoint=<path> is required")
    config["grouping"]["seed"] = int(config.get("seed", 0))
    config["anchor_selection"]["seed"] = int(config.get("seed", 0))
    return config


def _restrict_to_tiles(state: RenderState, tile_ids: torch.Tensor) -> RenderState:
    tiles_x = (state.image_width + state.tile_size - 1) // state.tile_size
    selected = torch.zeros_like(state.visible_mask)
    for tile_id in tile_ids.tolist():
        ty, tx = divmod(tile_id, tiles_x)
        bounds = state.tile_bounds
        selected |= (bounds[:, 0] <= tx) & (bounds[:, 2] >= tx) & (bounds[:, 1] <= ty) & (bounds[:, 3] >= ty)
    return RenderState(
        visible_mask=state.visible_mask & selected,
        projected_xy=state.projected_xy,
        depth=state.depth,
        projected_radius=state.projected_radius,
        projected_bounds=state.projected_bounds,
        tile_bounds=state.tile_bounds,
        image_width=state.image_width,
        image_height=state.image_height,
        tile_size=state.tile_size,
        metadata={**state.metadata, "restricted_to_sampled_tiles": [int(value) for value in tile_ids.tolist()]},
    )


def prepare(config: dict[str, Any]) -> PreparedExperiment:
    from ..runtime import load_runtime, target_for_view

    setup_start = perf_counter()
    runtime = load_runtime(config)
    setup_ms = (perf_counter() - setup_start) * 1000.0
    evaluation = config["evaluation"]
    audits = select_audit_views(
        runtime.dataset,
        primary_view_id=int(evaluation["primary_view_id"]),
        nearby_count=int(evaluation["nearby_view_count"]),
        global_count=int(evaluation["global_audit_view_count"]),
    )
    primary = audits["primary"][0][2]
    prepared = prepare_view(config, runtime, audits, primary, int(evaluation["primary_view_id"]))
    from ..types import TimingRecord
    prepared.timing_records.insert(0, TimingRecord("runtime_setup", setup_ms, "cpu", False, setup_ms))
    return prepared


def prepare_view(
    config: dict[str, Any],
    runtime: Any,
    audits: dict[str, list[tuple[str, int, object]]],
    primary: Any,
    view_id: int | None = None,
) -> PreparedExperiment:
    from ..runtime import target_for_view

    evaluation = config["evaluation"]
    timing_records = []
    device = runtime.model.gaussians.means.device
    target = target_for_view(primary)
    with PhaseTimer("render_state_and_sampling", device=device) as timer:
        state = build_render_state(runtime.model.gaussians, primary, tile_size=int(evaluation.get("tile_size", 16)))
        tile_ids, pixel_ids = sample_top_tiles_and_pixels(state, {**evaluation, "seed": int(config.get("seed", 0))})
        restricted_state = _restrict_to_tiles(state, tile_ids)
    timing_records.append(timer.record)
    with PhaseTimer("render_forward", device=device) as timer:
        with torch.no_grad():
            prediction = runtime.training_render(primary)
    timing_records.append(timer.record)
    with PhaseTimer("residual_construction", device=device) as timer:
        residual_provider = REGISTRIES["residual"].create(config["residual"]["name"])
        residual_data = residual_provider.build(prediction, target, pixel_ids)
    timing_records.append(timer.record)
    effective_view_id = int(evaluation["primary_view_id"]) if view_id is None else int(view_id)
    residual_data.metadata.update({
        "sampled_tile_ids": [int(value) for value in tile_ids.tolist()],
        "sampled_pixel_ids": [int(value) for value in pixel_ids.tolist()],
        "view_id": effective_view_id,
    })
    anchor_strategy = REGISTRIES["anchor_selection"].create(config["anchor_selection"]["name"])
    contribution_scores = None
    if anchor_strategy.requires_contribution_scores:
        with PhaseTimer("contribution_probe", device=device) as timer:
            contribution_scores, probe_metadata = estimate_position_jacobian_energy(
                lambda: runtime.training_render(primary),
                runtime.model.gaussians.means,
                residual_data,
                config["anchor_selection"],
            )
        timing_records.append(timer.record)
    else:
        probe_metadata = {"estimator": "not_required"}
    with PhaseTimer("anchor_selection", device=device) as timer:
        anchor_set = anchor_strategy.select(
            runtime.model.gaussians,
            restricted_state,
            primary,
            config["anchor_selection"],
            contribution_scores,
        )
    timing_records.append(timer.record)
    anchor_set.metadata.update(probe_metadata)
    with PhaseTimer("group_construction", device=device) as timer:
        grouping_strategy = REGISTRIES["grouping"].create(config["grouping"]["name"])
        group_set = grouping_strategy.build_groups(runtime.model.gaussians, restricted_state, primary, config["grouping"], anchor_set)
    timing_records.append(timer.record)
    if not group_set.groups:
        raise RuntimeError("grouping produced no groups for the sampled anchors")
    return PreparedExperiment(config, runtime, audits, primary, target, restricted_state, pixel_ids, anchor_set, group_set, residual_data, timing_records)


def checkpoint_file_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = checkpoint.stat()
    return {
        "path": str(checkpoint),
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def union_gaussian_ids(group_set: GroupSet) -> tuple[int, ...]:
    return tuple(sorted({index for group in group_set.groups for index in group.member_gaussian_ids}))


def compute_union_jacobian(prepared: PreparedExperiment, ids: tuple[int, ...]) -> tuple[JacobianData, CurvatureData]:
    provider = REGISTRIES["jacobian"].create(prepared.config["jacobian"]["name"])
    jacobian_data = provider.compute(
        lambda: prepared.runtime.training_render(prepared.primary_view),
        prepared.runtime.model.gaussians.means,
        ids,
        prepared.residual_data,
        prepared.config["jacobian"],
    )
    assembler = REGISTRIES["curvature"].create(prepared.config["curvature"]["name"])
    return jacobian_data, assembler.assemble(jacobian_data, 3)


def subset_data(union_jacobian: JacobianData, group: Group) -> tuple[JacobianData, CurvatureData]:
    lookup = {gaussian_id: index for index, gaussian_id in enumerate(union_jacobian.gaussian_ids)}
    columns: list[int] = []
    for gaussian_id in group.member_gaussian_ids:
        block = lookup[gaussian_id]
        columns.extend(range(block * 3, block * 3 + 3))
    index = torch.tensor(columns, dtype=torch.long, device=union_jacobian.jacobian.device)
    jacobian = union_jacobian.jacobian.index_select(1, index)
    subset = JacobianData(
        gaussian_ids=group.member_gaussian_ids,
        jacobian=jacobian,
        residual=union_jacobian.residual,
        gradient=jacobian.T @ union_jacobian.residual,
        metadata={**union_jacobian.metadata, "subset_of_union": True},
    )
    return subset, GaussNewtonAssembler().assemble(subset, 3)


def aggregate_predicted_reduction(
    union_curvature: CurvatureData,
    union_ids: tuple[int, ...],
    update_ids: torch.Tensor,
    delta: torch.Tensor,
) -> float:
    """Evaluate one quadratic model after aggregation, avoiding overlap double-counting."""

    lookup = {gaussian_id: index for index, gaussian_id in enumerate(union_ids)}
    union_delta = union_curvature.gradient.new_zeros(union_curvature.gradient.shape)
    for gaussian_id, update in zip(update_ids.tolist(), delta, strict=True):
        block = lookup[int(gaussian_id)]
        union_delta[block * 3:block * 3 + 3] = update.to(union_delta)
    value = -(torch.dot(union_curvature.gradient, union_delta) + 0.5 * torch.dot(union_delta, union_curvature.hessian @ union_delta))
    return float(value.item())


def snapshot_model_and_optimizer(runtime: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model_state = {key: value.detach().clone() for key, value in runtime.model.state_dict().items()}
    return model_state, copy.deepcopy(runtime.model.gaussians.optimizer.state_dict())


def restore_model_and_optimizer(runtime: Any, snapshot: tuple[dict[str, torch.Tensor], dict[str, Any]]) -> None:
    runtime.model.load_state_dict(snapshot[0], strict=True)
    runtime.model.gaussians.optimizer.load_state_dict(snapshot[1])


@torch.no_grad()
def evaluate_views(prepared: PreparedExperiment) -> dict[str, dict[str, float]]:
    from ..runtime import target_for_view

    rows: dict[str, dict[str, float]] = {}
    for category, entries in prepared.audit_views.items():
        for subset, view_id, view in entries:
            prediction = prepared.runtime.inference_render(view)
            target = target_for_view(view)
            loss = 0.5 * torch.mean((prediction - target.to(prediction)) ** 2)
            rows[f"{category}:{subset}:{view_id}"] = {"loss": float(loss.item())}
    return rows


def apply_remaining_attribute_adam(prepared: PreparedExperiment) -> float:
    runtime = prepared.runtime
    runtime.model.gaussians.optimizer.zero_grad()
    prediction = runtime.training_render(prepared.primary_view)
    loss = runtime.native_loss(prediction, prepared.target)
    loss.backward()
    runtime.model.gaussians.means.grad = None
    runtime.model.gaussians.optimizer.step()
    runtime.model.gaussians.optimizer.zero_grad()
    return float(loss.item())
