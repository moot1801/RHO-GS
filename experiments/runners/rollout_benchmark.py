"""Fixed-topology short rollout with coupled position and Adam for other attributes."""

from __future__ import annotations

import sys
from dataclasses import asdict

import torch

from ..coupling.diagnostics import diagnose_groups
from ..evaluation import MemoryTracker, PhaseTimer, image_metrics
from ..registry import REGISTRIES
from ..result_logger import ResultLogger
from ..runtime import target_for_view
from .common import (
    apply_remaining_attribute_adam,
    aggregate_predicted_reduction,
    checkpoint_file_metadata,
    compute_union_jacobian,
    parse_config,
    prepare,
    prepare_view,
    restore_model_and_optimizer,
    snapshot_model_and_optimizer,
    subset_data,
    union_gaussian_ids,
)


def main(argv: list[str] | None = None) -> int:
    config = parse_config(sys.argv[1:] if argv is None else argv)
    logger = ResultLogger(config)
    logger.initialize_expected_artifacts()
    rollout_rows: list[dict] = []
    timing_rows: list[dict] = []
    memory_rows: list[dict] = []
    group_diagnostic_rows: list[dict] = []
    accepted_count = 0
    numerical_failures = 0
    try:
        initial = prepare(config)
        timing_rows.extend({"rollout_iteration": 0, **asdict(record)} for record in initial.timing_records)
        runtime = initial.runtime
        device = runtime.model.gaussians.means.device
        initial_gaussian_count = int(runtime.model.gaussians.means.shape[0])
        train_views = list(runtime.dataset.train())
        generator = torch.Generator(device="cpu").manual_seed(int(config["benchmark"].get("camera_sequence_seed", 0)))
        sequence = torch.randperm(len(train_views), generator=generator, device="cpu").tolist()
        iterations = int(config["benchmark"].get("iterations", 100))
        evaluation_interval = int(config["benchmark"].get("evaluation_interval", 10))
        damping = float(config["solver"].get("initial_damping", 1.0e-3))
        block = REGISTRIES["parameter_block"].create(config["attributes"]["name"])
        solver = REGISTRIES["solver"].create(config["solver"]["name"])
        aggregator = REGISTRIES["aggregation"].create(config["aggregation"]["name"])
        acceptance = REGISTRIES["acceptance"].create(config["acceptance"]["name"])

        for rollout_iteration in range(iterations):
            view_id = sequence[rollout_iteration % len(sequence)]
            with MemoryTracker(device) as memory, PhaseTimer("total_rollout_step", device=device) as total_timer:
                prepared = prepare_view(config, runtime, initial.audit_views, train_views[view_id], view_id)
                timing_rows.extend({"rollout_iteration": rollout_iteration + 1, **asdict(record)} for record in prepared.timing_records)
                step_snapshot = snapshot_model_and_optimizer(runtime)
                position_snapshot = block.snapshot(runtime.model.gaussians, torch.arange(initial_gaussian_count, device=device))
                try:
                    ids = union_gaussian_ids(prepared.group_set)
                    logger.append_jsonl("anchors.jsonl", {
                        "rollout_iteration": rollout_iteration + 1,
                        "view_id": view_id,
                        **prepared.anchor_set.to_record(),
                    })
                    for group in prepared.group_set.groups:
                        logger.append_jsonl("groups.jsonl", {
                            "rollout_iteration": rollout_iteration + 1,
                            "view_id": view_id,
                            **group.to_record(),
                        })
                    with PhaseTimer("jacobian_and_curvature", device=device) as phase_timer:
                        union_jacobian, union_curvature = compute_union_jacobian(prepared, ids)
                    timing_rows.append({"rollout_iteration": rollout_iteration + 1, **asdict(phase_timer.record)})
                    step_diagnostics, _ = diagnose_groups(
                        union_jacobian,
                        prepared.group_set,
                        int(prepared.residual_data.metadata["channels"]),
                        config["diagnostics"],
                    )
                    group_diagnostic_rows.extend({
                        "rollout_iteration": rollout_iteration + 1,
                        "view_id": view_id,
                        **row,
                    } for row in step_diagnostics)
                    proposals = []
                    solve_time = 0.0
                    solver_config = {**config["solver"], "initial_damping": damping}
                    with PhaseTimer("linear_solve", device=device) as phase_timer:
                        for group in prepared.group_set.groups:
                            group_jacobian, group_curvature = subset_data(union_jacobian, group)
                            proposal = solver.propose_update(group, block, prepared.residual_data, group_jacobian, group_curvature, solver_config)
                            proposals.append(proposal)
                            solve_time += proposal.solve_time_ms
                    timing_rows.append({"rollout_iteration": rollout_iteration + 1, **asdict(phase_timer.record)})
                    with PhaseTimer("update_aggregation", device=device) as phase_timer:
                        update_ids, delta, aggregation_diagnostics = aggregator.aggregate(prepared.group_set.groups, proposals)
                        state = block.gather(runtime.model.gaussians, update_ids)
                        delta = block.project_or_clamp_update(state, delta, {**config["attributes"], "scene_scale": runtime.model.gaussians.training_cameras_extent})
                        predicted = aggregate_predicted_reduction(union_curvature, ids, update_ids, delta)
                    timing_rows.append({"rollout_iteration": rollout_iteration + 1, **asdict(phase_timer.record)})
                    with PhaseTimer("parameter_update", device=device) as phase_timer:
                        block.apply_update(runtime.model.gaussians, update_ids, delta)
                    timing_rows.append({"rollout_iteration": rollout_iteration + 1, **asdict(phase_timer.record)})
                    with PhaseTimer("rerender_and_residual", device=device) as phase_timer:
                        prediction_after = runtime.training_render(prepared.primary_view)
                        residual_provider = REGISTRIES["residual"].create(config["residual"]["name"])
                        residual_after = residual_provider.build(prediction_after, prepared.target, prepared.pixel_indices)
                    timing_rows.append({"rollout_iteration": rollout_iteration + 1, **asdict(phase_timer.record)})
                    loss_before = float(prepared.residual_data.loss.item())
                    loss_after = float(residual_after.loss.item())
                    decision = acceptance.decide(
                        loss_before=loss_before,
                        loss_after=loss_after,
                        predicted_reduction=predicted,
                        damping=damping,
                        config={**config["solver"], **config["acceptance"]},
                    )
                    if decision.accepted:
                        accepted_count += 1
                    else:
                        block.restore(runtime.model.gaussians, torch.arange(initial_gaussian_count, device=device), position_snapshot)
                    damping = decision.next_damping if decision.next_damping is not None else damping
                    with PhaseTimer("remaining_attribute_adam", device=device) as phase_timer:
                        native_loss = apply_remaining_attribute_adam(prepared)
                    timing_rows.append({"rollout_iteration": rollout_iteration + 1, **asdict(phase_timer.record)})
                    metrics: dict[str, float | None] = {"mse": None, "l1": None, "psnr": None, "ssim": None, "lpips": None}
                    if rollout_iteration % evaluation_interval == 0 or rollout_iteration + 1 == iterations:
                        with PhaseTimer("metric_evaluation", device=device) as phase_timer:
                            audit_view = initial.audit_views["primary"][0][2]
                            metrics = image_metrics(runtime.inference_render(audit_view), target_for_view(audit_view))
                        timing_rows.append({"rollout_iteration": rollout_iteration + 1, **asdict(phase_timer.record)})
                    rollout_rows.append({
                        "checkpoint_iteration": runtime.model.num_iterations_trained,
                        "rollout_iteration": rollout_iteration + 1,
                        "view_id": view_id,
                        "gaussian_count": int(runtime.model.gaussians.means.shape[0]),
                        "sampled_loss_before": loss_before,
                        "sampled_loss_after": loss_after,
                        "predicted_reduction": predicted,
                        "actual_reduction": loss_before - loss_after,
                        "gain_ratio": (loss_before - loss_after) / (predicted + 1.0e-12),
                        "accepted": decision.accepted,
                        "damping": damping,
                        "native_loss": native_loss,
                        "update_norm": float(torch.linalg.vector_norm(delta).item()),
                        "solve_time_ms": solve_time,
                        **aggregation_diagnostics,
                        **metrics,
                    })
                except (RuntimeError, FloatingPointError, torch.linalg.LinAlgError) as error:
                    numerical_failures += 1
                    restore_model_and_optimizer(runtime, step_snapshot)
                    logger.append_jsonl("failures.jsonl", {
                        "stage": "rollout_step",
                        "rollout_iteration": rollout_iteration + 1,
                        "view_id": view_id,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    })
                del step_snapshot, position_snapshot
            timing_rows.append({"rollout_iteration": rollout_iteration + 1, **asdict(total_timer.record)})
            memory_rows.append({"rollout_iteration": rollout_iteration + 1, **asdict(memory.record)})
            if int(runtime.model.gaussians.means.shape[0]) != initial_gaussian_count:
                raise RuntimeError("fixed-topology invariant violated: Gaussian count changed")
            if not all(torch.isfinite(parameter).all() for parameter in runtime.model.parameters()):
                raise FloatingPointError("NaN/Inf detected in model parameters")

        logger.write_csv("rollout_metrics.csv", rollout_rows)
        logger.write_csv("group_diagnostics.csv", group_diagnostic_rows)
        logger.write_csv("timing.csv", timing_rows)
        logger.write_csv("memory.csv", memory_rows)
        logger.write_json("checkpoint_metadata.json", {
            **checkpoint_file_metadata(runtime.checkpoint_path),
            "iteration": runtime.model.num_iterations_trained,
            "optimizer_state_restored": runtime.optimizer_state_restored,
            "initial_gaussian_count": initial_gaussian_count,
        })
        logger.write_json("summary.json", {
            "status": "completed",
            "iterations_requested": iterations,
            "iterations_recorded": len(rollout_rows),
            "acceptance_ratio": accepted_count / max(iterations, 1),
            "numerical_failure_count": numerical_failures,
            "fixed_topology_preserved": int(runtime.model.gaussians.means.shape[0]) == initial_gaussian_count,
            "average_allocated_bytes": sum(row["allocated_bytes"] for row in memory_rows) / max(len(memory_rows), 1),
            "peak_allocated_bytes": max((row["peak_allocated_bytes"] for row in memory_rows), default=0),
            "peak_reserved_bytes": max((row["peak_reserved_bytes"] for row in memory_rows), default=0),
            "group_diagnostics": {
                "group_count": len(group_diagnostic_rows),
                "zero_group_count": sum(bool(row["zero_group"]) for row in group_diagnostic_rows),
                "valid_group_count": sum(row["classification"] == "valid" for row in group_diagnostic_rows),
                "zero_group_ratio": sum(bool(row["zero_group"]) for row in group_diagnostic_rows) / max(len(group_diagnostic_rows), 1),
                "valid_group_ratio": sum(row["classification"] == "valid" for row in group_diagnostic_rows) / max(len(group_diagnostic_rows), 1),
            },
        })
        print(logger.directory)
        return 0
    except Exception as error:
        logger.append_jsonl("failures.jsonl", {"stage": "rollout_benchmark", "error_type": type(error).__name__, "message": str(error)})
        logger.write_json("summary.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
