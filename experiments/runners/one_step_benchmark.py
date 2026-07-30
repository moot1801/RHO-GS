"""Independent one-step counterfactual benchmark from a shared checkpoint."""

from __future__ import annotations

import sys
from dataclasses import asdict

import torch

from ..coupling.diagnostics import diagnose_groups
from ..evaluation import MemoryTracker, PhaseTimer
from ..registry import REGISTRIES
from ..result_logger import ResultLogger
from .common import (
    apply_remaining_attribute_adam,
    aggregate_predicted_reduction,
    checkpoint_file_metadata,
    compute_union_jacobian,
    evaluate_views,
    parse_config,
    prepare,
    restore_model_and_optimizer,
    snapshot_model_and_optimizer,
    subset_data,
    union_gaussian_ids,
)


def _audit_summary(before: dict, after: dict) -> dict[str, float]:
    deltas = [after[key]["loss"] - before[key]["loss"] for key in before]
    return {
        "bad_view_ratio": sum(value > 0 for value in deltas) / max(len(deltas), 1),
        "worst_view_degradation": max(deltas, default=0.0),
        "mean_audit_loss_change": sum(deltas) / max(len(deltas), 1),
    }


def main(argv: list[str] | None = None) -> int:
    config = parse_config(sys.argv[1:] if argv is None else argv)
    logger = ResultLogger(config)
    logger.initialize_expected_artifacts()
    timing_rows: list[dict] = []
    memory_rows: list[dict] = []
    result_rows: list[dict] = []
    total_memory = None
    total_timer = None
    try:
        prepared = prepare(config)
        timing_rows.extend(asdict(record) for record in prepared.timing_records)
        runtime = prepared.runtime
        device = runtime.model.gaussians.means.device
        total_memory = MemoryTracker(device)
        total_timer = PhaseTimer("total_counterfactual", device=device)
        total_memory.__enter__()
        total_timer.__enter__()
        original = snapshot_model_and_optimizer(runtime)
        with PhaseTimer("audit_evaluation_before", device=device) as timer:
            before_audit = evaluate_views(prepared)
        timing_rows.append(asdict(timer.record))
        ids = union_gaussian_ids(prepared.group_set)
        logger.append_jsonl("anchors.jsonl", {"view_id": config["evaluation"]["primary_view_id"], **prepared.anchor_set.to_record()})
        for group in prepared.group_set.groups:
            logger.append_jsonl("groups.jsonl", {"view_id": config["evaluation"]["primary_view_id"], **group.to_record()})
        with PhaseTimer("jacobian_and_curvature", device=device) as timer:
            union_jacobian, union_curvature = compute_union_jacobian(prepared, ids)
        timing_rows.append(asdict(timer.record))
        diagnostic_rows, diagnostic_summary = diagnose_groups(
            union_jacobian,
            prepared.group_set,
            int(prepared.residual_data.metadata["channels"]),
            config["diagnostics"],
        )

        block = REGISTRIES["parameter_block"].create(config["attributes"]["name"])
        solver = REGISTRIES["solver"].create(config["solver"]["name"])
        independent_solver = REGISTRIES["solver"].create("per_gaussian_lm")
        proposals = []
        relative_differences: list[float] = []
        cosine_similarities: list[float] = []
        with PhaseTimer("linear_solve", device=device) as timer:
            for group in prepared.group_set.groups:
                group_jacobian, group_curvature = subset_data(union_jacobian, group)
                solver_config = dict(config["solver"])
                solver_config.update({"_optimizer": runtime.model.gaussians.optimizer, "_parameter": runtime.model.gaussians.means})
                proposal = solver.propose_update(group, block, prepared.residual_data, group_jacobian, group_curvature, solver_config)
                independent = independent_solver.propose_update(group, block, prepared.residual_data, group_jacobian, group_curvature, config["solver"])
                proposals.append(proposal)
                difference = torch.linalg.vector_norm(proposal.delta - independent.delta) / torch.linalg.vector_norm(independent.delta).clamp_min(1.0e-12)
                cosine = torch.nn.functional.cosine_similarity(proposal.delta.reshape(1, -1), independent.delta.reshape(1, -1)).item()
                relative_differences.append(float(difference.item()))
                cosine_similarities.append(float(cosine))
        timing_rows.append(asdict(timer.record))

        with PhaseTimer("update_aggregation", device=device) as timer:
            aggregation = REGISTRIES["aggregation"].create(config["aggregation"]["name"])
            update_ids, delta, aggregation_diagnostics = aggregation.aggregate(prepared.group_set.groups, proposals)
            unclipped = delta.clone()
            state = block.snapshot(runtime.model.gaussians, update_ids)
            delta = block.project_or_clamp_update(state, delta, {**config["attributes"], "scene_scale": runtime.model.gaussians.training_cameras_extent})
            predicted_total = aggregate_predicted_reduction(union_curvature, ids, update_ids, delta)
            clipped = not torch.equal(delta, unclipped)
        timing_rows.append(asdict(timer.record))
        sampled_loss_before = float(prepared.residual_data.loss.item())
        with PhaseTimer("parameter_update", device=device) as timer:
            block.apply_update(runtime.model.gaussians, update_ids, delta)
        timing_rows.append(asdict(timer.record))
        with PhaseTimer("rerender_and_residual", device=device) as timer:
            prediction_after = runtime.training_render(prepared.primary_view)
            residual_provider = REGISTRIES["residual"].create(config["residual"]["name"])
            residual_after = residual_provider.build(prediction_after, prepared.target, prepared.pixel_indices)
        timing_rows.append(asdict(timer.record))
        sampled_loss_after = float(residual_after.loss.item())
        actual = sampled_loss_before - sampled_loss_after
        acceptance = REGISTRIES["acceptance"].create(config["acceptance"]["name"])
        decision = acceptance.decide(
            loss_before=sampled_loss_before,
            loss_after=sampled_loss_after,
            predicted_reduction=predicted_total,
            damping=float(config["solver"].get("initial_damping", 0.0)),
            config={**config["solver"], **config["acceptance"]},
        )
        with PhaseTimer("audit_evaluation_position", device=device) as timer:
            position_audit = evaluate_views(prepared)
        timing_rows.append(asdict(timer.record))
        if not decision.accepted:
            block.restore(runtime.model.gaussians, update_ids, state)
        with PhaseTimer("remaining_attribute_adam", device=device) as timer:
            hybrid_native_loss = apply_remaining_attribute_adam(prepared)
        timing_rows.append(asdict(timer.record))
        with PhaseTimer("audit_evaluation_hybrid", device=device) as timer:
            hybrid_audit = evaluate_views(prepared)
        timing_rows.append(asdict(timer.record))
        position_summary = _audit_summary(before_audit, position_audit)
        hybrid_summary = _audit_summary(before_audit, hybrid_audit)
        result_rows.append({
            "checkpoint_iteration": runtime.model.num_iterations_trained,
            "view_id": config["evaluation"]["primary_view_id"],
            "gradient_norm": float(torch.linalg.vector_norm(union_jacobian.gradient).item()),
            "update_norm": float(torch.linalg.vector_norm(delta).item()),
            "update_clipped": clipped,
            "relative_to_independent_mean": sum(relative_differences) / len(relative_differences),
            "cosine_to_independent_mean": sum(cosine_similarities) / len(cosine_similarities),
            "predicted_reduction": predicted_total,
            "actual_reduction": actual,
            "gain_ratio": actual / (predicted_total + 1.0e-12),
            "primary_sampled_loss_before": sampled_loss_before,
            "primary_sampled_loss_after": sampled_loss_after,
            "accepted": decision.accepted,
            "acceptance_reason": decision.reason,
            "hybrid_native_loss_before_step": hybrid_native_loss,
            **{f"position_{key}": value for key, value in position_summary.items()},
            **{f"hybrid_{key}": value for key, value in hybrid_summary.items()},
            **aggregation_diagnostics,
            "optimizer_state_restored": runtime.optimizer_state_restored,
        })
        restore_model_and_optimizer(runtime, original)
        total_timer.__exit__(None, None, None)
        total_memory.__exit__(None, None, None)
        timing_rows.append(asdict(total_timer.record))
        memory_rows.append({"phase": "total_counterfactual", **asdict(total_memory.record)})
        logger.write_csv("one_step_results.csv", result_rows)
        logger.write_csv("group_diagnostics.csv", diagnostic_rows)
        logger.write_csv("timing.csv", timing_rows)
        logger.write_csv("memory.csv", memory_rows)
        logger.write_json("checkpoint_metadata.json", {
            **checkpoint_file_metadata(runtime.checkpoint_path),
            "iteration": runtime.model.num_iterations_trained,
            "optimizer_state_restored": runtime.optimizer_state_restored,
        })
        logger.write_json("summary.json", {
            "status": "completed",
            **result_rows[0],
            "audit_view_ids": list(before_audit),
            "anchor_selection": prepared.anchor_set.to_record(),
            "group_diagnostics": diagnostic_summary,
        })
        print(logger.directory)
        return 0
    except Exception as error:
        if total_timer is not None and total_timer.record is None:
            total_timer.__exit__(type(error), error, error.__traceback__)
        if total_memory is not None and total_memory.record is None:
            total_memory.__exit__(type(error), error, error.__traceback__)
        logger.append_jsonl("failures.jsonl", {"stage": "one_step_benchmark", "error_type": type(error).__name__, "message": str(error)})
        logger.write_json("summary.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
