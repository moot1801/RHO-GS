"""Checkpoint-only sampled coupling analysis."""

from __future__ import annotations

import sys
from dataclasses import asdict

import torch

from ..coupling.curvature import edge_budget_oracle_capture_metrics, edge_capture_metrics, group_normalized_coupling, normalized_pairwise_coupling
from ..coupling.diagnostics import diagnose_groups
from ..coupling.universe import gaussian_id_hash
from ..evaluation import MemoryTracker, PhaseTimer
from ..result_logger import ResultLogger
from .common import checkpoint_file_metadata, compute_union_jacobian, parse_config, prepare, union_gaussian_ids


def main(argv: list[str] | None = None) -> int:
    config = parse_config(sys.argv[1:] if argv is None else argv)
    logger = ResultLogger(config)
    logger.initialize_expected_artifacts()
    timings: list[dict] = []
    memory_rows: list[dict] = []
    try:
        prepared = prepare(config)
        timings.extend(asdict(record) for record in prepared.timing_records)
        induced_ids = union_gaussian_ids(prepared.group_set)
        ids = prepared.universe.gaussian_ids if prepared.universe is not None else induced_ids
        missing = sorted(set(induced_ids) - set(ids))
        if missing:
            raise RuntimeError(f"grouping returned {len(missing)} Gaussian IDs outside the evaluation universe")
        logger.append_jsonl("anchors.jsonl", {"view_id": config["evaluation"]["primary_view_id"], **prepared.anchor_set.to_record()})
        for group in prepared.group_set.groups:
            logger.append_jsonl("groups.jsonl", {"view_id": config["evaluation"]["primary_view_id"], **group.to_record()})
        device = prepared.runtime.model.gaussians.means.device
        with MemoryTracker(device) as memory, PhaseTimer("jacobian_and_curvature", device=device) as timer:
            jacobian, curvature = compute_union_jacobian(prepared, ids)
        timings.append(asdict(timer.record))
        memory_rows.append({"phase": "jacobian_and_curvature", **asdict(memory.record)})
        diagnostic_rows, diagnostic_summary = diagnose_groups(
            jacobian,
            prepared.group_set,
            int(prepared.residual_data.metadata["channels"]),
            config["diagnostics"],
        )
        damping = float(config["solver"].get("initial_damping", 0.0))
        coupling_rows: list[dict] = []
        n_blocks = len(ids)
        lookup = {value: index for index, value in enumerate(ids)}
        mapped_groups = [tuple(lookup[value] for value in group.member_gaussian_ids) for group in prepared.group_set.groups]
        all_edges = {(i, j) for i in range(n_blocks) for j in range(i + 1, n_blocks)}
        induced_indices = {lookup[value] for value in induced_ids}
        induced_edges = {(i, j) for i in induced_indices for j in induced_indices if i < j}
        group_edges = {tuple(sorted((i, j))) for group in mapped_groups for i in group for j in group if i < j}
        anchor_indices = {lookup[value] for value in prepared.anchor_set.anchor_gaussian_ids}
        anchor_eligible_edges = {
            tuple(sorted((anchor, other)))
            for anchor in anchor_indices
            for other in range(n_blocks)
            if anchor != other
        }
        selected_anchor_edges = {
            tuple(sorted((lookup[group.anchor_gaussian_id], lookup[member])))
            for group in prepared.group_set.groups
            for member in group.member_gaussian_ids
            if member != group.anchor_gaussian_id
        }
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                values = normalized_pairwise_coupling(curvature.hessian, i, j, 3, damping)
                coupling_rows.append({
                    "checkpoint_iteration": prepared.runtime.model.num_iterations_trained,
                    "view_id": config["evaluation"]["primary_view_id"],
                    "gaussian_i": ids[i],
                    "gaussian_j": ids[j],
                    "c_ij": values["spectral_norm"],
                    "c_ij_fro": values["frobenius_norm"],
                    "raw_offdiag_fro": values["raw_frobenius_norm"],
                    "eligible_anchor_edge": (i, j) in anchor_eligible_edges,
                    "selected_anchor_edge": (i, j) in selected_anchor_edges,
                    "selected_group_edge": (i, j) in group_edges,
                    "inside_induced_union": (i, j) in induced_edges,
                })
        induced_capture = edge_capture_metrics(curvature.hessian, group_edges, induced_edges, 3)
        fixed_group_capture = edge_capture_metrics(curvature.hessian, group_edges, all_edges, 3)
        fixed_anchor_capture = edge_capture_metrics(curvature.hessian, selected_anchor_edges, anchor_eligible_edges, 3)
        oracle_capture = edge_budget_oracle_capture_metrics(
            curvature.hessian,
            anchor_eligible_edges,
            int(fixed_anchor_capture["selected_edge_count"]),
            3,
        )
        oracle_ratio = float(oracle_capture["capture_ratio"])
        relative_to_oracle = 0.0 if oracle_ratio == 0.0 else float(fixed_anchor_capture["capture_ratio"]) / oracle_ratio
        fixed_enabled = prepared.universe is not None
        primary_capture = fixed_group_capture if fixed_enabled else induced_capture
        epsilon_g = group_normalized_coupling(curvature.hessian, 3, damping)
        eigenvalues = torch.linalg.eigvalsh(0.5 * (curvature.hessian + curvature.hessian.T)).detach().cpu()
        logger.write_csv("coupling_metrics.csv", coupling_rows)
        logger.write_csv("group_diagnostics.csv", diagnostic_rows)
        logger.write_csv("timing.csv", timings)
        logger.write_csv("memory.csv", memory_rows)
        universe_record = prepared.universe.to_record() if prepared.universe is not None else {
            "strategy": "group_induced_legacy",
            "gaussian_ids": ids,
            "gaussian_count": len(ids),
            "gaussian_id_hash": gaussian_id_hash(ids),
            "metadata": {"constructed_after_grouping": True},
        }
        logger.write_json("universe.json", universe_record)
        logger.write_json("checkpoint_metadata.json", {
            **checkpoint_file_metadata(prepared.runtime.checkpoint_path),
            "iteration": prepared.runtime.model.num_iterations_trained,
            "gaussian_count": prepared.runtime.model.gaussians.means.shape[0],
            "optimizer_state_restored": prepared.runtime.optimizer_state_restored,
            "base_config": str(prepared.runtime.base_config_path),
        })
        logger.write_json("summary.json", {
            "status": "completed",
            "sampled_gaussian_count": len(ids),
            "induced_group_union_count": len(induced_ids),
            "sampled_residual_count": jacobian.residual.numel(),
            "capture_ratio": primary_capture["capture_ratio"],
            "capture_ratio_definition": "fixed_universe_group" if fixed_enabled else "induced_union_legacy",
            "induced_union_capture": induced_capture,
            "fixed_universe_group_capture": fixed_group_capture if fixed_enabled else None,
            "fixed_universe_anchor_capture": fixed_anchor_capture if fixed_enabled else None,
            "edge_budget_oracle_capture": oracle_capture if fixed_enabled else None,
            "oracle_relative_capture": relative_to_oracle if fixed_enabled else None,
            "evaluation_universe": universe_record,
            "epsilon_G": epsilon_g,
            "condition_number": float(torch.linalg.cond(curvature.hessian + damping * torch.eye(curvature.hessian.shape[0], device=device)).item()),
            "minimum_eigenvalue": float(eigenvalues.min().item()),
            "maximum_eigenvalue": float(eigenvalues.max().item()),
            "symmetry_relative_error": curvature.metadata["symmetry_relative_error"],
            "sampling": prepared.residual_data.metadata,
            "anchor_selection": prepared.anchor_set.to_record(),
            "group_diagnostics": diagnostic_summary,
        })
        print(logger.directory)
        return 0
    except Exception as error:
        logger.append_jsonl("failures.jsonl", {"stage": "analyze_checkpoint", "error_type": type(error).__name__, "message": str(error)})
        logger.write_json("summary.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
