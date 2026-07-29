"""Checkpoint-only sampled coupling analysis."""

from __future__ import annotations

import sys
from dataclasses import asdict

import torch

from ..coupling.curvature import coupling_capture_ratio, group_normalized_coupling, normalized_pairwise_coupling
from ..evaluation import MemoryTracker, PhaseTimer
from ..result_logger import ResultLogger
from .common import compute_union_jacobian, parse_config, prepare, union_gaussian_ids


def main(argv: list[str] | None = None) -> int:
    config = parse_config(sys.argv[1:] if argv is None else argv)
    logger = ResultLogger(config)
    logger.initialize_expected_artifacts()
    timings: list[dict] = []
    memory_rows: list[dict] = []
    try:
        prepared = prepare(config)
        timings.extend(asdict(record) for record in prepared.timing_records)
        ids = union_gaussian_ids(prepared.group_set)
        for group in prepared.group_set.groups:
            logger.append_jsonl("groups.jsonl", {"view_id": config["evaluation"]["primary_view_id"], **group.to_record()})
        device = prepared.runtime.model.gaussians.means.device
        with MemoryTracker(device) as memory, PhaseTimer("jacobian_and_curvature", device=device) as timer:
            jacobian, curvature = compute_union_jacobian(prepared, ids)
        timings.append(asdict(timer.record))
        memory_rows.append({"phase": "jacobian_and_curvature", **asdict(memory.record)})
        damping = float(config["solver"].get("initial_damping", 0.0))
        coupling_rows: list[dict] = []
        n_blocks = len(ids)
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
                })
        lookup = {value: index for index, value in enumerate(ids)}
        mapped_groups = [tuple(lookup[value] for value in group.member_gaussian_ids) for group in prepared.group_set.groups]
        capture = coupling_capture_ratio(curvature.hessian, mapped_groups, 3)
        epsilon_g = group_normalized_coupling(curvature.hessian, 3, damping)
        eigenvalues = torch.linalg.eigvalsh(0.5 * (curvature.hessian + curvature.hessian.T)).detach().cpu()
        logger.write_csv("coupling_metrics.csv", coupling_rows)
        logger.write_csv("timing.csv", timings)
        logger.write_csv("memory.csv", memory_rows)
        logger.write_json("checkpoint_metadata.json", {
            "path": str(prepared.runtime.checkpoint_path),
            "iteration": prepared.runtime.model.num_iterations_trained,
            "gaussian_count": prepared.runtime.model.gaussians.means.shape[0],
            "optimizer_state_restored": prepared.runtime.optimizer_state_restored,
            "base_config": str(prepared.runtime.base_config_path),
        })
        logger.write_json("summary.json", {
            "status": "completed",
            "sampled_gaussian_count": len(ids),
            "sampled_residual_count": jacobian.residual.numel(),
            "capture_ratio": capture,
            "epsilon_G": epsilon_g,
            "condition_number": float(torch.linalg.cond(curvature.hessian + damping * torch.eye(curvature.hessian.shape[0], device=device)).item()),
            "minimum_eigenvalue": float(eigenvalues.min().item()),
            "maximum_eigenvalue": float(eigenvalues.max().item()),
            "symmetry_relative_error": curvature.metadata["symmetry_relative_error"],
            "sampling": prepared.residual_data.metadata,
        })
        print(logger.directory)
        return 0
    except Exception as error:
        logger.append_jsonl("failures.jsonl", {"stage": "analyze_checkpoint", "error_type": type(error).__name__, "message": str(error)})
        logger.write_json("summary.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
