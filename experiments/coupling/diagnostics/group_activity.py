"""Exact-Jacobian group activity and shared-support classification."""

from __future__ import annotations

from typing import Any

import torch

from ...types import GroupSet, JacobianData


def diagnose_groups(
    jacobian_data: JacobianData,
    group_set: GroupSet,
    channels: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    if channels <= 0 or jacobian_data.jacobian.shape[0] % channels:
        raise ValueError("Jacobian residual rows must be divisible by the residual channel count")
    lookup = {gaussian_id: index for index, gaussian_id in enumerate(jacobian_data.gaussian_ids)}
    dimension = jacobian_data.jacobian.shape[1] // max(len(jacobian_data.gaussian_ids), 1)
    all_blocks = jacobian_data.jacobian.reshape(jacobian_data.jacobian.shape[0], len(lookup), dimension)
    global_max = float(torch.linalg.vector_norm(all_blocks, dim=(0, 2)).max().item()) if lookup else 0.0
    absolute = float(config.get("jacobian_block_absolute_threshold", 0.0))
    relative = float(config.get("jacobian_block_relative_threshold", 1.0e-8))
    active_threshold = absolute + relative * global_max
    support_threshold = float(config.get("pixel_support_threshold", 0.0))
    offdiag_threshold = float(config.get("offdiagonal_fro_threshold", 0.0))
    rows: list[dict[str, Any]] = []
    total_pairs = 0
    total_valid_pairs = 0
    for group in group_set.groups:
        indices = torch.tensor([lookup[value] for value in group.member_gaussian_ids], dtype=torch.long, device=all_blocks.device)
        blocks = all_blocks.index_select(1, indices)
        block_norms = torch.linalg.vector_norm(blocks, dim=(0, 2))
        active = block_norms > active_threshold
        pixels = blocks.reshape(-1, channels, group.size, dimension)
        support = pixels.square().sum(dim=(1, 3)) > support_threshold
        group_matrix = blocks.reshape(blocks.shape[0], -1)
        hessian = group_matrix.T @ group_matrix
        valid_pair_count = 0
        shared_support_total = 0
        shared_support_max = 0
        offdiagonal_energy = 0.0
        significant_offdiagonal_count = 0
        pair_count = group.size * (group.size - 1) // 2
        for i in range(group.size):
            for j in range(i + 1, group.size):
                shared = int((support[:, i] & support[:, j]).sum().item())
                shared_support_total += shared
                shared_support_max = max(shared_support_max, shared)
                block = hessian[i * dimension:(i + 1) * dimension, j * dimension:(j + 1) * dimension]
                raw_fro = float(torch.linalg.matrix_norm(block, ord="fro").item())
                offdiagonal_energy += raw_fro * raw_fro
                if bool(active[i] and active[j]) and shared > 0:
                    valid_pair_count += 1
                    if raw_fro > offdiag_threshold:
                        significant_offdiagonal_count += 1
        anchor_index = group.member_gaussian_ids.index(group.anchor_gaussian_id)
        if not bool(active[anchor_index]):
            classification = "anchor_zero"
        elif int(active.sum().item()) < 2 and group.size > 1:
            classification = "member_zero"
        elif valid_pair_count == 0 and group.size > 1:
            classification = "no_shared_support"
        elif significant_offdiagonal_count == 0 and group.size > 1:
            classification = "small_offdiagonal"
        else:
            classification = "valid"
        zero_group = valid_pair_count == 0 and group.size > 1
        rows.append({
            "group_id": group.group_id,
            "anchor_gaussian_id": group.anchor_gaussian_id,
            "member_gaussian_ids": group.member_gaussian_ids,
            "group_size": group.size,
            "classification": classification,
            "zero_group": zero_group,
            "active_member_count": int(active.sum().item()),
            "zero_block_count": int((~active).sum().item()),
            "pair_count": pair_count,
            "valid_pair_count": valid_pair_count,
            "shared_support_pixel_count": shared_support_total,
            "maximum_shared_support_pixel_count": shared_support_max,
            "group_jacobian_norm": float(torch.linalg.vector_norm(group_matrix).item()),
            "anchor_jacobian_norm": float(block_norms[anchor_index].item()),
            "offdiagonal_energy": offdiagonal_energy,
            "significant_offdiagonal_pair_count": significant_offdiagonal_count,
            "active_threshold": active_threshold,
            "block_norms": tuple(float(value) for value in block_norms.detach().cpu().tolist()),
        })
        total_pairs += pair_count
        total_valid_pairs += valid_pair_count
    group_count = len(rows)
    zero_count = sum(bool(row["zero_group"]) for row in rows)
    valid_count = sum(row["classification"] == "valid" for row in rows)
    summary: dict[str, float | int] = {
        "group_count": group_count,
        "zero_group_count": zero_count,
        "valid_group_count": valid_count,
        "zero_group_ratio": zero_count / max(group_count, 1),
        "valid_group_ratio": valid_count / max(group_count, 1),
        "valid_pair_count": total_valid_pairs,
        "pair_count": total_pairs,
        "valid_pair_ratio": total_valid_pairs / max(total_pairs, 1),
        "jacobian_block_active_threshold": active_threshold,
    }
    return rows, summary
