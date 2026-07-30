"""Dense sampled-subset Gauss-Newton assembly used as correctness oracle."""

from __future__ import annotations

import torch

from ...registry import REGISTRIES
from ...types import CurvatureData, JacobianData


@REGISTRIES["curvature"].register("gauss_newton")
class GaussNewtonAssembler:
    name = "gauss_newton"

    def assemble(self, jacobian_data: JacobianData, block_dimension: int) -> CurvatureData:
        jacobian = jacobian_data.jacobian
        hessian = jacobian.T @ jacobian
        n_blocks = len(jacobian_data.gaussian_ids)
        expected = n_blocks * block_dimension
        if hessian.shape != (expected, expected):
            raise ValueError(f"Hessian shape {hessian.shape} does not match {n_blocks} blocks of dimension {block_dimension}")
        blocks = torch.stack([
            hessian[i * block_dimension:(i + 1) * block_dimension, i * block_dimension:(i + 1) * block_dimension]
            for i in range(n_blocks)
        ]) if n_blocks else hessian.new_zeros((0, block_dimension, block_dimension))
        symmetry_error = float(torch.linalg.vector_norm(hessian - hessian.T) / torch.linalg.vector_norm(hessian).clamp_min(1.0e-12))
        return CurvatureData(
            gaussian_ids=jacobian_data.gaussian_ids,
            hessian=hessian,
            gradient=jacobian_data.gradient,
            diagonal_blocks=blocks,
            metadata={"symmetry_relative_error": symmetry_error, "residual_count": jacobian.shape[0]},
        )


def _inverse_sqrt_psd(matrix: torch.Tensor, damping: float) -> torch.Tensor:
    regularized = 0.5 * (matrix + matrix.T) + damping * torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    eigenvalues, eigenvectors = torch.linalg.eigh(regularized)
    inverse_sqrt = eigenvalues.clamp_min(torch.finfo(matrix.dtype).eps).rsqrt()
    return (eigenvectors * inverse_sqrt.unsqueeze(0)) @ eigenvectors.T


def normalized_pairwise_coupling(hessian: torch.Tensor, i: int, j: int, block_dimension: int, damping: float) -> dict[str, float | torch.Tensor]:
    si = slice(i * block_dimension, (i + 1) * block_dimension)
    sj = slice(j * block_dimension, (j + 1) * block_dimension)
    hii, hij, hjj = hessian[si, si], hessian[si, sj], hessian[sj, sj]
    normalized = _inverse_sqrt_psd(hii, damping) @ hij @ _inverse_sqrt_psd(hjj, damping)
    return {
        "matrix": normalized,
        "spectral_norm": float(torch.linalg.matrix_norm(normalized, ord=2).item()),
        "frobenius_norm": float(torch.linalg.matrix_norm(normalized, ord="fro").item()),
        "raw_frobenius_norm": float(torch.linalg.matrix_norm(hij, ord="fro").item()),
    }


def group_normalized_coupling(hessian: torch.Tensor, block_dimension: int, damping: float) -> float:
    diagonal = torch.zeros_like(hessian)
    for start in range(0, hessian.shape[0], block_dimension):
        section = slice(start, start + block_dimension)
        diagonal[section, section] = hessian[section, section]
    off_diagonal = hessian - diagonal
    scale = _inverse_sqrt_psd(diagonal, damping)
    return float(torch.linalg.matrix_norm(scale @ off_diagonal @ scale, ord=2).item())


def coupling_capture_ratio(hessian: torch.Tensor, groups: list[tuple[int, ...]], block_dimension: int) -> float:
    n_blocks = hessian.shape[0] // block_dimension
    included = {tuple(sorted((i, j))) for group in groups for i in group for j in group if i < j}
    eligible = {(i, j) for i in range(n_blocks) for j in range(i + 1, n_blocks)}
    return float(edge_capture_metrics(hessian, included, eligible, block_dimension)["capture_ratio"])


def _canonical_edges(edges: set[tuple[int, int]], n_blocks: int) -> set[tuple[int, int]]:
    canonical: set[tuple[int, int]] = set()
    for first, second in edges:
        i, j = sorted((int(first), int(second)))
        if i == j:
            continue
        if i < 0 or j >= n_blocks:
            raise ValueError(f"edge ({first}, {second}) is outside {n_blocks} Hessian blocks")
        canonical.add((i, j))
    return canonical


def _offdiagonal_energy(hessian: torch.Tensor, edge: tuple[int, int], block_dimension: int) -> torch.Tensor:
    i, j = edge
    block = hessian[
        i * block_dimension:(i + 1) * block_dimension,
        j * block_dimension:(j + 1) * block_dimension,
    ]
    return torch.linalg.matrix_norm(block, ord="fro").square()


def edge_capture_metrics(
    hessian: torch.Tensor,
    selected_edges: set[tuple[int, int]],
    eligible_edges: set[tuple[int, int]],
    block_dimension: int,
) -> dict[str, float | int]:
    """Measure raw off-diagonal energy capture on an explicit fixed edge universe."""

    n_blocks = hessian.shape[0] // block_dimension
    eligible = _canonical_edges(eligible_edges, n_blocks)
    selected = _canonical_edges(selected_edges, n_blocks) & eligible
    total = hessian.new_zeros(())
    captured = hessian.new_zeros(())
    for edge in sorted(eligible):
        energy = _offdiagonal_energy(hessian, edge, block_dimension)
        total += energy
        if edge in selected:
            captured += energy
    total_value = float(total.item())
    captured_value = float(captured.item())
    capture_ratio = 0.0 if total_value == 0.0 else float((captured / total).item())
    return {
        "eligible_edge_count": len(eligible),
        "selected_edge_count": len(selected),
        "total_offdiagonal_energy": total_value,
        "captured_offdiagonal_energy": captured_value,
        "capture_ratio": capture_ratio,
    }


def edge_budget_oracle_capture_metrics(
    hessian: torch.Tensor,
    eligible_edges: set[tuple[int, int]],
    edge_budget: int,
    block_dimension: int,
) -> dict[str, float | int]:
    """Loose upper bound from the highest-energy eligible edges at the same global budget."""

    n_blocks = hessian.shape[0] // block_dimension
    eligible = _canonical_edges(eligible_edges, n_blocks)
    budget = min(max(0, int(edge_budget)), len(eligible))
    energies = sorted(
        (float(_offdiagonal_energy(hessian, edge, block_dimension).item()) for edge in eligible),
        reverse=True,
    )
    total = sum(energies)
    captured = sum(energies[:budget])
    return {
        "eligible_edge_count": len(eligible),
        "selected_edge_count": budget,
        "total_offdiagonal_energy": total,
        "captured_offdiagonal_energy": captured,
        "capture_ratio": 0.0 if total == 0.0 else captured / total,
        "oracle_basis": "global_top_raw_offdiagonal_energy_at_equal_edge_budget",
    }
