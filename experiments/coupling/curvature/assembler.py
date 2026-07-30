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
    total = hessian.new_zeros(())
    captured = hessian.new_zeros(())
    included = {tuple(sorted((i, j))) for group in groups for i in group for j in group if i < j}
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            block = hessian[i * block_dimension:(i + 1) * block_dimension, j * block_dimension:(j + 1) * block_dimension]
            energy = torch.linalg.matrix_norm(block, ord="fro").square()
            total += energy
            if (i, j) in included:
                captured += energy
    if float(total.item()) == 0.0:
        return 0.0
    return float((captured / total).item())
