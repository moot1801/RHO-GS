from __future__ import annotations

import unittest

import torch

from experiments.coupling.curvature import GaussNewtonAssembler, coupling_capture_ratio, edge_capture_metrics
from experiments.coupling.jacobian import FiniteDifferenceJacobian, RepeatedVJPJacobian, jacobian_relative_error
from experiments.coupling.residuals import RGBL2SampledResidual


class JacobianCurvatureTest(unittest.TestCase):
    def setUp(self):
        self.parameter = torch.nn.Parameter(torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.5, -0.2]], dtype=torch.float64))
        self.matrix = torch.tensor([
            [1.0, 0.0, 0.5, 0.2, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 1.5, 0.0, 0.0, 0.4],
            [0.2, 0.1, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.3, 0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.4, 0.0, 0.2, 1.0],
        ], dtype=torch.float64)

    def render(self):
        return (self.matrix @ self.parameter.reshape(-1)).reshape(2, 1, 3)

    def test_finite_difference_and_vjp_agree(self):
        prediction = self.render().detach()
        residual = RGBL2SampledResidual().build(prediction, torch.zeros_like(prediction), torch.tensor([0, 1]))
        analytic = RepeatedVJPJacobian().compute(self.render, self.parameter, (0, 1), residual, {})
        finite = FiniteDifferenceJacobian().compute(self.render, self.parameter, (0, 1), residual, {"finite_difference_epsilon": 1.0e-5})
        self.assertLess(jacobian_relative_error(analytic.jacobian, finite.jacobian), 1.0e-8)

    def test_jtr_matches_backward_and_jtj_is_symmetric_psd(self):
        prediction = self.render()
        residual = RGBL2SampledResidual().build(prediction.detach(), torch.zeros_like(prediction), torch.tensor([0, 1]))
        analytic = RepeatedVJPJacobian().compute(self.render, self.parameter, (0, 1), residual, {})
        loss = 0.5 * torch.sum(self.render() ** 2)
        gradient = torch.autograd.grad(loss, self.parameter)[0].reshape(-1)
        self.assertTrue(torch.allclose(analytic.gradient, gradient, atol=1.0e-10, rtol=1.0e-10))
        curvature = GaussNewtonAssembler().assemble(analytic, 3)
        self.assertLess(curvature.metadata["symmetry_relative_error"], 1.0e-12)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(curvature.hessian).min()), -1.0e-10)
        self.assertTrue(torch.allclose(curvature.diagonal_blocks[0], curvature.hessian[:3, :3]))

    def test_capture_ratio_preserves_small_nonzero_energy(self):
        hessian = torch.zeros(6, 6, dtype=torch.float32)
        hessian[:3, 3:] = torch.eye(3) * 1.0e-4
        hessian[3:, :3] = hessian[:3, 3:].T
        self.assertEqual(coupling_capture_ratio(hessian, [(0, 1)], 3), 1.0)
        self.assertEqual(coupling_capture_ratio(torch.zeros_like(hessian), [(0, 1)], 3), 0.0)

    def test_explicit_capture_preserves_tensor_dtype_division(self):
        hessian = torch.zeros(9, 9, dtype=torch.float32)
        hessian[:3, 3:6] = torch.eye(3) * 0.1
        hessian[3:6, :3] = hessian[:3, 3:6].T
        hessian[:3, 6:9] = torch.eye(3) * 0.2
        hessian[6:9, :3] = hessian[:3, 6:9].T
        metrics = edge_capture_metrics(hessian, {(0, 1)}, {(0, 1), (0, 2)}, 3)
        selected = torch.linalg.matrix_norm(hessian[:3, 3:6], ord="fro").square()
        total = selected + torch.linalg.matrix_norm(hessian[:3, 6:9], ord="fro").square()
        expected = float((selected / total).item())
        self.assertEqual(metrics["capture_ratio"], expected)


if __name__ == "__main__":
    unittest.main()
