from __future__ import annotations

import unittest

import torch

from experiments.coupling.aggregation import AnchorOnlyAggregation, WeightedAverageAggregation
from experiments.coupling.parameter_blocks import PositionBlock
from experiments.coupling.solvers import AdamAdapter, GroupLevenbergMarquardt, PerGaussianLevenbergMarquardt
from experiments.runners.common import aggregate_predicted_reduction
from experiments.types import CurvatureData, Group, JacobianData, ResidualData, UpdateProposal


def proposal(ids, values, predicted=1.0):
    return UpdateProposal(ids, torch.tensor(values, dtype=torch.float64), predicted, 0.1, 1.0, 1, True, "ok", 0.0)


class SolverAggregationTest(unittest.TestCase):
    def setUp(self):
        self.group = Group(0, 10, (10, 11))
        hessian = torch.eye(6, dtype=torch.float64) * 2.0
        gradient = torch.arange(1, 7, dtype=torch.float64)
        self.curvature = CurvatureData((10, 11), hessian, gradient, torch.stack((hessian[:3, :3], hessian[3:, 3:])))
        self.jacobian = JacobianData((10, 11), torch.eye(6, dtype=torch.float64), torch.ones(6, dtype=torch.float64), gradient)
        self.residual = ResidualData(torch.ones(6), torch.ones(2, 3), torch.zeros(2, 3), torch.tensor([0, 1]), torch.tensor(3.0))
        self.config = {"initial_damping": 0.1, "damping_mode": "identity", "diagnostic_dtype": "float64", "condition_number_threshold": 1.0e12}

    def test_lm_solution_residual_and_predicted_reduction(self):
        result = GroupLevenbergMarquardt().propose_update(self.group, PositionBlock(), self.residual, self.jacobian, self.curvature, self.config)
        expected = -self.curvature.gradient / 2.1
        self.assertTrue(torch.allclose(result.delta.reshape(-1), expected))
        self.assertGreater(result.predicted_reduction, 0.0)
        self.assertTrue(result.converged)

    def test_per_gaussian_matches_block_diagonal_group(self):
        group_result = GroupLevenbergMarquardt().propose_update(self.group, PositionBlock(), self.residual, self.jacobian, self.curvature, self.config)
        per_result = PerGaussianLevenbergMarquardt().propose_update(self.group, PositionBlock(), self.residual, self.jacobian, self.curvature, self.config)
        self.assertTrue(torch.allclose(group_result.delta, per_result.delta))

    def test_aggregation_and_union_prediction(self):
        groups = [Group(0, 10, (10, 11)), Group(1, 11, (11, 12))]
        proposals = [proposal((10, 11), [[1, 0, 0], [2, 0, 0]]), proposal((11, 12), [[4, 0, 0], [8, 0, 0]])]
        ids, delta, diagnostics = AnchorOnlyAggregation().aggregate(groups, proposals)
        self.assertEqual(ids.tolist(), [10, 11])
        self.assertEqual(delta[:, 0].tolist(), [1.0, 4.0])
        ids_weighted, delta_weighted, _ = WeightedAverageAggregation().aggregate(groups, proposals)
        self.assertEqual(ids_weighted.tolist(), [10, 11, 12])
        self.assertAlmostEqual(float(delta_weighted[1, 0]), 3.0)
        curvature = CurvatureData((10, 11, 12), torch.eye(9), -torch.ones(9), torch.stack([torch.eye(3)] * 3))
        predicted = aggregate_predicted_reduction(curvature, (10, 11, 12), ids, delta.float())
        expected_delta = torch.tensor([1.0, 0, 0, 4.0, 0, 0, 0, 0, 0])
        expected = -torch.dot(-torch.ones(9), expected_delta) - 0.5 * torch.dot(expected_delta, expected_delta)
        self.assertAlmostEqual(predicted, float(expected))

    def test_adam_rejects_missing_moment_state_by_default(self):
        parameter = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float64))
        optimizer = torch.optim.Adam([{"params": [parameter], "lr": 0.1, "name": "means"}])
        with self.assertRaises(RuntimeError):
            AdamAdapter().propose_update(
                self.group,
                PositionBlock(),
                self.residual,
                self.jacobian,
                self.curvature,
                {"_optimizer": optimizer, "_parameter": parameter, "allow_cold_start": False},
            )


if __name__ == "__main__":
    unittest.main()
