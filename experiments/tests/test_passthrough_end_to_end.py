from __future__ import annotations

import subprocess
import sys
import unittest

import torch

from experiments.coupling.aggregation import AnchorOnlyAggregation
from experiments.coupling.curvature import GaussNewtonAssembler
from experiments.coupling.jacobian import RepeatedVJPJacobian
from experiments.coupling.parameter_blocks import PositionBlock
from experiments.coupling.residuals import RGBL2SampledResidual
from experiments.coupling.solvers import GroupLevenbergMarquardt
from experiments.types import Group


class FakeGaussians:
    def __init__(self):
        self.means = torch.nn.Parameter(torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float64))


class PassthroughEndToEndTest(unittest.TestCase):
    def test_experiment_import_does_not_activate_nerficg_method(self):
        script = "import experiments, sys; assert 'Framework' not in sys.modules; assert 'Methods.RHO_GS' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_synthetic_one_step_decreases_loss(self):
        gaussians = FakeGaussians()

        def render():
            return gaussians.means.reshape(1, 1, 3)

        target = torch.zeros_like(render())
        residual_provider = RGBL2SampledResidual()
        residual = residual_provider.build(render().detach(), target, torch.tensor([0]))
        jacobian = RepeatedVJPJacobian().compute(render, gaussians.means, (0,), residual, {})
        curvature = GaussNewtonAssembler().assemble(jacobian, 3)
        group = Group(0, 0, (0,))
        proposal = GroupLevenbergMarquardt().propose_update(
            group,
            PositionBlock(),
            residual,
            jacobian,
            curvature,
            {"initial_damping": 1.0e-3, "damping_mode": "identity", "diagnostic_dtype": "float64"},
        )
        ids, delta, _ = AnchorOnlyAggregation().aggregate([group], [proposal])
        PositionBlock().apply_update(gaussians, ids, delta)
        after = residual_provider.build(render().detach(), target, torch.tensor([0]))
        self.assertLess(float(after.loss), float(residual.loss))


if __name__ == "__main__":
    unittest.main()
