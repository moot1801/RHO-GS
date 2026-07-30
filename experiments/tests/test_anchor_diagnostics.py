from __future__ import annotations

import unittest

import torch

from experiments.coupling.anchor_selection import FixedAnchorSelection, RandomContributor, estimate_position_jacobian_energy
from experiments.coupling.diagnostics import diagnose_groups
from experiments.coupling.residuals import RGBL2SampledResidual
from experiments.types import Group, GroupSet, JacobianData, RenderState


class FakeGaussians:
    def __init__(self):
        self.means = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.float64))


def render_state(device: torch.device | str = "cpu") -> RenderState:
    return RenderState(
        visible_mask=torch.tensor([True, True, True], device=device),
        projected_xy=torch.zeros(3, 2, device=device),
        depth=torch.ones(3, device=device),
        projected_radius=torch.ones(3, device=device),
        projected_bounds=torch.zeros(3, 4, device=device),
        tile_bounds=torch.zeros(3, 4, dtype=torch.long, device=device),
        image_width=2,
        image_height=1,
        tile_size=16,
    )


class AnchorDiagnosticsTest(unittest.TestCase):
    def test_probe_and_random_contributor_are_reproducible(self):
        gaussians = FakeGaussians()
        matrix = torch.zeros(6, 9, dtype=torch.float64)
        matrix[0, 0] = 1.0
        matrix[1, 3] = 2.0

        def render():
            return (matrix @ gaussians.means.reshape(-1)).reshape(2, 1, 3)

        residual = RGBL2SampledResidual().build(render().detach(), torch.zeros_like(render()), torch.tensor([0, 1]))
        scores, metadata = estimate_position_jacobian_energy(
            render, gaussians.means, residual, {"probe_count": 3, "probe_seed": 11}
        )
        self.assertTrue(torch.allclose(scores.cpu(), torch.tensor([1.0, 4.0, 0.0], dtype=torch.float64)))
        self.assertEqual(metadata["probe_count"], 3)
        config = {
            "seed": 7,
            "sampled_anchor_count": 2,
            "maximum_anchors": 2,
            "relative_score_threshold": 0.0,
            "absolute_score_threshold": 0.0,
            "filter_group_members": True,
        }
        first = RandomContributor().select(gaussians, render_state(), None, config, scores)
        second = RandomContributor().select(gaussians, render_state(), None, config, scores)
        self.assertEqual(first.anchor_gaussian_ids, second.anchor_gaussian_ids)
        self.assertEqual(first.candidate_gaussian_ids, (0, 1))
        self.assertEqual(first.candidate_hash, second.candidate_hash)

    def test_fixed_ids_preserve_order_and_validate_range(self):
        gaussians = FakeGaussians()
        selected = FixedAnchorSelection().select(
            gaussians,
            render_state(),
            None,
            {"fixed_ids": [2, 0], "require_geometry_eligible": True, "seed": 0},
        )
        self.assertEqual(selected.anchor_gaussian_ids, (2, 0))
        with self.assertRaises(ValueError):
            FixedAnchorSelection().select(gaussians, render_state(), None, {"fixed_ids": [3]}, None)

    def test_zero_group_and_shared_support_classification(self):
        jacobian = torch.zeros(6, 9, dtype=torch.float64)
        jacobian[0, 0] = 1.0
        jacobian[0, 3] = 2.0
        data = JacobianData(
            gaussian_ids=(0, 1, 2),
            jacobian=jacobian,
            residual=torch.zeros(6, dtype=torch.float64),
            gradient=torch.zeros(9, dtype=torch.float64),
        )
        groups = GroupSet(
            strategy="test",
            groups=[Group(0, 0, (0, 1)), Group(1, 2, (2, 1))],
            seed=0,
            overlapping=True,
            directed=False,
        )
        rows, summary = diagnose_groups(data, groups, 3, {"jacobian_block_relative_threshold": 1.0e-8})
        self.assertEqual(rows[0]["classification"], "valid")
        self.assertEqual(rows[0]["valid_pair_count"], 1)
        self.assertEqual(rows[1]["classification"], "anchor_zero")
        self.assertTrue(rows[1]["zero_group"])
        self.assertEqual(summary["zero_group_count"], 1)
        self.assertEqual(summary["valid_pair_count"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for probe regression")
    def test_probe_with_cuda_default_device(self):
        previous_device = torch.get_default_device()
        try:
            torch.set_default_device("cuda")
            gaussians = FakeGaussians()
            matrix = torch.eye(9, dtype=torch.float64, device="cuda")[:6]

            def render():
                return (matrix @ gaussians.means.reshape(-1)).reshape(2, 1, 3)

            residual = RGBL2SampledResidual().build(render().detach(), torch.zeros_like(render()), torch.tensor([0, 1], device="cuda"))
            scores, _ = estimate_position_jacobian_energy(render, gaussians.means, residual, {"probe_count": 2, "probe_seed": 3})
            self.assertEqual(scores.device.type, "cuda")
            self.assertTrue(torch.equal(scores, torch.tensor([3.0, 3.0, 0.0], dtype=torch.float64, device="cuda")))
        finally:
            torch.set_default_device(previous_device)


if __name__ == "__main__":
    unittest.main()
