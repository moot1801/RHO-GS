from __future__ import annotations

import unittest

import torch

from experiments.coupling.curvature import edge_budget_oracle_capture_metrics, edge_capture_metrics
from experiments.coupling.grouping import KNN3DGrouping, RandomInUniverseGrouping
from experiments.coupling.universe import FixedCubeUniverse
from experiments.types import AnchorSet, GaussianUniverse, RenderState


class FakeGaussians:
    def __init__(self):
        self.means = torch.nn.Parameter(torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [1.0, 0.0, 0.0],
        ]))


def render_state() -> RenderState:
    return RenderState(
        visible_mask=torch.tensor([True, True, False, True]),
        projected_xy=torch.zeros(4, 2),
        depth=torch.ones(4),
        projected_radius=torch.ones(4),
        projected_bounds=torch.zeros(4, 4),
        tile_bounds=torch.zeros(4, 4, dtype=torch.int64),
        image_width=16,
        image_height=16,
        tile_size=16,
        metadata={"restricted_to_sampled_tiles": [0]},
    )


class UniverseCaptureTest(unittest.TestCase):
    def test_fixed_cube_is_deterministic_and_precedes_visibility_filter(self):
        config = {
            "center_xyz": [0.0, 0.0, 0.0],
            "half_extent": 0.2,
            "visible_only": True,
            "minimum_gaussians": 2,
            "maximum_gaussians": 4,
            "overflow_policy": "error",
        }
        first = FixedCubeUniverse().build(FakeGaussians(), render_state(), None, config)
        second = FixedCubeUniverse().build(FakeGaussians(), render_state(), None, config)
        self.assertEqual(first.gaussian_ids, (0, 1))
        self.assertEqual(first.gaussian_id_hash, second.gaussian_id_hash)
        self.assertEqual(first.metadata["selection_order"], "cube_then_optional_filters_then_overflow")

    def test_fixed_cube_overflow_is_explicit(self):
        config = {
            "center_gaussian_id": 0,
            "half_extent": 0.5,
            "visible_only": False,
            "minimum_gaussians": 2,
            "maximum_gaussians": 2,
            "overflow_policy": "error",
        }
        with self.assertRaisesRegex(RuntimeError, "above maximum_gaussians"):
            FixedCubeUniverse().build(FakeGaussians(), render_state(), None, config)

    def test_knn_and_random_members_stay_inside_universe(self):
        universe = GaussianUniverse("fixed_cube", (0, 2), "fixed")
        anchors = AnchorSet("fixed", (0,), (0.0,), 3, (0, 2), "anchors")
        config = {"seed": 3, "group_size": 2, "candidate_pool_size": 4, "overlapping": True}
        knn = KNN3DGrouping().build_groups(FakeGaussians(), render_state(), None, config, anchors, universe)
        random = RandomInUniverseGrouping().build_groups(FakeGaussians(), render_state(), None, config, anchors, universe)
        self.assertEqual(set(knn.groups[0].member_gaussian_ids), {0, 2})
        self.assertEqual(set(random.groups[0].member_gaussian_ids), {0, 2})

    def test_random_grouping_respects_contributor_member_filter(self):
        universe = GaussianUniverse("fixed_cube", (0, 1, 2), "fixed")
        anchors = AnchorSet("fixed", (0,), (0.0,), 3, (0, 2), "anchors", filter_group_members=True)
        result = RandomInUniverseGrouping().build_groups(
            FakeGaussians(), render_state(), None,
            {"seed": 3, "group_size": 3, "overlapping": True},
            anchors, universe,
        )
        self.assertEqual(set(result.groups[0].member_gaussian_ids), {0, 2})

    def test_random_grouping_can_restrict_members_to_visible_universe(self):
        universe = GaussianUniverse("fixed_cube", (0, 1, 2), "fixed")
        anchors = AnchorSet("fixed", (0,), (0.0,), 3, (0, 1, 2), "anchors")
        result = RandomInUniverseGrouping().build_groups(
            FakeGaussians(), render_state(), None,
            {"seed": 3, "group_size": 3, "visible_only": True, "overlapping": True},
            anchors, universe,
        )
        self.assertEqual(set(result.groups[0].member_gaussian_ids), {0, 1})

    def test_fixed_anchor_outside_universe_is_rejected_by_grouping(self):
        universe = GaussianUniverse("fixed_cube", (0, 1), "fixed")
        anchors = AnchorSet("fixed", (3,), (0.0,), 0, (3,), "anchors")
        with self.assertRaisesRegex(ValueError, "outside fixed universe"):
            KNN3DGrouping().build_groups(
                FakeGaussians(), render_state(), None,
                {"group_size": 2, "candidate_pool_size": 4, "overlapping": True},
                anchors, universe,
            )

    def test_explicit_edge_capture_and_equal_budget_oracle(self):
        hessian = torch.zeros(9, 9, dtype=torch.float64)
        for edge, value in {((0, 1)): 1.0, ((0, 2)): 2.0, ((1, 2)): 3.0}.items():
            i, j = edge
            block = torch.eye(3, dtype=hessian.dtype) * value
            hessian[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = block
            hessian[j * 3:(j + 1) * 3, i * 3:(i + 1) * 3] = block.T
        eligible = {(0, 1), (0, 2)}
        capture = edge_capture_metrics(hessian, {(0, 1)}, eligible, 3)
        oracle = edge_budget_oracle_capture_metrics(hessian, eligible, 1, 3)
        self.assertAlmostEqual(capture["capture_ratio"], 0.2)
        self.assertAlmostEqual(oracle["capture_ratio"], 0.8)
        self.assertEqual(capture["selected_edge_count"], oracle["selected_edge_count"])


if __name__ == "__main__":
    unittest.main()
