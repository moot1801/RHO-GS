from __future__ import annotations

import unittest

import torch

from experiments.coupling.grouping import KNN3DGrouping, VisibleOverlapKNNGrouping
from experiments.coupling.parameter_blocks import PositionBlock
from experiments.sampling import sample_top_tiles_and_pixels
from experiments.types import AnchorSet, RenderState


class FakeGaussians:
    def __init__(self):
        self.means = torch.nn.Parameter(torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]))


def render_state() -> RenderState:
    return RenderState(
        visible_mask=torch.tensor([True, True, True, False]),
        projected_xy=torch.zeros(4, 2),
        depth=torch.ones(4),
        projected_radius=torch.ones(4),
        projected_bounds=torch.zeros(4, 4),
        tile_bounds=torch.tensor([[0, 0, 1, 1], [1, 0, 2, 1], [4, 4, 5, 5], [0, 0, 0, 0]]),
        image_width=96,
        image_height=96,
        tile_size=16,
    )


class GroupingParameterTest(unittest.TestCase):
    def test_grouping_is_deterministic(self):
        config = {"seed": 7, "sampled_anchor_count": 3, "maximum_groups": 3, "group_size": 2, "candidate_pool_size": 4, "overlapping": True}
        first = KNN3DGrouping().build_groups(FakeGaussians(), None, None, config)
        second = KNN3DGrouping().build_groups(FakeGaussians(), None, None, config)
        self.assertEqual([group.member_gaussian_ids for group in first.groups], [group.member_gaussian_ids for group in second.groups])

    def test_visible_overlap_excludes_non_overlapping_neighbor(self):
        config = {"seed": 0, "sampled_anchor_count": 3, "maximum_groups": 3, "group_size": 3, "candidate_pool_size": 4, "minimum_shared_tiles": 1, "overlapping": True}
        result = VisibleOverlapKNNGrouping().build_groups(FakeGaussians(), render_state(), None, config)
        for group in result.groups:
            if group.anchor_gaussian_id in (0, 1):
                self.assertNotIn(2, group.member_gaussian_ids)

    def test_anchor_set_can_filter_group_members(self):
        config = {"group_size": 3, "candidate_pool_size": 4, "minimum_shared_tiles": 1, "overlapping": True}
        anchors = AnchorSet(
            strategy="test",
            anchor_gaussian_ids=(0,),
            scores=(1.0,),
            seed=0,
            candidate_gaussian_ids=(0, 1),
            candidate_hash="test",
            filter_group_members=True,
        )
        result = VisibleOverlapKNNGrouping().build_groups(FakeGaussians(), render_state(), None, config, anchors)
        self.assertEqual(result.groups[0].member_gaussian_ids, (0, 1))
        self.assertEqual(result.groups[0].metadata["anchor_candidate_hash"], "test")
        self.assertTrue(result.groups[0].metadata["member_contributor_filter"])

    def test_snapshot_update_restore_and_clip(self):
        gaussians = FakeGaussians()
        block = PositionBlock()
        indices = torch.tensor([0, 2])
        snapshot = block.snapshot(gaussians, indices)
        delta = block.project_or_clamp_update(snapshot, torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]), {"max_step_norm": 0.5})
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(delta, dim=-1), torch.tensor([0.5, 0.5])))
        block.apply_update(gaussians, indices, delta)
        self.assertFalse(torch.equal(block.gather(gaussians, indices), snapshot))
        block.restore(gaussians, indices, snapshot)
        self.assertTrue(torch.equal(block.gather(gaussians, indices), snapshot))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for default-device regression")
    def test_sampling_with_cuda_default_device(self):
        previous_device = torch.get_default_device()
        try:
            torch.set_default_device("cuda")
            state = render_state()
            first = sample_top_tiles_and_pixels(state, {"top_tiles": 1, "pixels_per_tile": 4, "seed": 7})
            second = sample_top_tiles_and_pixels(state, {"top_tiles": 1, "pixels_per_tile": 4, "seed": 7})
            grouping_config = {
                "seed": 7,
                "sampled_anchor_count": 3,
                "maximum_groups": 3,
                "group_size": 2,
                "candidate_pool_size": 4,
                "overlapping": True,
            }
            first_groups = KNN3DGrouping().build_groups(FakeGaussians(), None, None, grouping_config)
            second_groups = KNN3DGrouping().build_groups(FakeGaussians(), None, None, grouping_config)
            self.assertEqual(first[0].device.type, "cuda")
            self.assertEqual(first[1].device.type, "cuda")
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(torch.equal(first[1], second[1]))
            self.assertEqual(
                [group.member_gaussian_ids for group in first_groups.groups],
                [group.member_gaussian_ids for group in second_groups.groups],
            )
        finally:
            torch.set_default_device(previous_device)


if __name__ == "__main__":
    unittest.main()
