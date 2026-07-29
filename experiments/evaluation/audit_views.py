"""Deterministic primary, nearby, and global view selection."""

from __future__ import annotations

import torch


def _pose_distance(primary, candidate) -> float:
    translation = torch.linalg.vector_norm(primary.position - candidate.position)
    orientation = 1.0 - torch.dot(primary.forward, candidate.forward).clamp(-1.0, 1.0)
    return float((translation + orientation).item())


def select_audit_views(dataset, *, primary_view_id: int, nearby_count: int, global_count: int) -> dict[str, list[tuple[str, int, object]]]:
    train_views = list(dataset.train())
    if not 0 <= primary_view_id < len(train_views):
        raise IndexError(f"primary training view {primary_view_id} outside [0, {len(train_views)})")
    primary = train_views[primary_view_id]
    nearby_ids = sorted((index for index in range(len(train_views)) if index != primary_view_id), key=lambda index: (_pose_distance(primary, train_views[index]), index))[:nearby_count]
    test_views = list(dataset.test())
    global_ids: list[int] = []
    if test_views and global_count > 0:
        global_ids.append(0)
        while len(global_ids) < min(global_count, len(test_views)):
            candidate = max(
                (index for index in range(len(test_views)) if index not in global_ids),
                key=lambda index: min(_pose_distance(test_views[index], test_views[selected]) for selected in global_ids),
            )
            global_ids.append(candidate)
    return {
        "primary": [("train", primary_view_id, primary)],
        "nearby": [("train", index, train_views[index]) for index in nearby_ids],
        "global": [("test", index, test_views[index]) for index in global_ids],
    }
