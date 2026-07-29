"""Deterministic view/pixel sampling and projected-footprint approximation."""

from __future__ import annotations

from typing import Any

import torch

from .types import RenderState


def build_render_state(gaussians: Any, view: Any, *, tile_size: int = 16, sigma_extent: float = 3.0) -> RenderState:
    xy, depth, in_frustum = view.project_points(gaussians.means)
    max_scale = gaussians.scales.max(dim=-1).values
    focal = max(float(view.camera.focal_x), float(view.camera.focal_y))
    radius = sigma_extent * focal * max_scale / depth.clamp_min(1.0e-6)
    radius = radius.clamp_min(0.5)
    mins = xy - radius[:, None]
    maxs = xy + radius[:, None]
    width = int(view.camera.width)
    height = int(view.camera.height)
    footprint_visible = (maxs[:, 0] >= 0) & (maxs[:, 1] >= 0) & (mins[:, 0] < width) & (mins[:, 1] < height)
    visible = in_frustum & footprint_visible & torch.isfinite(radius)
    bounds = torch.cat((mins, maxs), dim=-1)
    max_tile_x = (width - 1) // tile_size
    max_tile_y = (height - 1) // tile_size
    tile_mins = torch.floor(mins / tile_size).to(torch.int64)
    tile_maxs = torch.floor(maxs / tile_size).to(torch.int64)
    tile_mins[:, 0].clamp_(0, max_tile_x)
    tile_mins[:, 1].clamp_(0, max_tile_y)
    tile_maxs[:, 0].clamp_(0, max_tile_x)
    tile_maxs[:, 1].clamp_(0, max_tile_y)
    tile_bounds = torch.cat((tile_mins, tile_maxs), dim=-1)
    return RenderState(
        visible_mask=visible,
        projected_xy=xy,
        depth=depth,
        projected_radius=radius,
        projected_bounds=bounds,
        tile_bounds=tile_bounds,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        metadata={"footprint": "isotropic_3sigma_max_scale_approximation", "sigma_extent": sigma_extent},
    )


def sample_top_tiles_and_pixels(render_state: RenderState, config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample tiles with the most approximated overlapping visible footprints."""

    tiles_x = (render_state.image_width + render_state.tile_size - 1) // render_state.tile_size
    tiles_y = (render_state.image_height + render_state.tile_size - 1) // render_state.tile_size
    bounds = render_state.tile_bounds[render_state.visible_mask]
    difference = torch.zeros((tiles_y + 1, tiles_x + 1), dtype=torch.int64, device=render_state.visible_mask.device)
    if bounds.numel():
        ones = torch.ones(bounds.shape[0], dtype=torch.int64, device=bounds.device)
        x0, y0, x1, y1 = bounds.unbind(dim=1)
        difference.index_put_((y0, x0), ones, accumulate=True)
        difference.index_put_((y1 + 1, x0), -ones, accumulate=True)
        difference.index_put_((y0, x1 + 1), -ones, accumulate=True)
        difference.index_put_((y1 + 1, x1 + 1), ones, accumulate=True)
    counts = difference.cumsum(dim=0).cumsum(dim=1)[:tiles_y, :tiles_x].reshape(-1)
    n_tiles = min(int(config.get("top_tiles", 2)), counts.numel())
    tile_ids = torch.topk(counts, k=n_tiles, largest=True, sorted=True).indices
    pixels_per_tile = int(config.get("pixels_per_tile", 16))
    generator = torch.Generator(device="cpu").manual_seed(int(config.get("seed", 0)))
    pixel_ids: list[int] = []
    for tile_id in tile_ids.tolist():
        ty, tx = divmod(tile_id, tiles_x)
        x0, y0 = tx * render_state.tile_size, ty * render_state.tile_size
        x1 = min(x0 + render_state.tile_size, render_state.image_width)
        y1 = min(y0 + render_state.tile_size, render_state.image_height)
        local_count = (x1 - x0) * (y1 - y0)
        selected = torch.randperm(local_count, generator=generator)[:min(pixels_per_tile, local_count)]
        for local in selected.tolist():
            y, x = divmod(local, x1 - x0)
            pixel_ids.append((y0 + y) * render_state.image_width + x0 + x)
    return tile_ids, torch.tensor(pixel_ids, dtype=torch.int64, device=counts.device)
