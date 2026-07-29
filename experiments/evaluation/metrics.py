"""Metrics that do not initiate network downloads."""

from __future__ import annotations

import math

import torch


def image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | None]:
    prediction = prediction.clamp(0.0, 1.0)
    target = target.to(prediction).clamp(0.0, 1.0)
    mse = float(torch.mean((prediction - target) ** 2).item())
    l1 = float(torch.mean(torch.abs(prediction - target)).item())
    psnr = -10.0 * math.log10(max(mse, 1.0e-12))
    ssim: float | None
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure
        ssim = float(structural_similarity_index_measure(prediction[None], target[None], data_range=1.0).item())
    except (ImportError, RuntimeError, ValueError):
        ssim = None
    return {"mse": mse, "l1": l1, "psnr": psnr, "ssim": ssim, "lpips": None}
