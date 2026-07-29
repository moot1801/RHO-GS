"""Portable experiment state format, independent from pickled Trainer objects."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAT_NAME = "rho_gs_coupling_state"
FORMAT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_sampler_state(sampler: Any | None) -> dict[str, Any] | None:
    if sampler is None:
        return None
    id_sampler = getattr(sampler, "id_sampler", sampler)
    return {
        "class": type(id_sampler).__name__,
        "num_elements": int(id_sampler.num_elements),
        "indices": id_sampler.indices.detach().cpu().clone(),
        "current_id": int(id_sampler.current_id),
    }


def restore_sampler_state(sampler: Any, state: dict[str, Any] | None) -> None:
    if state is None:
        return
    id_sampler = getattr(sampler, "id_sampler", sampler)
    if int(id_sampler.num_elements) != int(state["num_elements"]):
        raise ValueError("sampler size differs from portable state")
    id_sampler.indices = state["indices"].clone()
    id_sampler.current_id = int(state["current_id"])


def build_portable_state(model: Any, optimizer: Any | None, iteration: int, sampler: Any | None, metadata: dict[str, Any]) -> dict[str, Any]:
    gaussian_count = int(model.gaussians.means.shape[0])
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "model_metadata": {
            "model_name": model.model_name,
            "creation_date": model.creation_date,
            "num_iterations_trained": int(iteration),
            "output_directory": str(model.output_directory),
            "active_sh_degree": int(model.gaussians.active_sh_degree),
            "max_sh_degree": int(model.gaussians.max_sh_degree),
            "gaussian_count": gaussian_count,
        },
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "sampler_state": capture_sampler_state(sampler),
        "gaussian_identity": torch.arange(gaussian_count, dtype=torch.int64),
        "metadata": metadata,
    }


def save_portable_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_portable_state(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get("format") != FORMAT_NAME or state.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported portable state: {path}")
    return state


def is_portable_state(path: Path) -> bool:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError):
        return False
    return isinstance(state, dict) and state.get("format") == FORMAT_NAME
