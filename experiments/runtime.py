"""Thin adapter between reference experiments and NeRFICG/RHO_GS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .state import load_portable_state, restore_rng_state


@dataclass(slots=True)
class Runtime:
    framework: Any
    dataset: Any
    model: Any
    renderer: Any
    optimizer_state_restored: bool
    checkpoint_path: Path
    base_config_path: Path

    def training_render(self, view: Any) -> torch.Tensor:
        self.model.train()
        return self.renderer.render_image_training(
            view=view,
            update_densification_info=False,
            bg_color=view.camera.background_color,
        )

    @torch.no_grad()
    def inference_render(self, view: Any) -> torch.Tensor:
        self.model.eval()
        return self.renderer.render_image_inference(view, to_chw=True)["rgb"]

    def native_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from Optim.Losses.DSSIM import fused_dssim

        return 0.8 * torch.nn.functional.l1_loss(prediction, target) + 0.2 * fused_dssim(prediction, target)


def infer_base_config(checkpoint: Path) -> Path:
    candidates = [checkpoint.parent.parent / "training_config.yaml", checkpoint.parent / "training_config.yaml"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot infer training_config.yaml from {checkpoint}")


def _camera_extent(dataset: Any) -> float:
    dataset.train()
    centers = torch.stack([view.position for view in dataset])
    return float(1.1 * torch.linalg.vector_norm(centers - centers.mean(dim=0), dim=1).max())


def load_runtime(config: dict[str, Any]) -> Runtime:
    """Initialize Framework once and load either model-only or portable state."""

    import Framework
    from Implementations import Datasets as DI
    from Implementations import Methods as MI

    checkpoint = Path(str(config["checkpoint"])).resolve()
    base_config = Path(config["base_config"]).resolve() if config.get("base_config") else infer_base_config(checkpoint)
    Framework.setup(config_path=str(base_config), require_custom_config=True)
    scale = config.get("evaluation", {}).get("image_scale_factor")
    if scale is not None:
        Framework.config.DATASET.IMAGE_SCALE_FACTOR = float(scale)
    dataset = DI.get_dataset(Framework.config.GLOBAL.DATASET_TYPE, Framework.config.DATASET.PATH)

    portable: dict[str, Any] | None = None
    try:
        candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(candidate, dict) and candidate.get("format") == "rho_gs_coupling_state":
            portable = candidate
    except OSError:
        raise
    if portable is None:
        relative_checkpoint = str(checkpoint.relative_to(Framework.Directories.NERFICG_ROOT))
        model = MI.get_model(Framework.config.GLOBAL.METHOD_TYPE, checkpoint=relative_checkpoint)
    else:
        model = MI.get_model(Framework.config.GLOBAL.METHOD_TYPE, checkpoint=None, name=portable["model_metadata"]["model_name"])
        missing, unexpected = model.load_state_dict(portable["model_state_dict"], strict=False)
        for key in unexpected:
            target = model
            components = key.split(".")
            for component in components[:-1]:
                target = getattr(target, component)
            name = components[-1]
            value = portable["model_state_dict"][key]
            if name in target._parameters:
                setattr(target, name, torch.nn.Parameter(value.to(Framework.config.GLOBAL.DEFAULT_DEVICE)))
            else:
                if hasattr(target, name):
                    delattr(target, name)
                target.register_buffer(name, value.to(Framework.config.GLOBAL.DEFAULT_DEVICE))
        if missing and not unexpected:
            raise RuntimeError(f"portable state is missing model keys: {missing}")
        model.num_iterations_trained = int(portable["model_metadata"]["num_iterations_trained"])
        model.gaussians.active_sh_degree = int(portable["model_metadata"]["active_sh_degree"])
        model.gaussians.active_sh_bases = (model.gaussians.active_sh_degree + 1) ** 2
    renderer = MI.get_renderer(Framework.config.GLOBAL.METHOD_TYPE, model)

    model.gaussians.training_setup(Framework.config.TRAINING, _camera_extent(dataset))
    restored = portable is not None and portable.get("optimizer_state_dict") is not None
    if restored:
        model.gaussians.optimizer.load_state_dict(portable["optimizer_state_dict"])
        restore_rng_state(portable["rng_state"])
    return Runtime(Framework, dataset, model, renderer, restored, checkpoint, base_config)


def target_for_view(view: Any) -> torch.Tensor:
    from Datasets.utils import apply_background_color

    target = view.rgb
    if view.alpha is not None:
        target = apply_background_color(target, view.alpha, view.camera.background_color)
    return target
