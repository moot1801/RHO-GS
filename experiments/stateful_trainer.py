"""Optional RHO-GS trainer subclass that emits portable optimizer snapshots."""

from __future__ import annotations

from pathlib import Path

import torch

import Framework
from Methods.Base.utils import training_callback
from Methods.RHO_GS.Trainer import RHOGSTrainer

from .state import build_portable_state, save_portable_state


@Framework.Configurable.configure(
    PORTABLE_STATE=Framework.ConfigParameterList(
        ACTIVE=False,
        ITERATIONS=[1_000, 15_000, 30_000],
        DIRECTORY="portable_states",
    )
)
class StatefulRHOGSTrainer(RHOGSTrainer):
    """Same training callbacks as RHO-GS plus a non-pickled state artifact."""

    @training_callback(active="PORTABLE_STATE.ACTIVE", priority=70)
    @torch.no_grad()
    def save_portable_snapshot(self, iteration: int, _) -> None:
        completed_iteration = iteration + 1
        if completed_iteration not in {int(value) for value in self.PORTABLE_STATE.ITERATIONS}:
            return
        path = self.output_directory / str(self.PORTABLE_STATE.DIRECTORY) / f"iteration_{completed_iteration:06d}.rho_state.pt"
        state = build_portable_state(
            model=self.model,
            optimizer=self.model.gaussians.optimizer,
            iteration=completed_iteration,
            sampler=self.train_sampler,
            metadata={"capture_phase": "post_optimizer", "fixed_topology": False},
        )
        save_portable_state(path, state)
