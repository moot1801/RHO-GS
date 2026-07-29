"""Acceptance decisions with explicit damping transitions."""

from __future__ import annotations

from ...registry import REGISTRIES
from ...types import AcceptanceDecision


@REGISTRIES["acceptance"].register("always_accept")
class AlwaysAccept:
    name = "always_accept"

    def decide(self, *, loss_before: float, loss_after: float, predicted_reduction: float, damping: float, config: dict) -> AcceptanceDecision:
        return AcceptanceDecision(True, "always_accept", None, damping)


@REGISTRIES["acceptance"].register("accept_if_primary_loss_decreases")
class PrimaryLossDecreaseAcceptance:
    name = "accept_if_primary_loss_decreases"

    def decide(self, *, loss_before: float, loss_after: float, predicted_reduction: float, damping: float, config: dict) -> AcceptanceDecision:
        accepted = loss_after < loss_before
        return AcceptanceDecision(accepted, "primary_loss_decreased" if accepted else "primary_loss_not_decreased", None, damping)


@REGISTRIES["acceptance"].register("lm_gain_ratio")
class LMGainRatioAcceptance:
    name = "lm_gain_ratio"

    def decide(self, *, loss_before: float, loss_after: float, predicted_reduction: float, damping: float, config: dict) -> AcceptanceDecision:
        actual = loss_before - loss_after
        epsilon = float(config.get("epsilon", 1.0e-12))
        gain = actual / (predicted_reduction + epsilon)
        accepted = predicted_reduction > 0.0 and gain >= float(config.get("minimum_gain_ratio", 0.0))
        factor = float(config.get("damping_decrease_factor", 1.0 / 3.0)) if accepted else float(config.get("damping_increase_factor", 10.0))
        next_damping = min(max(damping * factor, float(config.get("minimum_damping", 1.0e-8))), float(config.get("maximum_damping", 1.0e8)))
        return AcceptanceDecision(accepted, "gain_ratio_accepted" if accepted else "gain_ratio_rejected", gain, next_damping)
