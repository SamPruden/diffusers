from typing import Iterable, Optional
from .controller import Controller, StepPatcher, ResidualStepPatcher, TControllerParams
import torch


class SumAccumulatorStepPatcher(ResidualStepPatcher):
    def __init__(self, patchers: Iterable[StepPatcher]):
        self.patchers = list(patchers)

    def residual(self, hook: str, sample: torch.Tensor) -> Optional[torch.Tensor]:
        residuals = (patcher.residual(hook, sample) for patcher in self.patchers)
        residuals = (residual for residual in residuals if residual is not None)
        return sum(residuals) or None

# TODO: Add weighting
class SumAccumulatorController(Controller[TControllerParams]):
    """
    Simply sums the residuals of all controllers.
    """
    def __init__(self, controllers: Iterable[Controller[TControllerParams]]):
        self.controllers = list(controllers)

    def __call__(self, *args: TControllerParams.args, **kwargs: TControllerParams.kwargs) -> SumAccumulatorStepPatcher:
        return SumAccumulatorStepPatcher(controller(*args, **kwargs) for controller in self.controllers)


class ClampedAccumulatorStepPatcher(ResidualStepPatcher):
    def __init__(self, patchers: Iterable[StepPatcher], clamp_factor: float):
        self.patchers = list(patchers)
        self.clamp_factor = clamp_factor

    # TODO: Performance
    def residual(self, hook: str, sample: torch.Tensor) -> Optional[torch.Tensor]:
        residuals = (patcher.residual(hook, sample) for patcher in self.patchers)
        residuals = [residual for residual in residuals if residual is not None]

        if not residuals: return None
        if len(residuals) == 1: return min(1, self.clamp_factor) * residuals[0]

        # TODO: Review performance
        residuals = torch.stack(residuals, -1)
        return torch.clamp(
            residuals.sum(-1),
            self.clamp_factor * residuals.min(-1),
            self.clamp_factor * residuals.max(-1)
        )

# TODO: Add weighting
class ClampedAccumulatorController(Controller[TControllerParams]):
    def __init__(self, controllers: Iterable[Controller[TControllerParams]], clamp_factor: float):
        self.controllers = list(controllers)
        self.clamp_factor = clamp_factor

    def __call__(self, *args: TControllerParams.args, **kwargs: TControllerParams.kwargs) -> ClampedAccumulatorStepPatcher:
        return ClampedAccumulatorStepPatcher(
            (controller(*args, **kwargs) for controller in self.controllers),
            self.clamp_factor
        )
