from typing import Iterable, Optional
from diffusers.controllers.controller import Controller, StepPatcher, ResidualStepPatcher, TControllerParams
import torch


class ClampedAccumulatorStepPatcher(ResidualStepPatcher):
    def __init__(self, controller: Controller, patchers: Iterable[StepPatcher], clamp_factor: float):
        super().__init__(controller)
        self.patchers = list(patchers)
        self.clamp_factor = clamp_factor

    # TODO: Performance
    def residual(self, hook: str, sample: torch.Tensor) -> Optional[torch.Tensor]:
        residuals = [patcher.residual(hook, sample) for patcher in self.patchers]
        residuals = [residual for residual in residuals if residual]

        if not residuals: return None
        if len(residuals) == 1: return min(1, self.clamp_factor) * residuals[0]

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
            self,
            [controller(*args, **kwargs) for controller in self.controllers],
            self.clamp_factor
        )


