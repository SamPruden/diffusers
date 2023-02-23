from __future__ import annotations
from typing import Optional, Generic, Iterable, Dict
from typing_extensions import ParamSpec
from abc import ABC, abstractmethod
import torch

class StepPatcher(ABC):
    """
    Base class for step patchers.
    Step patchers map a hook and sample to a new patched sample value.
    Implementers should derive from either `ValueStepPatcher` or `ResidualStepPatcher`.
    """

    # TODO: Not sure about this
    # Passing a controller reference here helps advanced accumulators.
    # It's possible that an accumulator may want to enforce some different
    # rule for a specific controller.
    # However, this means that patchers now 100% require a controller to exist, which isn't great.
    # It also adds to boilerplate, without a clear justification.
    # We're covering the "somebody may want this in the future" scenario, not a real need.
    # But if we don't do that now, then adding this would be a backcompat breaking change.
    # ... Unless we make it optional
    # But then people might be annoyed that pre-existing implementations don't do this.
    # Alternative: What if we put a `name: str` here instead?
    def __init__(self, controller: Controller):
        self.controller = Controller

    def __call__(self, hook: str, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns:
            The new sample value after patching.
        """
        return self.value(hook, sample) or sample

    @abstractmethod
    def value(self, hook: str, sample: torch.FloatTensor) -> Optional[torch.FloatTensor]:
        """
        Returns:
            The new sample value after patching.
            May be `None` if the value is unchanged.
        """
        pass

    @abstractmethod
    def residual(self, hook: str, sample: torch.FloatTensor) -> Optional[torch.FloatTensor]:
        """
        Returns:
            A residual to be added to the original sample to create the new value.
            May be `None` if the residual is zero.
        """
        pass


class ValueStepPatcher(StepPatcher):
    """
    A `StepPatcher` defined in terms of the new patched values.
    """

    def residual(self, hook: str, sample: torch.FloatTensor) -> Optional[torch.FloatTensor]:
        value = self.value(hook, sample)
        return value - sample if value else None


class ResidualStepPatcher(StepPatcher):
    """
    A `StepPatcher` defined in terms of the patch residuals.
    """

    def value(self, hook: str, sample: torch.FloatTensor) -> torch.FloatTensor:
        residual = self.residual(hook, sample)
        return sample + residual if residual else sample


class DictValueStepPatcher(ValueStepPatcher):
    """
    A `StepPatcher` defined by a dictionary mapping hook strs to new patched values.
    """
    
    def __init__(self, controller: Controller, dict: Dict[str, torch.FloatTensor]):
        super().__init__(controller)
        self.hook_dict = dict

    def value(self, hook: str, sample: torch.FloatTensor) -> Optional[torch.FloatTensor]:
        return self.hook_dict.get(hook)


class DictResidualStepPatcher(ResidualStepPatcher):
    """
    A `StepPatcher` defined by a dictionary matching hook strs to patched residuals.
    """
    
    def __init__(self, controller: Controller, dict: Dict[str, torch.FloatTensor]):
        super().__init__(controller)
        self.hook_dict = dict

    def residual(self, hook: str, sample: torch.FloatTensor) -> Optional[torch.FloatTensor]:
        return self.hook_dict.get(hook)


TControllerParams = ParamSpec("TControllerParams")
class Controller(Generic[TControllerParams], ABC):
    """
    Base class for controllers.
    Controllers take in user parameters and model state, and create a new StepPatcher each step.
    """

    @abstractmethod
    def __call__(self, *args: TControllerParams.args, **kwargs: TControllerParams.kwargs) -> StepPatcher:
        pass
