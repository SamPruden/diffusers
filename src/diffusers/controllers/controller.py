from __future__ import annotations
from typing import Optional, Generic, Dict
from typing_extensions import ParamSpec
from abc import ABC, abstractmethod
import torch

class StepPatcher(ABC):
    """
    Base class for step patchers.
    Step patchers map a hook and sample to a new patched sample value.
    Implementers should derive from either `ValueStepPatcher` or `ResidualStepPatcher`.
    """

    def __call__(self, hook: str, sample: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            The new sample value after patching.
        """
        value = self.value(hook, sample)
        return value if value is not None else sample

    @abstractmethod
    def value(self, hook: str, sample: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Returns:
            The new sample value after patching.
            May be `None` if the value is unchanged.
        """
        pass

    @abstractmethod
    def residual(self, hook: str, sample: torch.Tensor) -> Optional[torch.Tensor]:
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

    def residual(self, hook: str, sample: torch.Tensor) -> Optional[torch.Tensor]:
        value = self.value(hook, sample)
        return (value - sample) if value is not None else None


class ResidualStepPatcher(StepPatcher):
    """
    A `StepPatcher` defined in terms of the patched residuals.
    """

    def value(self, hook: str, sample: torch.Tensor) -> Optional[torch.Tensor]:
        residual = self.residual(hook, sample)
        return (sample + residual) if residual is not None else None


class DictValueStepPatcher(ValueStepPatcher):
    """
    A `StepPatcher` defined by a dictionary mapping hook strs to new patched values.
    """
    
    def __init__(self, dict: Dict[str, torch.Tensor]):
        self.hook_dict = dict

    def value(self, hook: str, sample: torch.Tensor) -> Optional[torch.Tensor]:
        return self.hook_dict.get(hook)


class DictResidualStepPatcher(ResidualStepPatcher):
    """
    A `StepPatcher` defined by a dictionary matching hook strs to patched residuals.
    """
    
    def __init__(self, dict: Dict[str, torch.Tensor]):
        self.hook_dict = dict

    def residual(self, hook: str, sample: torch.Tensor) -> Optional[torch.Tensor]:
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
