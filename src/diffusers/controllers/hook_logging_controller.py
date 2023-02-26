from typing import Optional
from .controller import Controller, TControllerParams, ValueStepPatcher
import torch

class HookLoggingStepPatcher(ValueStepPatcher):
    def value(self, hook: str, sample: torch.FloatTensor) -> Optional[torch.FloatTensor]:
        print(hook)
        return None

class HookLoggingController(Controller[TControllerParams]):
    """
    Simply prints every hook called.
    Useful during development/debugging.
    """
    
    def __call__(self, *args: TControllerParams.args, **kwargs: TControllerParams.kwargs):
        return HookLoggingStepPatcher()