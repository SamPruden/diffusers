from typing import Union
from .controller import Controller, DictResidualStepPatcher
from models import UNet2DConditionModel, UNet2DConditionController
import torch

class CannyControlNet(UNet2DConditionController):
    def __init__(self, canny_hint: torch.Tensor):
        self.hint = canny_hint
        # TODO: We're definitely doing everything wrong here
        # I guess we probably want a ControlNet.from_pretrained situation?
        self.control_net = UNet2DConditionModel.from_pretrained("")

    def __call__(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
    ) -> DictResidualStepPatcher:
        pass