from typing import Union, TypeVar
from .controller import DictResidualStepPatcher
from models import ControlNetModel, UNet2DConditionController
import torch

class ControlNetController(UNet2DConditionController):
    def __init__(self, controlnet: ControlNetModel, hint: torch.Tensor):
        self.controlnet = controlnet
        self.hint = hint
    
    def __call__(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
    ) -> DictResidualStepPatcher:
        down_residuals, mid_residual = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states,
            controlnet_hint = self.hint,
            return_dict = False,
        )
        return DictResidualStepPatcher({
            "UNet2DConditionModel.up_blocks.0": down_residuals[3],
            "UNet2DConditionModel.up_blocks.1": down_residuals[2],
            "UNet2DConditionModel.up_blocks.2": down_residuals[1],
            "UNet2DConditionModel.up_blocks.3": down_residuals[0],
            "UNet2DConditionModel.mid_block": mid_residual,
        })
        

class ImageControlNetController(ControlNetController):
    pass