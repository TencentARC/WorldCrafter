from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput


@dataclass
class WorldCrafterPipelineOutput(BaseOutput):
    frames: torch.Tensor
