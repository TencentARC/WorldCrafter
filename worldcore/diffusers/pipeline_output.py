from dataclasses import dataclass

import torch

from diffusers.utils import BaseOutput


@dataclass
class WorldCorePipelineOutput(BaseOutput):
    frames: torch.Tensor
