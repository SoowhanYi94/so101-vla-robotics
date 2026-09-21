import math

import torch
from torch import Tensor
from torch import nn


class Gelu(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return 0.5 * inputs * (
            1.0
            + torch.erf(
                inputs / math.sqrt(2.0)
            )
        )