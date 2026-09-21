import torch
from torch import Tensor
from torch import nn


class Dropout(nn.Module):
    def __init__(
        self,
        probability: float = 0.0,
    ) -> None:
        super().__init__()

        if probability < 0.0 or probability >= 1.0:
            raise ValueError(
                "probability must be in the range [0, 1)."
            )

        self.probability = probability

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return inputs

        keep_probability = 1.0 - self.probability

        mask = (
            torch.rand_like(inputs)
            < keep_probability
        ).to(inputs.dtype)

        return inputs * mask / keep_probability

    def extra_repr(self) -> str:
        return f"probability={self.probability}"