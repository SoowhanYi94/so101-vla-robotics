import torch
from torch import Tensor
from torch import nn


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_dimension: int,
        epsilon: float = 1.0e-5,
        use_affine: bool = True,
    ) -> None:
        super().__init__()

        if normalized_dimension <= 0:
            raise ValueError(
                "normalized_dimension must be positive."
            )

        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        self.normalized_dimension = normalized_dimension
        self.epsilon = epsilon

        if use_affine:
            self.weight = nn.Parameter(
                torch.ones(normalized_dimension)
            )

            self.bias = nn.Parameter(
                torch.zeros(normalized_dimension)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.normalized_dimension:
            raise ValueError(
                "Expected the final input dimension to be "
                f"{self.normalized_dimension}, "
                f"received {inputs.shape[-1]}."
            )

        computation = inputs.float()

        mean = computation.mean(
            dim=-1,
            keepdim=True,
        )

        variance = (
            computation - mean
        ).square().mean(
            dim=-1,
            keepdim=True,
        )

        outputs = (
            computation - mean
        ) * torch.rsqrt(
            variance + self.epsilon
        )

        outputs = outputs.to(inputs.dtype)

        if self.weight is not None:
            outputs = outputs * self.weight.to(outputs.dtype)
            outputs = outputs + self.bias.to(outputs.dtype)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"normalized_dimension={self.normalized_dimension}, "
            f"epsilon={self.epsilon}, "
            f"use_affine={self.weight is not None}"
        )