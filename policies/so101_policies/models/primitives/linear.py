import math

import torch
from torch import Tensor
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive.")

        if output_dimension <= 0:
            raise ValueError("output_dimension must be positive.")

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.weight = nn.Parameter(
            torch.empty(output_dimension, input_dimension)
        )

        if use_bias:
            self.bias = nn.Parameter(
                torch.empty(output_dimension)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.input_dimension)

        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.input_dimension:
            raise ValueError(
                "Expected the final input dimension to be "
                f"{self.input_dimension}, received {inputs.shape[-1]}."
            )

        outputs = torch.matmul(
            inputs,
            self.weight.transpose(0, 1),
        )

        if self.bias is not None:
            outputs = outputs + self.bias

        return outputs

    def extra_repr(self) -> str:
        return (
            f"input_dimension={self.input_dimension}, "
            f"output_dimension={self.output_dimension}, "
            f"use_bias={self.bias is not None}"
        )