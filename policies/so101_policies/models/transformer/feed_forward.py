from torch import Tensor
from torch import nn

from so101_policies.models.primitives.activation import Gelu
from so101_policies.models.primitives.dropout import Dropout
from so101_policies.models.primitives.linear import Linear


class FeedForward(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        hidden_dimension: int,
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()

        if model_dimension <= 0:
            raise ValueError(
                "model_dimension must be positive."
            )

        if hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be positive."
            )

        self.input_projection = Linear(
            model_dimension,
            hidden_dimension,
        )

        self.activation = Gelu()

        self.hidden_dropout = Dropout(
            dropout_probability
        )

        self.output_projection = Linear(
            hidden_dimension,
            model_dimension,
        )

        self.output_dropout = Dropout(
            dropout_probability
        )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.input_projection(inputs)
        outputs = self.activation(outputs)
        outputs = self.hidden_dropout(outputs)
        outputs = self.output_projection(outputs)
        outputs = self.output_dropout(outputs)

        return outputs