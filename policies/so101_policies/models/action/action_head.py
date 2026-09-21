from torch import Tensor
from torch import nn

from so101_policies.models.primitives.activation import Gelu
from so101_policies.models.primitives.linear import Linear


class ActionHead(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        action_chunk_length: int,
    ) -> None:
        super().__init__()

        if action_dimension <= 0:
            raise ValueError(
                "action_dimension must be positive."
            )

        if action_chunk_length <= 0:
            raise ValueError(
                "action_chunk_length must be positive."
            )

        self.action_dimension = action_dimension
        self.action_chunk_length = (
            action_chunk_length
        )

        self.input_projection = Linear(
            model_dimension,
            hidden_dimension,
        )

        self.activation = Gelu()

        self.output_projection = Linear(
            hidden_dimension,
            action_dimension * action_chunk_length,
        )

    def forward(self, context: Tensor) -> Tensor:
        if context.ndim != 2:
            raise ValueError(
                "context must have shape "
                "[batch, model_dimension]."
            )

        outputs = self.input_projection(context)
        outputs = self.activation(outputs)
        outputs = self.output_projection(outputs)

        return outputs.view(
            context.shape[0],
            self.action_chunk_length,
            self.action_dimension,
        )