from typing import Optional

from torch import Tensor
from torch import nn

from so101_policies.models.primitives.layer_norm import LayerNorm
from so101_policies.models.transformer.block import (
    TransformerBlock,
)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        feed_forward_dimension: int,
        number_of_layers: int,
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()

        if number_of_layers <= 0:
            raise ValueError(
                "number_of_layers must be positive."
            )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dimension=model_dimension,
                    number_of_heads=number_of_heads,
                    feed_forward_dimension=(
                        feed_forward_dimension
                    ),
                    dropout_probability=(
                        dropout_probability
                    ),
                    causal=False,
                )
                for _ in range(number_of_layers)
            ]
        )

        self.output_norm = LayerNorm(
            model_dimension
        )

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        outputs = inputs

        for block in self.blocks:
            outputs = block(
                outputs,
                attention_mask,
            )

        outputs = self.output_norm(outputs)

        if attention_mask is not None:
            outputs = outputs * attention_mask[
                :, :, None
            ].to(outputs.dtype)

        return outputs