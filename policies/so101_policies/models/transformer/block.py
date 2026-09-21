from typing import Optional

from torch import Tensor
from torch import nn

from so101_policies.models.primitives.layer_norm import LayerNorm
from so101_policies.models.transformer.attention import (
    MultiHeadSelfAttention,
)
from so101_policies.models.transformer.feed_forward import (
    FeedForward,
)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        feed_forward_dimension: int,
        dropout_probability: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()

        self.attention_norm = LayerNorm(
            model_dimension
        )

        self.attention = MultiHeadSelfAttention(
            model_dimension=model_dimension,
            number_of_heads=number_of_heads,
            dropout_probability=dropout_probability,
            causal=causal,
        )

        self.feed_forward_norm = LayerNorm(
            model_dimension
        )

        self.feed_forward = FeedForward(
            model_dimension=model_dimension,
            hidden_dimension=feed_forward_dimension,
            dropout_probability=dropout_probability,
        )

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        outputs = inputs + self.attention(
            self.attention_norm(inputs),
            attention_mask,
        )

        outputs = outputs + self.feed_forward(
            self.feed_forward_norm(outputs)
        )

        if attention_mask is not None:
            outputs = outputs * attention_mask[
                :, :, None
            ].to(outputs.dtype)

        return outputs