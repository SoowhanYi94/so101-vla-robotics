import math
from typing import Optional

import torch
from torch import Tensor
from torch import nn

from so101_policies.models.primitives.dropout import Dropout
from so101_policies.models.primitives.linear import Linear


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        dropout_probability: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()

        if model_dimension <= 0:
            raise ValueError(
                "model_dimension must be positive."
            )

        if number_of_heads <= 0:
            raise ValueError(
                "number_of_heads must be positive."
            )

        if model_dimension % number_of_heads != 0:
            raise ValueError(
                "model_dimension must be divisible "
                "by number_of_heads."
            )

        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = (
            model_dimension // number_of_heads
        )

        self.scale = 1.0 / math.sqrt(
            self.head_dimension
        )

        self.causal = causal

        self.query_key_value = Linear(
            model_dimension,
            3 * model_dimension,
        )

        self.output_projection = Linear(
            model_dimension,
            model_dimension,
        )

        self.attention_dropout = Dropout(
            dropout_probability
        )

        self.output_dropout = Dropout(
            dropout_probability
        )

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape "
                "[batch, sequence, model_dimension]."
            )

        batch_size, sequence_length, dimension = (
            inputs.shape
        )

        if dimension != self.model_dimension:
            raise ValueError(
                f"Expected model dimension "
                f"{self.model_dimension}, "
                f"received {dimension}."
            )

        query, key, value = self.query_key_value(
            inputs
        ).chunk(
            3,
            dim=-1,
        )

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) * self.scale

        if self.causal:
            causal_mask = torch.ones(
                sequence_length,
                sequence_length,
                device=inputs.device,
                dtype=torch.bool,
            ).tril()

            scores = scores.masked_fill(
                ~causal_mask,
                torch.finfo(scores.dtype).min,
            )

        if attention_mask is not None:
            if attention_mask.shape != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "attention_mask must have shape "
                    "[batch, sequence]."
                )

            valid_tokens = attention_mask.to(
                dtype=torch.bool
            )

            key_mask = valid_tokens[:, None, None, :]

            scores = scores.masked_fill(
                ~key_mask,
                torch.finfo(scores.dtype).min,
            )

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = self.attention_dropout(
            attention
        )

        outputs = torch.matmul(
            attention,
            value,
        )

        outputs = outputs.transpose(1, 2).contiguous()

        outputs = outputs.view(
            batch_size,
            sequence_length,
            self.model_dimension,
        )

        outputs = self.output_projection(outputs)
        outputs = self.output_dropout(outputs)

        if attention_mask is not None:
            outputs = outputs * valid_tokens[
                :, :, None
            ].to(outputs.dtype)

        return outputs

    def _split_heads(
        self,
        inputs: Tensor,
    ) -> Tensor:
        batch_size, sequence_length, _ = inputs.shape

        return inputs.view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        ).transpose(1, 2)