import math

import torch
from torch import Tensor
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
    ) -> None:
        super().__init__()

        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive.")

        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be positive."
            )

        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension

        self.weight = nn.Parameter(
            torch.empty(
                vocabulary_size,
                embedding_dimension,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        standard_deviation = (
            1.0 / math.sqrt(self.embedding_dimension)
        )

        with torch.no_grad():
            self.weight.normal_(
                mean=0.0,
                std=standard_deviation,
            )

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "token_ids must contain integer indices."
            )

        return self.weight[token_ids.long()]

    def extra_repr(self) -> str:
        return (
            f"vocabulary_size={self.vocabulary_size}, "
            f"embedding_dimension={self.embedding_dimension}"
        )