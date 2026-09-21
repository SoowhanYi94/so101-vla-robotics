import torch
from torch import Tensor
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        maximum_sequence_length: int,
        embedding_dimension: int,
    ) -> None:
        super().__init__()

        if maximum_sequence_length <= 0:
            raise ValueError(
                "maximum_sequence_length must be positive."
            )

        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be positive."
            )

        self.maximum_sequence_length = (
            maximum_sequence_length
        )

        self.embedding_dimension = embedding_dimension

        self.weight = nn.Parameter(
            torch.empty(
                maximum_sequence_length,
                embedding_dimension,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.normal_(
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        sequence_length: int,
        offset: int = 0,
    ) -> Tensor:
        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive."
            )

        if offset < 0:
            raise ValueError(
                "offset cannot be negative."
            )

        end = offset + sequence_length

        if end > self.maximum_sequence_length:
            raise ValueError(
                "Requested positions exceed "
                "maximum_sequence_length."
            )

        return self.weight[offset:end].unsqueeze(0)

    def extra_repr(self) -> str:
        return (
            f"maximum_sequence_length="
            f"{self.maximum_sequence_length}, "
            f"embedding_dimension="
            f"{self.embedding_dimension}"
        )