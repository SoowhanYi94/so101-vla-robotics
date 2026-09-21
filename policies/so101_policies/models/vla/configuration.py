from dataclasses import dataclass


@dataclass(frozen=True)
class VLAConfiguration:
    image_height: int = 128
    image_width: int = 128
    image_channels: int = 3
    patch_size: int = 16

    vocabulary_size: int = 1024
    maximum_text_length: int = 16

    state_dimension: int = 6
    action_dimension: int = 6
    action_chunk_length: int = 10

    model_dimension: int = 128
    number_of_heads: int = 4
    feed_forward_dimension: int = 512
    number_of_layers: int = 4

    dropout_probability: float = 0.1

    def __post_init__(self) -> None:
        if self.image_height % self.patch_size != 0:
            raise ValueError(
                "image_height must be divisible by patch_size."
            )

        if self.image_width % self.patch_size != 0:
            raise ValueError(
                "image_width must be divisible by patch_size."
            )

        if self.model_dimension % self.number_of_heads != 0:
            raise ValueError(
                "model_dimension must be divisible "
                "by number_of_heads."
            )

    @property
    def number_of_image_tokens(self) -> int:
        return (
            self.image_height // self.patch_size
        ) * (
            self.image_width // self.patch_size
        )

    @property
    def maximum_sequence_length(self) -> int:
        return (
            1
            + self.number_of_image_tokens
            + self.maximum_text_length
            + 1
        )