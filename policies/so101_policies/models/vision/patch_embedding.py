from torch import Tensor
from torch import nn

from so101_policies.models.primitives.linear import Linear


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_size: int,
        input_channels: int,
        model_dimension: int,
    ) -> None:
        super().__init__()

        if image_height % patch_size != 0:
            raise ValueError(
                "image_height must be divisible by patch_size."
            )

        if image_width % patch_size != 0:
            raise ValueError(
                "image_width must be divisible by patch_size."
            )

        if input_channels <= 0:
            raise ValueError(
                "input_channels must be positive."
            )

        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.input_channels = input_channels

        self.grid_height = image_height // patch_size
        self.grid_width = image_width // patch_size

        self.number_of_patches = (
            self.grid_height * self.grid_width
        )

        patch_dimension = (
            input_channels
            * patch_size
            * patch_size
        )

        self.projection = Linear(
            patch_dimension,
            model_dimension,
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "images must have shape "
                "[batch, channels, height, width]."
            )

        batch_size, channels, height, width = (
            images.shape
        )

        expected_shape = (
            self.input_channels,
            self.image_height,
            self.image_width,
        )

        if (channels, height, width) != expected_shape:
            raise ValueError(
                "Expected image shape "
                f"{expected_shape}, received "
                f"{(channels, height, width)}."
            )

        patches = images.unfold(
            2,
            self.patch_size,
            self.patch_size,
        ).unfold(
            3,
            self.patch_size,
            self.patch_size,
        )

        patches = patches.permute(
            0,
            2,
            3,
            1,
            4,
            5,
        ).contiguous()

        patches = patches.view(
            batch_size,
            self.number_of_patches,
            -1,
        )

        return self.projection(patches)