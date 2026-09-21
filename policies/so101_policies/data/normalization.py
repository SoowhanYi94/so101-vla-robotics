import torch
from torch import Tensor
from torch import nn


class FeatureNormalizer(nn.Module):
    def __init__(
        self,
        state_mean: Tensor,
        state_standard_deviation: Tensor,
        action_mean: Tensor,
        action_standard_deviation: Tensor,
        minimum_standard_deviation: float = 1.0e-6,
    ) -> None:
        super().__init__()

        if state_mean.shape != state_standard_deviation.shape:
            raise ValueError(
                "State statistics must have identical shapes."
            )

        if action_mean.shape != action_standard_deviation.shape:
            raise ValueError(
                "Action statistics must have identical shapes."
            )

        self.register_buffer(
            "state_mean",
            state_mean.float(),
        )

        self.register_buffer(
            "state_standard_deviation",
            state_standard_deviation.float().clamp_min(
                minimum_standard_deviation
            ),
        )

        self.register_buffer(
            "action_mean",
            action_mean.float(),
        )

        self.register_buffer(
            "action_standard_deviation",
            action_standard_deviation.float().clamp_min(
                minimum_standard_deviation
            ),
        )

    def normalize_states(
        self,
        states: Tensor,
    ) -> Tensor:
        return (
            states - self.state_mean
        ) / self.state_standard_deviation

    def normalize_actions(
        self,
        actions: Tensor,
    ) -> Tensor:
        return (
            actions - self.action_mean
        ) / self.action_standard_deviation

    def denormalize_actions(
        self,
        normalized_actions: Tensor,
    ) -> Tensor:
        return (
            normalized_actions
            * self.action_standard_deviation
            + self.action_mean
        )

    @staticmethod
    def image_to_float(images: Tensor) -> Tensor:
        if images.dtype == torch.uint8:
            return images.float() / 255.0

        return images.float()