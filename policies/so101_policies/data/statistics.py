from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetStatistics:
    state_mean: Tensor
    state_standard_deviation: Tensor
    action_mean: Tensor
    action_standard_deviation: Tensor


def compute_dataset_statistics(
    dataset: Dataset,
    batch_size: int,
    number_of_workers: int,
) -> DatasetStatistics:
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=False,
    )

    state_sum = None
    state_squared_sum = None
    action_sum = None
    action_squared_sum = None

    state_count = 0
    action_count = 0

    for batch in data_loader:
        states = batch["robot_state"].double()
        actions = batch["actions"].double()

        action_mask = batch["action_mask"][
            :, :, None
        ].double()

        current_state_sum = states.sum(dim=0)
        current_state_squared_sum = (
            states.square().sum(dim=0)
        )

        current_action_sum = (
            actions * action_mask
        ).sum(
            dim=(0, 1)
        )

        current_action_squared_sum = (
            actions.square() * action_mask
        ).sum(
            dim=(0, 1)
        )

        if state_sum is None:
            state_sum = current_state_sum
            state_squared_sum = (
                current_state_squared_sum
            )
            action_sum = current_action_sum
            action_squared_sum = (
                current_action_squared_sum
            )
        else:
            state_sum += current_state_sum
            state_squared_sum += (
                current_state_squared_sum
            )
            action_sum += current_action_sum
            action_squared_sum += (
                current_action_squared_sum
            )

        state_count += states.shape[0]
        action_count += int(action_mask.sum().item())

    if state_count == 0 or action_count == 0:
        raise ValueError(
            "Cannot compute statistics from an empty dataset."
        )

    state_mean = state_sum / state_count
    action_mean = action_sum / action_count

    state_variance = (
        state_squared_sum / state_count
        - state_mean.square()
    ).clamp_min(1.0e-12)

    action_variance = (
        action_squared_sum / action_count
        - action_mean.square()
    ).clamp_min(1.0e-12)

    return DatasetStatistics(
        state_mean=state_mean.float(),
        state_standard_deviation=(
            state_variance.sqrt().float()
        ),
        action_mean=action_mean.float(),
        action_standard_deviation=(
            action_variance.sqrt().float()
        ),
    )