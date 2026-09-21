import torch
from torch import Tensor


def masked_action_mse(
    predicted_actions: Tensor,
    target_actions: Tensor,
    action_mask: Tensor | None = None,
) -> Tensor:
    if predicted_actions.shape != target_actions.shape:
        raise ValueError(
            "predicted_actions and target_actions "
            "must have identical shapes."
        )

    if predicted_actions.ndim != 3:
        raise ValueError(
            "Actions must have shape "
            "[batch, chunk, action_dimension]."
        )

    squared_error = (
        predicted_actions - target_actions
    ).square()

    if action_mask is None:
        return squared_error.mean()

    expected_mask_shape = predicted_actions.shape[:2]

    if action_mask.shape != expected_mask_shape:
        raise ValueError(
            "action_mask must have shape [batch, chunk]."
        )

    expanded_mask = action_mask[
        :, :, None
    ].expand_as(
        squared_error
    ).to(
        squared_error.dtype
    )

    valid_values = expanded_mask.sum()

    if valid_values.item() == 0:
        raise ValueError(
            "action_mask contains no valid actions."
        )

    return (
        squared_error * expanded_mask
    ).sum() / valid_values