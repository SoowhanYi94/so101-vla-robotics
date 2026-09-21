import argparse
import math
from pathlib import Path

import torch

from so101_policies.data.dataset_factory import create_dataset
from so101_policies.inference.checkpoint_loader import load_checkpoint
from so101_policies.training.configuration import TrainingConfiguration
from so101_policies.training.train import create_data_loaders


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a VLA checkpoint on its validation split."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    return torch.device(name)


def main() -> None:
    arguments = parse_arguments()
    device = select_device(arguments.device)

    policy = load_checkpoint(
        arguments.checkpoint,
        device,
    )

    training_configuration = TrainingConfiguration()

    dataset, _ = create_dataset(
        model_configuration=policy.configuration,
        training_configuration=training_configuration,
        dataset_root=arguments.dataset_root,
    )

    dataset.tokenizer = policy.tokenizer

    _, validation_loader, _ = create_data_loaders(
        dataset,
        training_configuration,
    )

    action_dimension = policy.configuration.action_dimension

    absolute_error_sum = torch.zeros(
        action_dimension,
        device=device,
    )

    squared_error_sum = torch.zeros(
        action_dimension,
        device=device,
    )

    valid_count = torch.zeros(
        action_dimension,
        device=device,
    )

    maximum_error = 0.0
    sample_count = 0

    with torch.inference_mode():
        for batch in validation_loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            token_ids = batch["token_ids"].to(
                device,
                non_blocking=True,
            )

            text_mask = batch["text_attention_mask"].to(
                device,
                non_blocking=True,
            )

            robot_states = batch["robot_state"].to(
                device,
                non_blocking=True,
            )

            target_actions = batch["actions"].to(
                device,
                non_blocking=True,
            )

            action_mask = batch["action_mask"].to(
                device,
                non_blocking=True,
            )

            normalized_states = (
                policy.normalizer.normalize_states(
                    robot_states
                )
            )

            normalized_predictions = policy.model(
                images=images,
                token_ids=token_ids,
                text_attention_mask=text_mask,
                robot_states=normalized_states,
            )

            predicted_actions = (
                policy.normalizer.denormalize_actions(
                    normalized_predictions
                )
            )

            valid_mask = action_mask.unsqueeze(-1).expand_as(
                predicted_actions
            )

            error = predicted_actions - target_actions
            absolute_error = error.abs()
            squared_error = error.square()

            absolute_error_sum += (
                absolute_error * valid_mask
            ).sum(dim=(0, 1))

            squared_error_sum += (
                squared_error * valid_mask
            ).sum(dim=(0, 1))

            valid_count += valid_mask.sum(dim=(0, 1))

            if valid_mask.any():
                batch_maximum = absolute_error[
                    valid_mask
                ].max().item()

                maximum_error = max(
                    maximum_error,
                    batch_maximum,
                )

            sample_count += images.shape[0]

    per_joint_mae = absolute_error_sum / valid_count
    per_joint_mse = squared_error_sum / valid_count

    overall_mae = (
        absolute_error_sum.sum()
        / valid_count.sum()
    ).item()

    overall_mse = (
        squared_error_sum.sum()
        / valid_count.sum()
    ).item()

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {policy.epoch}")
    print(f"Validation samples: {sample_count}")
    print(f"Validation MAE: {overall_mae:.6f} rad")
    print(
        "Validation MAE: "
        f"{math.degrees(overall_mae):.3f} deg"
    )
    print(f"Validation MSE: {overall_mse:.6f}")
    print(f"Maximum error: {maximum_error:.6f} rad")
    print(
        "Maximum error: "
        f"{math.degrees(maximum_error):.3f} deg"
    )

    print("\nPer-joint error:")

    for index in range(action_dimension):
        name = (
            JOINT_NAMES[index]
            if index < len(JOINT_NAMES)
            else f"joint_{index}"
        )

        mae = per_joint_mae[index].item()
        mse = per_joint_mse[index].item()

        print(
            f"  {name:14s} "
            f"MAE {mae:.6f} rad "
            f"({math.degrees(mae):.3f} deg), "
            f"MSE {mse:.6f}"
        )


if __name__ == "__main__":
    main()