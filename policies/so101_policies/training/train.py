import argparse
import random
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from so101_policies.data.dataset_factory import (
    create_dataset,
)
from so101_policies.data.normalization import (
    FeatureNormalizer,
)
from so101_policies.data.statistics import (
    compute_dataset_statistics,
)
from so101_policies.models.vla.configuration import (
    VLAConfiguration,
)
from so101_policies.models.vla.model import SmallVLA
from so101_policies.training.checkpoint import (
    save_checkpoint,
)
from so101_policies.training.configuration import (
    TrainingConfiguration,
)
from so101_policies.training.losses import (
    masked_action_mse,
)


def create_data_loaders(
    dataset: Dataset,
    configuration: TrainingConfiguration,
) -> tuple[DataLoader, DataLoader, Dataset]:
    validation_size = max(
        1,
        int(0.1 * len(dataset)),
    )

    training_size = len(dataset) - validation_size

    generator = torch.Generator().manual_seed(
        configuration.random_seed
    )

    training_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )

    common_arguments = {
        "batch_size": configuration.batch_size,
        "num_workers": configuration.number_of_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        **common_arguments,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_arguments,
    )

    return (
        training_loader,
        validation_loader,
        training_dataset,
    )


def run_epoch(
    model: SmallVLA,
    normalizer: FeatureNormalizer,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    maximum_gradient_norm: float,
) -> float:
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    number_of_batches = 0

    for batch in data_loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        token_ids = batch["token_ids"].to(
            device,
            non_blocking=True,
        )

        text_mask = batch[
            "text_attention_mask"
        ].to(
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

        robot_states = normalizer.normalize_states(
            robot_states
        )

        target_actions = normalizer.normalize_actions(
            target_actions
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):
            predicted_actions = model(
                images=images,
                token_ids=token_ids,
                text_attention_mask=text_mask,
                robot_states=robot_states,
            )

            loss = masked_action_mse(
                predicted_actions,
                target_actions,
                action_mask,
            )

        if training:
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                maximum_gradient_norm,
            )

            optimizer.step()

        total_loss += loss.detach().item()
        number_of_batches += 1

    if number_of_batches == 0:
        raise RuntimeError(
            "Data loader produced no batches."
        )

    return total_loss / number_of_batches


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "datasets/raw/svla_so101_pickplace"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    training_configuration = TrainingConfiguration()

    if arguments.epochs is not None:
        training_configuration = replace(
            training_configuration,
            number_of_epochs=arguments.epochs,
        )

    random.seed(
        training_configuration.random_seed
    )

    torch.manual_seed(
        training_configuration.random_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            training_configuration.random_seed
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    initial_model_configuration = (
        VLAConfiguration()
    )

    dataset, tokenizer = create_dataset(
        model_configuration=(
            initial_model_configuration
        ),
        training_configuration=(
            training_configuration
        ),
        dataset_root=arguments.dataset_root,
    )

    model_configuration = replace(
        initial_model_configuration,
        vocabulary_size=(
            tokenizer.vocabulary_size
        ),
    )

    (
        training_loader,
        validation_loader,
        training_dataset,
    ) = create_data_loaders(
        dataset,
        training_configuration,
    )

    statistics = compute_dataset_statistics(
        dataset=training_dataset,
        batch_size=training_configuration.batch_size,
        number_of_workers=(
            training_configuration.number_of_workers
        ),
    )

    normalizer = FeatureNormalizer(
        state_mean=statistics.state_mean,
        state_standard_deviation=(
            statistics.state_standard_deviation
        ),
        action_mean=statistics.action_mean,
        action_standard_deviation=(
            statistics.action_standard_deviation
        ),
    ).to(device)

    model = SmallVLA(
        model_configuration
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_configuration.learning_rate,
        weight_decay=(
            training_configuration.weight_decay
        ),
    )

    checkpoint_directory = Path(
        training_configuration.checkpoint_directory
    )

    best_validation_loss = float("inf")

    print(f"device: {device}")
    print(f"training samples: {len(training_dataset)}")
    print(
        f"validation samples: "
        f"{len(validation_loader.dataset)}"
    )

    for epoch in range(
        1,
        training_configuration.number_of_epochs + 1,
    ):
        training_loss = run_epoch(
            model=model,
            normalizer=normalizer,
            data_loader=training_loader,
            device=device,
            optimizer=optimizer,
            maximum_gradient_norm=(
                training_configuration.maximum_gradient_norm
            ),
        )

        with torch.inference_mode():
            validation_loss = run_epoch(
                model=model,
                normalizer=normalizer,
                data_loader=validation_loader,
                device=device,
                optimizer=None,
                maximum_gradient_norm=(
                    training_configuration.maximum_gradient_norm
                ),
            )

        metrics = {
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        }

        print(
            f"epoch {epoch:03d} | "
            f"train {training_loss:.6f} | "
            f"validation {validation_loss:.6f}"
        )

        save_checkpoint(
            path=checkpoint_directory / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            model_configuration=model_configuration,
            tokenizer=tokenizer,
            statistics=statistics,
            metrics=metrics,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                path=checkpoint_directory / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                model_configuration=model_configuration,
                tokenizer=tokenizer,
                statistics=statistics,
                metrics=metrics,
            )


if __name__ == "__main__":
    main()