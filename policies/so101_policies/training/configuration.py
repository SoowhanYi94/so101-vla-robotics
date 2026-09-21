from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfiguration:
    dataset_repository: str = "lerobot/svla_so101_pickplace"

    image_key: str = "observation.images.up"
    instruction: str = "pick and place the object"
    dataset_frequency_hz: float = 30.0

    batch_size: int = 16
    number_of_epochs: int = 20
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    maximum_gradient_norm: float = 1.0

    number_of_workers: int = 4
    random_seed: int = 7

    checkpoint_directory: str = (
        "models/checkpoints/small_vla"
    )

    def __post_init__(self) -> None:
        if self.dataset_frequency_hz <= 0.0:
            raise ValueError(
                "dataset_frequency_hz must be positive."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.number_of_epochs <= 0:
            raise ValueError(
                "number_of_epochs must be positive."
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.number_of_workers < 0:
            raise ValueError(
                "number_of_workers cannot be negative."
            )