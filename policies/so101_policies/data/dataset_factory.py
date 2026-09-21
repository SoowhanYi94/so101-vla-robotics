from pathlib import Path

from torch.utils.data import Dataset

from so101_policies.data.lerobot_adapter import LeRobotAdapter
from so101_policies.models.language.tokenizer import VocabularyTokenizer
from so101_policies.models.vla.configuration import VLAConfiguration
from so101_policies.training.configuration import TrainingConfiguration


def create_dataset(
    model_configuration: VLAConfiguration,
    training_configuration: TrainingConfiguration,
    dataset_root: Path | None = None,
) -> tuple[Dataset, VocabularyTokenizer]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "LeRobot is required to load the training dataset."
        ) from error

    action_timestamps = [
        step / training_configuration.dataset_frequency_hz
        for step in range(model_configuration.action_chunk_length)
    ]

    dataset_arguments = {
        "repo_id": training_configuration.dataset_repository,
        "delta_timestamps": {
            "action": action_timestamps,
        },
        "video_backend": "pyav",
    }

    if dataset_root is not None:
        dataset_arguments["root"] = dataset_root

    base_dataset = LeRobotDataset(
        **dataset_arguments
    )

    tokenizer = VocabularyTokenizer.from_texts(
        [training_configuration.instruction],
        maximum_vocabulary_size=model_configuration.vocabulary_size,
    )

    adapted_dataset = LeRobotAdapter(
        dataset=base_dataset,
        tokenizer=tokenizer,
        configuration=model_configuration,
        instruction=training_configuration.instruction,
        image_key=training_configuration.image_key,
        values_are_degrees=True,
    )

    return adapted_dataset, tokenizer