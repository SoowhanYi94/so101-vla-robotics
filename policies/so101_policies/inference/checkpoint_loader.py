from dataclasses import dataclass
from pathlib import Path

import torch

from so101_policies.data.normalization import (
    FeatureNormalizer,
)
from so101_policies.models.language.tokenizer import (
    VocabularyTokenizer,
)
from so101_policies.models.vla.configuration import (
    VLAConfiguration,
)
from so101_policies.models.vla.model import SmallVLA


@dataclass
class LoadedPolicy:
    model: SmallVLA
    normalizer: FeatureNormalizer
    tokenizer: VocabularyTokenizer
    configuration: VLAConfiguration
    epoch: int
    metrics: dict


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedPolicy:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required_keys = {
        "epoch",
        "model",
        "model_configuration",
        "tokenizer_vocabulary",
        "statistics",
        "metrics",
    }

    missing_keys = required_keys - checkpoint.keys()

    if missing_keys:
        raise ValueError(
            "Checkpoint is missing keys: "
            + ", ".join(sorted(missing_keys))
        )

    configuration = VLAConfiguration(
        **checkpoint["model_configuration"]
    )

    tokenizer = VocabularyTokenizer(
        checkpoint["tokenizer_vocabulary"]
    )

    model = SmallVLA(
        configuration
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    statistics = checkpoint["statistics"]

    normalizer = FeatureNormalizer(
        state_mean=statistics["state_mean"],
        state_standard_deviation=(
            statistics["state_standard_deviation"]
        ),
        action_mean=statistics["action_mean"],
        action_standard_deviation=(
            statistics["action_standard_deviation"]
        ),
    ).to(device)

    normalizer.eval()

    return LoadedPolicy(
        model=model,
        normalizer=normalizer,
        tokenizer=tokenizer,
        configuration=configuration,
        epoch=int(checkpoint["epoch"]),
        metrics=dict(checkpoint["metrics"]),
    )