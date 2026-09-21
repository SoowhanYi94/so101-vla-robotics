from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from so101_policies.data.statistics import (
    DatasetStatistics,
)
from so101_policies.models.language.tokenizer import (
    VocabularyTokenizer,
)
from so101_policies.models.vla.configuration import (
    VLAConfiguration,
)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    model_configuration: VLAConfiguration,
    tokenizer: VocabularyTokenizer,
    statistics: DatasetStatistics,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_configuration": asdict(
            model_configuration
        ),
        "tokenizer_vocabulary": (
            tokenizer.id_to_token
        ),
        "statistics": {
            "state_mean": statistics.state_mean,
            "state_standard_deviation": (
                statistics.state_standard_deviation
            ),
            "action_mean": statistics.action_mean,
            "action_standard_deviation": (
                statistics.action_standard_deviation
            ),
        },
        "metrics": metrics,
    }

    torch.save(
        checkpoint,
        path,
    )