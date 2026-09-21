import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.data import Dataset

from so101_policies.models.language.tokenizer import (
    VocabularyTokenizer,
)
from so101_policies.models.vla.configuration import (
    VLAConfiguration,
)


class LeRobotAdapter(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: VocabularyTokenizer,
        configuration: VLAConfiguration,
        instruction: str,
        image_key: str,
        values_are_degrees: bool = True,
        joint_signs: Tensor | None = None,
        joint_offsets: Tensor | None = None,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.instruction = instruction
        self.image_key = image_key

        joint_dimension = configuration.state_dimension

        if joint_signs is None:
            joint_signs = torch.ones(joint_dimension)

        if joint_offsets is None:
            joint_offsets = torch.zeros(joint_dimension)

        if joint_signs.shape != (joint_dimension,):
            raise ValueError(
                "joint_signs has an incorrect shape."
            )

        if joint_offsets.shape != (joint_dimension,):
            raise ValueError(
                "joint_offsets has an incorrect shape."
            )

        self.joint_signs = joint_signs.float()
        self.joint_offsets = joint_offsets.float()

        self.unit_scale = (
            math.pi / 180.0
            if values_are_degrees
            else 1.0
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> Mapping[str, Any]:
        sample = self.dataset[index]

        image = self._prepare_image(
            sample[self.image_key]
        )

        state = self._convert_joints(
            sample["observation.state"]
        )

        actions = self._convert_joints(
            sample["action"]
        )

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        expected_action_shape = (
            self.configuration.action_chunk_length,
            self.configuration.action_dimension,
        )

        if actions.shape != expected_action_shape:
            raise ValueError(
                f"Expected action shape "
                f"{expected_action_shape}, "
                f"received {tuple(actions.shape)}. "
                "Configure LeRobot delta_timestamps "
                "to return an action chunk."
            )

        action_mask = self._read_action_mask(
            sample,
            actions.shape[0],
        )

        token_ids, text_mask = (
            self.tokenizer.encode(
                self.instruction,
                self.configuration.maximum_text_length,
            )
        )

        return {
            "image": image,
            "token_ids": token_ids,
            "text_attention_mask": text_mask,
            "robot_state": state,
            "actions": actions,
            "action_mask": action_mask,
        }

    def _prepare_image(
        self,
        image: Tensor,
    ) -> Tensor:
        image = torch.as_tensor(image)

        if image.ndim != 3:
            raise ValueError(
                "Image must have shape [channels, height, width]."
            )

        if image.shape[0] not in (1, 3, 4):
            image = image.permute(2, 0, 1)

        image = image[:3].float()

        if image.max() > 1.0:
            image = image / 255.0

        image = functional.interpolate(
            image.unsqueeze(0),
            size=(
                self.configuration.image_height,
                self.configuration.image_width,
            ),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return image.clamp(0.0, 1.0)

    def _convert_joints(
        self,
        values: Tensor,
    ) -> Tensor:
        values = torch.as_tensor(
            values,
            dtype=torch.float32,
        )

        if values.shape[-1] != (
            self.configuration.state_dimension
        ):
            raise ValueError(
                "Joint vector has an incorrect dimension."
            )

        return (
            values * self.unit_scale
        ) * self.joint_signs + self.joint_offsets

    @staticmethod
    def _read_action_mask(
        sample: Mapping[str, Any],
        chunk_length: int,
    ) -> Tensor:
        if "action_is_pad" not in sample:
            return torch.ones(
                chunk_length,
                dtype=torch.bool,
            )

        padding = torch.as_tensor(
            sample["action_is_pad"],
            dtype=torch.bool,
        )

        return ~padding