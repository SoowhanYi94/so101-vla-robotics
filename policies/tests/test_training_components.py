import math
import unittest

import torch
from torch.utils.data import Dataset

from so101_policies.data.lerobot_adapter import (
    LeRobotAdapter,
)
from so101_policies.data.normalization import (
    FeatureNormalizer,
)
from so101_policies.models.language.tokenizer import (
    VocabularyTokenizer,
)
from so101_policies.models.vla.configuration import (
    VLAConfiguration,
)
from so101_policies.training.losses import (
    masked_action_mse,
)


class SyntheticLeRobotDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        return {
            "observation.images.up": torch.full(
                (3, 480, 640),
                fill_value=index,
                dtype=torch.uint8,
            ),
            "observation.state": torch.tensor(
                [0.0, 90.0, -90.0, 45.0, 0.0, 10.0]
            ),
            "action": torch.zeros(10, 6),
            "action_is_pad": torch.tensor(
                [False] * 9 + [True]
            ),
        }


class TestTrainingComponents(unittest.TestCase):
    def test_lerobot_adapter(self) -> None:
        configuration = VLAConfiguration()

        tokenizer = VocabularyTokenizer.from_texts(
            ["pick and place the object"]
        )

        dataset = LeRobotAdapter(
            dataset=SyntheticLeRobotDataset(),
            tokenizer=tokenizer,
            configuration=configuration,
            instruction="pick and place the object",
            image_key="observation.images.up",
            values_are_degrees=True,
        )

        sample = dataset[0]

        self.assertEqual(
            tuple(sample["image"].shape),
            (3, 128, 128),
        )

        self.assertEqual(
            tuple(sample["actions"].shape),
            (10, 6),
        )

        self.assertEqual(
            tuple(sample["token_ids"].shape),
            (16,),
        )

        self.assertAlmostEqual(
            sample["robot_state"][1].item(),
            math.pi / 2.0,
            places=5,
        )

        self.assertFalse(
            sample["action_mask"][-1].item()
        )

    def test_normalization_round_trip(self) -> None:
        normalizer = FeatureNormalizer(
            state_mean=torch.zeros(6),
            state_standard_deviation=torch.ones(6),
            action_mean=torch.arange(6).float(),
            action_standard_deviation=torch.full(
                (6,),
                2.0,
            ),
        )

        actions = torch.randn(2, 10, 6)

        reconstructed = (
            normalizer.denormalize_actions(
                normalizer.normalize_actions(
                    actions
                )
            )
        )

        self.assertTrue(
            torch.allclose(
                actions,
                reconstructed,
                atol=1.0e-6,
            )
        )

    def test_masked_action_loss(self) -> None:
        predicted = torch.zeros(1, 2, 1)
        target = torch.tensor([[[1.0], [10.0]]])
        mask = torch.tensor([[True, False]])

        loss = masked_action_mse(
            predicted,
            target,
            mask,
        )

        self.assertAlmostEqual(
            loss.item(),
            1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()