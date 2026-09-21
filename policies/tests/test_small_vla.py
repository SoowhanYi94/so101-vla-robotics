import unittest

import torch

from so101_policies.models.vla.configuration import (
    VLAConfiguration,
)
from so101_policies.models.vla.model import SmallVLA


class TestSmallVLA(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

        self.configuration = VLAConfiguration(
            dropout_probability=0.0
        )

        self.model = SmallVLA(
            self.configuration
        )

    def test_forward_shape(self) -> None:
        batch_size = 2

        images = torch.randn(
            batch_size,
            self.configuration.image_channels,
            self.configuration.image_height,
            self.configuration.image_width,
        )

        token_ids = torch.randint(
            low=0,
            high=self.configuration.vocabulary_size,
            size=(
                batch_size,
                self.configuration.maximum_text_length,
            ),
        )

        text_mask = torch.ones_like(
            token_ids,
            dtype=torch.bool,
        )

        robot_states = torch.randn(
            batch_size,
            self.configuration.state_dimension,
        )

        actions = self.model(
            images=images,
            token_ids=token_ids,
            text_attention_mask=text_mask,
            robot_states=robot_states,
        )

        expected_shape = (
            batch_size,
            self.configuration.action_chunk_length,
            self.configuration.action_dimension,
        )

        self.assertEqual(
            tuple(actions.shape),
            expected_shape,
        )

        self.assertTrue(
            torch.isfinite(actions).all()
        )

    def test_backward_pass(self) -> None:
        batch_size = 2

        images = torch.randn(
            batch_size,
            self.configuration.image_channels,
            self.configuration.image_height,
            self.configuration.image_width,
        )

        token_ids = torch.randint(
            low=0,
            high=self.configuration.vocabulary_size,
            size=(
                batch_size,
                self.configuration.maximum_text_length,
            ),
        )

        text_mask = torch.ones_like(
            token_ids,
            dtype=torch.bool,
        )

        robot_states = torch.randn(
            batch_size,
            self.configuration.state_dimension,
        )

        target_actions = torch.randn(
            batch_size,
            self.configuration.action_chunk_length,
            self.configuration.action_dimension,
        )

        predicted_actions = self.model(
            images=images,
            token_ids=token_ids,
            text_attention_mask=text_mask,
            robot_states=robot_states,
        )

        loss = (
            predicted_actions - target_actions
        ).square().mean()

        loss.backward()

        missing_gradients = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is None
        ]

        self.assertEqual(
            missing_gradients,
            [],
        )

    def test_parameter_count(self) -> None:
        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        self.assertGreater(
            parameter_count,
            0,
        )

        self.assertLess(
            parameter_count,
            5_000_000,
        )


if __name__ == "__main__":
    unittest.main()