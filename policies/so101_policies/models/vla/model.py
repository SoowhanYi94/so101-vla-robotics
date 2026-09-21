import torch
from torch import Tensor
from torch import nn

from so101_policies.models.action.action_head import ActionHead
from so101_policies.models.primitives.dropout import Dropout
from so101_policies.models.primitives.embedding import Embedding
from so101_policies.models.primitives.linear import Linear
from so101_policies.models.primitives.positional_embedding import (
    PositionalEmbedding,
)
from so101_policies.models.transformer.encoder import (
    TransformerEncoder,
)
from so101_policies.models.vision.patch_embedding import (
    PatchEmbedding,
)
from so101_policies.models.vla.configuration import (
    VLAConfiguration,
)


class SmallVLA(nn.Module):
    CONTEXT_MODALITY = 0
    IMAGE_MODALITY = 1
    TEXT_MODALITY = 2
    STATE_MODALITY = 3

    def __init__(
        self,
        configuration: VLAConfiguration,
    ) -> None:
        super().__init__()

        self.configuration = configuration

        dimension = configuration.model_dimension

        self.image_embedding = PatchEmbedding(
            image_height=configuration.image_height,
            image_width=configuration.image_width,
            patch_size=configuration.patch_size,
            input_channels=configuration.image_channels,
            model_dimension=dimension,
        )

        self.text_embedding = Embedding(
            vocabulary_size=configuration.vocabulary_size,
            embedding_dimension=dimension,
        )

        self.state_embedding = Linear(
            configuration.state_dimension,
            dimension,
        )

        self.context_token = nn.Parameter(
            torch.empty(1, 1, dimension)
        )

        self.modality_embedding = nn.Parameter(
            torch.empty(4, dimension)
        )

        self.position_embedding = PositionalEmbedding(
            maximum_sequence_length=(
                configuration.maximum_sequence_length
            ),
            embedding_dimension=dimension,
        )

        self.input_dropout = Dropout(
            configuration.dropout_probability
        )

        self.encoder = TransformerEncoder(
            model_dimension=dimension,
            number_of_heads=(
                configuration.number_of_heads
            ),
            feed_forward_dimension=(
                configuration.feed_forward_dimension
            ),
            number_of_layers=(
                configuration.number_of_layers
            ),
            dropout_probability=(
                configuration.dropout_probability
            ),
        )

        self.action_head = ActionHead(
            model_dimension=dimension,
            hidden_dimension=(
                configuration.feed_forward_dimension
            ),
            action_dimension=(
                configuration.action_dimension
            ),
            action_chunk_length=(
                configuration.action_chunk_length
            ),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.context_token.normal_(
                mean=0.0,
                std=0.02,
            )

            self.modality_embedding.normal_(
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        images: Tensor,
        token_ids: Tensor,
        text_attention_mask: Tensor,
        robot_states: Tensor,
    ) -> Tensor:
        self._validate_inputs(
            images,
            token_ids,
            text_attention_mask,
            robot_states,
        )

        batch_size = images.shape[0]

        context_tokens = self.context_token.expand(
            batch_size,
            -1,
            -1,
        )

        context_tokens = (
            context_tokens
            + self.modality_embedding[
                self.CONTEXT_MODALITY
            ]
        )

        image_tokens = self.image_embedding(images)

        image_tokens = (
            image_tokens
            + self.modality_embedding[
                self.IMAGE_MODALITY
            ]
        )

        text_tokens = self.text_embedding(token_ids)

        text_tokens = (
            text_tokens
            + self.modality_embedding[
                self.TEXT_MODALITY
            ]
        )

        state_tokens = self.state_embedding(
            robot_states
        ).unsqueeze(1)

        state_tokens = (
            state_tokens
            + self.modality_embedding[
                self.STATE_MODALITY
            ]
        )

        tokens = torch.cat(
            [
                context_tokens,
                image_tokens,
                text_tokens,
                state_tokens,
            ],
            dim=1,
        )

        context_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=images.device,
        )

        image_mask = torch.ones(
            batch_size,
            image_tokens.shape[1],
            dtype=torch.bool,
            device=images.device,
        )

        state_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=images.device,
        )

        attention_mask = torch.cat(
            [
                context_mask,
                image_mask,
                text_attention_mask.bool(),
                state_mask,
            ],
            dim=1,
        )

        tokens = tokens + self.position_embedding(
            tokens.shape[1]
        )

        tokens = self.input_dropout(tokens)

        encoded_tokens = self.encoder(
            tokens,
            attention_mask,
        )

        context = encoded_tokens[:, 0]

        return self.action_head(context)

    def _validate_inputs(
        self,
        images: Tensor,
        token_ids: Tensor,
        text_attention_mask: Tensor,
        robot_states: Tensor,
    ) -> None:
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape [batch, text_length]."
            )

        if token_ids.shape[1] > (
            self.configuration.maximum_text_length
        ):
            raise ValueError(
                "Text sequence exceeds maximum_text_length."
            )

        if text_attention_mask.shape != token_ids.shape:
            raise ValueError(
                "text_attention_mask must match token_ids."
            )

        if robot_states.ndim != 2:
            raise ValueError(
                "robot_states must have shape "
                "[batch, state_dimension]."
            )

        batch_size = images.shape[0]

        if token_ids.shape[0] != batch_size:
            raise ValueError("Input batch sizes do not match.")

        if robot_states.shape != (
            batch_size,
            self.configuration.state_dimension,
        ):
            raise ValueError(
                "robot_states has an incorrect shape."
            )