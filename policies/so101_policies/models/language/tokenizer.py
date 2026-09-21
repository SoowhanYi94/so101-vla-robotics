import re
from collections import Counter
from typing import Iterable

import torch
from torch import Tensor


class VocabularyTokenizer:
    PAD_TOKEN = "[PAD]"
    UNKNOWN_TOKEN = "[UNK]"
    BEGIN_TOKEN = "[BOS]"
    END_TOKEN = "[EOS]"

    def __init__(
        self,
        vocabulary: list[str],
    ) -> None:
        special_tokens = [
            self.PAD_TOKEN,
            self.UNKNOWN_TOKEN,
            self.BEGIN_TOKEN,
            self.END_TOKEN,
        ]

        ordered_tokens = []

        for token in special_tokens + vocabulary:
            if token not in ordered_tokens:
                ordered_tokens.append(token)

        self.token_to_id = {
            token: index
            for index, token in enumerate(
                ordered_tokens
            )
        }

        self.id_to_token = ordered_tokens

    @property
    def vocabulary_size(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        maximum_vocabulary_size: int = 1024,
    ) -> "VocabularyTokenizer":
        if maximum_vocabulary_size < 4:
            raise ValueError(
                "maximum_vocabulary_size must be at least 4."
            )

        token_counts = Counter()

        for text in texts:
            token_counts.update(
                cls.tokenize(text)
            )

        available_size = (
            maximum_vocabulary_size - 4
        )

        vocabulary = [
            token
            for token, _ in token_counts.most_common(
                available_size
            )
        ]

        return cls(vocabulary)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(
            r"[a-z0-9]+|[^\w\s]",
            text.lower(),
        )

    def encode(
        self,
        text: str,
        maximum_length: int,
    ) -> tuple[Tensor, Tensor]:
        if maximum_length < 2:
            raise ValueError(
                "maximum_length must be at least 2."
            )

        tokens = [
            self.BEGIN_TOKEN,
            *self.tokenize(text),
            self.END_TOKEN,
        ]

        tokens = tokens[:maximum_length]

        if tokens[-1] != self.END_TOKEN:
            tokens[-1] = self.END_TOKEN

        token_ids = [
            self.token_to_id.get(
                token,
                self.token_to_id[
                    self.UNKNOWN_TOKEN
                ],
            )
            for token in tokens
        ]

        attention_mask = [True] * len(token_ids)

        padding_length = (
            maximum_length - len(token_ids)
        )

        token_ids.extend(
            [self.token_to_id[self.PAD_TOKEN]]
            * padding_length
        )

        attention_mask.extend(
            [False] * padding_length
        )

        return (
            torch.tensor(
                token_ids,
                dtype=torch.long,
            ),
            torch.tensor(
                attention_mask,
                dtype=torch.bool,
            ),
        )