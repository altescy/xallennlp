import math
from typing import Optional, cast

import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn import util


@Seq2SeqEncoder.register("self_attention")
class SelfAttentionEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        bidirectional: bool = True,
        positional_encoding: Optional[str] = None,
        positional_embedding_size: int = 512,
    ) -> None:
        super().__init__(stateful=False)
        self._input_dim = input_dim
        self._bidirectional = bidirectional

        if positional_encoding is None:
            self._sinusoidal_positional_encoding = False
            self._positional_embedding = None
        elif positional_encoding == "sinusoidal":
            self._sinusoidal_positional_encoding = True
            self._positional_embedding = None
        elif positional_encoding == "embedding":
            self._sinusoidal_positional_encoding = False
            self._positional_embedding = torch.nn.Embedding(positional_embedding_size, input_dim)
        else:
            raise ValueError("positional_encoding must be one of None, 'sinusoidal', or 'embedding'")

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def is_bidirectional(self) -> bool:
        return self._bidirectional

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, embedding_size = inputs.size()

        if mask is None:
            mask = cast(
                torch.BoolTensor,
                inputs.new_ones((batch_size, sequence_length), dtype=torch.bool),
            )

        if self._sinusoidal_positional_encoding:
            output = util.add_positional_features(inputs)
        if self._positional_embedding is not None:
            position_ids = torch.arange(inputs.size(1), dtype=torch.long, device=output.device)
            position_ids = position_ids.unsqueeze(0).expand(embedding_size)
            output = output + self._positional_embedding(position_ids)

        Q = inputs
        K = inputs
        V = inputs

        # Shape: (batch_size, sequence_length, sequence_length)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(embedding_size)

        # Shape: (batch_size, sequence_length, sequence_length)
        attention_mask = cast(torch.BoolTensor, mask.unsqueeze(1) & mask.unsqueeze(2))
        if self._bidirectional:
            attention_mask = cast(torch.BoolTensor, torch.tril(attention_mask))

        # Shape: (batch_size, sequence_length, sequence_length)
        normalized_attention_scores = util.masked_softmax(
            attention_scores,
            attention_mask,
            dim=2,
        )

        # Shape: (batch_size, sequence_length, embedding_size)
        output = torch.bmm(normalized_attention_scores, V)

        return cast(torch.Tensor, output)
