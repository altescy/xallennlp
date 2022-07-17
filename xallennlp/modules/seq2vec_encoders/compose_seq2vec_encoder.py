from typing import Optional

import torch
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


@Seq2VecEncoder.register("compose")
class ComposeSeq2VecEncoder(Seq2VecEncoder):
    def __init__(
        self,
        seq2seq_encoder: Seq2SeqEncoder,
        seq2vec_encoder: Seq2VecEncoder,
        feedforward: Optional[FeedForward] = None,
    ) -> None:
        super().__init__()
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        output = inputs

        output = self._seq2seq_encoder(output, mask)
        output = self._seq2vec_encoder(output, mask)

        if self._feedforward is not None:
            output = self._feedforward(output)

        return output

    def get_input_dim(self) -> int:
        return self._seq2seq_encoder.get_input_dim()

    def get_output_dim(self) -> int:
        if self._feedforward is not None:
            return int(self._feedforward.get_output_dim())  # type: ignore[no-untyped-call]
        return self._seq2vec_encoder.get_output_dim()
