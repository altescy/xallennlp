from typing import Optional, cast

import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.time_distributed import TimeDistributed


@Seq2SeqEncoder.register("residual")
class ResidualSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        projection: bool = True,
    ) -> None:
        super().__init__(stateful=encoder.stateful)

        self._input_dim = encoder.get_input_dim()
        self._encoder = encoder
        self._projection: Optional[torch.nn.Module] = None
        if projection:
            self._projection = TimeDistributed(  # type: ignore
                torch.nn.Linear(
                    encoder.get_output_dim(),
                    encoder.get_input_dim(),
                )
            )
        else:
            check_dimensions_match(
                self._encoder.get_input_dim(),
                self._encoder.get_output_dim(),
                "encoder input dim",
                "encoder output dim",
            )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        # Shape: (batch_size, max_length, embedding_size)
        encodings = self._encoder(inputs, mask)
        if self._projection is not None:
            encodings = self._projection(encodings)

        output = inputs + encodings

        return cast(torch.Tensor, output)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self.get_input_dim()
