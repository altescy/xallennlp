from typing import List, Optional

import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder


@Seq2SeqEncoder.register("unconst_compose")
class UnconstComposeEncoder(Seq2SeqEncoder):
    def __init__(self, encoders: List[Seq2SeqEncoder]) -> None:
        if not encoders:
            raise ValueError("Need at least one encoder.")

        if len(set(encoder.get_input_dim() for encoder in encoders)) > 1:
            raise ValueError("All encoders' input dim must be the same.")

        stateful = any(encoder.stateful for encoder in encoders)

        super().__init__(stateful=stateful)

        self._encoders = torch.nn.ModuleList(encoders)
        self._input_dim = int(self._encoders[0].get_input_dim())
        self._output_dim = sum(encoder.get_output_dim() for encoder in encoders)
        self._bidirectional = any(encoder.is_bidirectional() for encoder in encoders)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def is_bidirectional(self) -> bool:
        return self._bidirectional

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        output = torch.cat(
            [encoder(inputs, mask) for encoder in self._encoders],
            dim=-1,
        )
        return output
