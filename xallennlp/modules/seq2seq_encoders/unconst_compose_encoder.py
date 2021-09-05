from typing import List, Optional

import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder


@Seq2SeqEncoder.register("unconst_compose")
class UnconstComposeEncoder(Seq2SeqEncoder):
    def __init__(self, encoders: List[Seq2SeqEncoder]) -> None:
        if not encoders:
            raise ValueError("Need at least one encoder.")

        stateful = any(encoder.stateful for encoder in encoders)

        super().__init__(stateful=stateful)

        self._encoders = torch.nn.ModuleList(encoders)
        self._input_dim = int(self._encoders[0].get_input_dim())
        self._output_dim = int(self._encoders[-1].get_output_dim())
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
        output = inputs

        for encoder in self._encoders:
            output = encoder(output, mask)

        return output
