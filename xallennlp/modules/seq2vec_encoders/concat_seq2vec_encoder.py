from typing import List, Optional

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import combine_tensors


@Seq2VecEncoder.register("concat")
class ConcatSeq2VecEncoder(Seq2VecEncoder):
    def __init__(
        self,
        encoders: List[Seq2VecEncoder],
        combination: Optional[str] = None,
    ) -> None:
        if len(set(encoder.get_input_dim() for encoder in encoders)) > 1:
            raise ConfigurationError("all input dims of encoders must be the same")

        super().__init__()

        self._combination = combination
        self._encoders = torch.nn.ModuleList(encoders)
        self._input_dim = encoders[0].get_input_dim()
        self._output_dim = sum(encoder.get_output_dim() for encoder in encoders)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        tensors = [encoder(inputs, mask) for encoder in self._encoders]
        if self._combination:
            return combine_tensors(self._combination, tensors)
        else:
            return torch.cat(tensors, dim=-1)
