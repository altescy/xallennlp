from typing import Optional, cast

import torch
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from xallennlp.utils import masked_pool


@Seq2VecEncoder.register("pooling")
class PoolingSeq2VecEncoder(Seq2VecEncoder):
    def __init__(
        self,
        input_dim: int,
        method: str = "average",
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._method = method

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = cast(torch.BoolTensor, mask.unsqueeze(-1))
        return masked_pool(inputs, mask, self._method)
