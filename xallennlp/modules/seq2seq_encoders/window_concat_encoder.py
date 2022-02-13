from typing import Optional, Tuple, Union

import torch
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("window_concat")
class WindowConcatEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        window_size: Union[int, Tuple[int, int]],
    ) -> None:
        super().__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        if not all(s >= 0 for s in window_size):
            raise ValueError("Window size must be greater than or equal to zero.")
        self._input_dim = input_dim
        self._window_size = window_size

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return (sum(self._window_size) + 1) * self._input_dim

    def is_bidirectional(self) -> bool:
        return True

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        batch_size, max_length, embedding_dim = inputs.size()
        if mask is not None:
            inputs = inputs * mask.float().unsqueeze(2)

        output = inputs
        lws, rws = self._window_size
        if lws > 0:
            pad = inputs.new_zeros((batch_size, lws, embedding_dim))
            x = torch.cat([pad, inputs], dim=1)
            x = torch.cat([x[:, offset : offset + max_length] for offset in range(lws)], dim=2)
            output = torch.cat([output, x], dim=2)
        if rws > 0:
            pad = inputs.new_zeros((batch_size, rws, embedding_dim))
            x = torch.cat([inputs, pad], dim=1)
            x = torch.cat([x[:, offset : offset + max_length] for offset in range(1, rws + 1)], dim=2)
            output = torch.cat([output, x], dim=2)

        return output
