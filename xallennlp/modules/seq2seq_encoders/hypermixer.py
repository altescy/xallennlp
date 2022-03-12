from typing import Optional, cast

import torch
import torch.nn.functional as F
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.util import add_positional_features


class TokenMixer(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self._feedforward = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(-1)
            inputs *= mask

        # Shape: (batch_size, max_length, hidden_dim)
        W1 = self._compute_weights(inputs)
        # Shape: (batch_size, hidden_dim, max_length)
        W2 = W1.transpose(1, 2)

        # Shape: (batch_size, hidden_dim, embedding_dim)
        output = torch.bmm(W2, inputs)
        output = F.gelu(output)
        # Shape: (batch_size, max_length, embedding_dim)
        output = torch.bmm(W1, output)

        return output

    def _compute_weights(self, inputs: torch.Tensor) -> torch.Tensor:
        output = add_positional_features(inputs)
        output = self._feedforward(output)
        return cast(torch.Tensor, output)


class HyperMixerLayer(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self._layer_norm_1 = torch.nn.LayerNorm(input_dim)
        self._layer_norm_2 = torch.nn.LayerNorm(input_dim)
        self._token_mixer = TokenMixer(input_dim, hidden_dim)
        self._feedforward = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res1 = inputs
        output = self._layer_norm_1(inputs)

        output = self._token_mixer(output)
        output = output + res1

        res2 = output
        output = self._layer_norm_2(output)

        output = self._feedforward(output)
        output = output + res2

        return cast(torch.Tensor, output)


@Seq2SeqEncoder.register("hypermixer")
class HyperMixer(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([HyperMixerLayer(input_dim, hidden_dim) for _ in range(num_layers)])
        self._dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=dropout) for _ in range(num_layers)])

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def is_bidirectional(self) -> bool:
        return True

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        h = inputs
        m = mask.float() if mask is not None else None
        for layer, dropout in zip(self._layers, self._dropouts):
            h = dropout(layer(h, m))
        return h
