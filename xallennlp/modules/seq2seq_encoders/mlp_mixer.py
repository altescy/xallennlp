from typing import Optional, cast

import torch
import torch.nn.functional as F
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


class SpatialLinear(torch.nn.Module):
    def __init__(self, spatial_dim: int) -> None:
        super().__init__()
        self._weights = torch.nn.Parameter(torch.randn(spatial_dim, spatial_dim))
        self._biases = torch.nn.Parameter(torch.randn(spatial_dim))

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, max_length, embedding_dim = inputs.size()

        # Shape: (batch_size * embedding_dim, max_length)
        inputs = inputs.transpose(1, 2).reshape(-1, max_length)
        if mask is not None:
            # Shape: (batch_size * embedding_dim, max_length)
            mask = mask.repeat_interleave(embedding_dim, dim=0)
            inputs *= mask

        # Shape: (max_length, max_length)
        weights = self._weights[:max_length, :max_length]
        # Shape: (max_length)
        biases = self._biases[:max_length]

        # Shape: (batch_size * embedding_dim, max_length)
        output = inputs @ weights + biases
        # Shape: (batch_size, max_length, embedding_dim)
        output = output.reshape(batch_size, embedding_dim, max_length).transpose(1, 2)

        return output


class MixerLayer(torch.nn.Module):
    def __init__(
        self,
        channel_dim: int,
        spatial_dim: int,
    ) -> None:
        super().__init__()
        self._layer_norm_1 = torch.nn.LayerNorm(channel_dim)
        self._layer_norm_2 = torch.nn.LayerNorm(channel_dim)
        self._spatial_linear_1 = SpatialLinear(spatial_dim)
        self._spatial_linear_2 = SpatialLinear(spatial_dim)
        self._channel_linear_1 = torch.nn.Linear(channel_dim, channel_dim)
        self._channel_linear_2 = torch.nn.Linear(channel_dim, channel_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res1 = inputs
        h = self._layer_norm_1(inputs)

        h = self._spatial_linear_1(h, mask)
        h = F.gelu(h)
        h = self._spatial_linear_2(h, mask)

        h = h + res1
        res2 = h

        h = self._layer_norm_2(h)

        h = self._channel_linear_1(h)
        h = F.gelu(h)
        h = self._channel_linear_2(h)

        h = h + res2
        return cast(torch.Tensor, h)


@Seq2SeqEncoder.register("mlp_mixer")
class MLPMixer(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        max_length: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim

        self._layers = torch.nn.ModuleList([MixerLayer(input_dim, max_length) for _ in range(num_layers)])
        self._dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=dropout) for _ in range(num_layers)])

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        h = inputs
        for layer, dropout in zip(self._layers, self._dropouts):
            h = dropout(layer(h, mask))
        return h
