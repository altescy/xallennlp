from typing import List, Optional, cast

import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.transformer.layer_norm import LayerNorm
from allennlp.nn.activations import Activation, GeluFast
from allennlp.nn.util import add_positional_features


class FNetLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        feedforward_hidden_dim: int,
        activation: Activation,
    ) -> None:
        super().__init__()  # type: ignore
        self._input_dim = input_dim
        self._feedforward = TimeDistributed(  # type: ignore
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, feedforward_hidden_dim),
                activation,
                torch.nn.Linear(feedforward_hidden_dim, input_dim),
            )
        )
        self._layer_nomralize_fft = LayerNorm(input_dim)
        self._layer_nomralize_ffn = LayerNorm(input_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        output = inputs
        if mask is not None:
            output = output * mask

        # Fourier Transform
        output_fft = inputs
        output_fft = torch.fft.fft(output_fft, dim=2)
        output_fft = torch.fft.fft(output_fft, dim=1)
        output_fft = output_fft.real

        output = self._layer_nomralize_fft(output_fft + output)
        if mask is not None:
            output = output * mask

        # Feed Forward
        output_ffn = output
        output_ffn = self._feedforward(output_ffn)

        output = self._layer_nomralize_ffn(output_ffn + output)
        if mask is not None:
            output = output * mask
        return output


@Seq2SeqEncoder.register("fnet")
class FNetEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        feedforward_hidden_dim: int = 2048,
        positional_encoding: Optional[str] = None,
        positional_embedding_size: int = 512,
        activation: Optional[Activation] = None,
        dropout: float = 0.0,
    ) -> None:
        if activation is None:
            activation = GeluFast()  # type: ignore

        super().__init__(stateful=False)

        self._input_dim = input_dim

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

        fnet_layers: List[FNetLayer] = []
        for _ in range(num_layers):
            fnet_layers.append(
                FNetLayer(
                    input_dim=input_dim,
                    feedforward_hidden_dim=feedforward_hidden_dim,
                    activation=activation,
                )
            )
        self._fnet_layers = torch.nn.ModuleList(fnet_layers)

        dropout_layers = [torch.nn.Dropout(p=dropout) for _ in range(num_layers)]
        self._dropout_layers = torch.nn.ModuleList(dropout_layers)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def is_bidirectional(self) -> bool:
        return False

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        output = inputs
        if self._sinusoidal_positional_encoding:
            output = add_positional_features(output)
        if self._positional_embedding is not None:
            position_ids = torch.arange(inputs.size(1), dtype=torch.long, device=output.device)
            position_ids = position_ids.unsqueeze(0).expand(inputs.shape[:-1])
            output = output + self._positional_embedding(position_ids)

        if mask is not None:
            mask = cast(torch.FloatTensor, mask.unsqueeze(-1).float())  # type: ignore

        for layer, dropout in zip(self._fnet_layers, self._dropout_layers):
            output = dropout(layer(output, mask))

        return output
