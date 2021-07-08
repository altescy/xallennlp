from typing import List, Union

import torch
from allennlp.common import FromParams
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation


class FeedForward(torch.nn.Module, FromParams):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dims: Union[int, List[int]],
        activations: Union[Activation, List[Activation]],
        biases: Union[bool, List[bool]] = True,
        dropout: Union[float, List[float]] = 0.0,
    ) -> None:

        super().__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(biases, list):
            biases = [biases] * num_layers
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore
        if len(hidden_dims) != num_layers:
            raise ConfigurationError("len(hidden_dims) (%d) != num_layers (%d)" % (len(hidden_dims), num_layers))
        if len(activations) != num_layers:
            raise ConfigurationError("len(activations) (%d) != num_layers (%d)" % (len(activations), num_layers))
        if len(biases) != num_layers:
            raise ConfigurationError("len(biases) (%d) != num_layers (%d)" % (len(biases), num_layers))
        if len(dropout) != num_layers:
            raise ConfigurationError("len(dropout) (%d) != num_layers (%d)" % (len(dropout), num_layers))
        self._activations = torch.nn.ModuleList(activations)
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim, layer_bias in zip(input_dims, hidden_dims, biases):
            linear_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim, bias=layer_bias))
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        output = inputs
        for layer, activation, dropout in zip(self._linear_layers, self._activations, self._dropout):
            output = dropout(activation(layer(output)))
        return output
