import torch
from allennlp.nn.activations import SwishActivation
from xallennlp.modules.feedforward import FeedForward


def test_feedforward_without_bias() -> None:
    feedforward = FeedForward(
        input_dim=8,
        num_layers=3,
        hidden_dims=4,
        biases=False,
        activations=SwishActivation(),
    )
    inputs = torch.rand(16, 8)
    output = feedforward(inputs)

    assert output.size() == (16, 4)
