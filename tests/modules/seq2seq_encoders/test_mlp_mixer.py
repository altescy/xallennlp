import torch
from xallennlp.modules.seq2seq_encoders.mlp_mixer import MLPMixer


def test_mlp_mixer() -> None:
    inputs = torch.randn(2, 3, 4)
    module = MLPMixer(input_dim=4, num_layers=2, dropout=0.5)
    output = module(inputs)

    assert output.size() == (2, 3, 4)
