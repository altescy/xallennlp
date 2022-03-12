import torch
from xallennlp.modules.seq2seq_encoders.hypermixer import HyperMixer


def test_hypermixer() -> None:
    inputs = torch.randn(2, 3, 4)
    mask = torch.BoolTensor([[True, True, False], [False, False, False]])
    module = HyperMixer(input_dim=4, hidden_dim=4, num_layers=2, dropout=0.1)
    output = module(inputs, mask)

    assert output.size() == (2, 3, 4)
