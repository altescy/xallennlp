import torch
from xallennlp.modules.seq2seq_encoders import FNetEncoder


def test_fnet_encoder() -> None:
    inputs = torch.rand(8, 5, 16)

    encoder = FNetEncoder(16, 4, 32, positional_encoding="sinusoidal")
    assert encoder.is_bidirectional()

    output = encoder(inputs)
    assert output.size() == (8, 5, 16)
