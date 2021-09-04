import torch
from xallennlp.modules.seq2seq_encoders import SelfAttentionEncoder


def test_self_attention_encoder() -> None:
    inputs = torch.randn(2, 3, 4)

    encoder = SelfAttentionEncoder(input_dim=4, positional_encoding="sinusoidal")
    assert encoder.is_bidirectional()

    output = encoder(inputs)
    assert output.size() == (2, 3, 4)
