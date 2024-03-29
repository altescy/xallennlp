import torch
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import LstmSeq2SeqEncoder
from xallennlp.modules.seq2seq_encoders import ResidualSeq2SeqEncoder


def test_residual_seq2seq_encoder() -> None:
    encoder = ResidualSeq2SeqEncoder(
        LstmSeq2SeqEncoder(
            input_size=8,
            hidden_size=8,
        ),
        projection=False,
    )

    assert not encoder.is_bidirectional()

    inputs = torch.randn((16, 4, 8))
    output = encoder(inputs)

    assert output.size() == (16, 4, 8)


def test_residual_seq2seq_encoder_with_projection() -> None:
    encoder = ResidualSeq2SeqEncoder(
        LstmSeq2SeqEncoder(
            input_size=8,
            hidden_size=4,
        ),
        projection=True,
    )

    assert not encoder.is_bidirectional()

    inputs = torch.randn((16, 4, 8))
    output = encoder(inputs)

    assert output.size() == (16, 4, 8)
