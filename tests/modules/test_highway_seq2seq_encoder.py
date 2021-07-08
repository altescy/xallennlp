import torch
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import LstmSeq2SeqEncoder

from xallennlp.modules.seq2seq_encoders import HighwaySeq2SeqEncoder


def test_highway_seq2seq_encoder() -> None:
    encoder = HighwaySeq2SeqEncoder(
        LstmSeq2SeqEncoder(
            input_size=8,
            hidden_size=8,
        ),
        LstmSeq2SeqEncoder(
            input_size=8,
            hidden_size=8,
        ),
        LstmSeq2SeqEncoder(
            input_size=8,
            hidden_size=8,
        ),
        projection=False,
    )

    inputs = torch.randn((16, 4, 8))
    output = encoder(inputs)

    assert output.size() == (16, 4, 8)


def test_highway_seq2seq_encoder_without_carry_gate_encoder() -> None:
    encoder = HighwaySeq2SeqEncoder(
        LstmSeq2SeqEncoder(
            input_size=8,
            hidden_size=8,
        ),
        LstmSeq2SeqEncoder(
            input_size=8,
            hidden_size=8,
        ),
        projection=False,
    )

    inputs = torch.randn((16, 4, 8))
    output = encoder(inputs)

    assert output.size() == (16, 4, 8)


def test_highway_seq2seq_encoder_with_timestep_level_gate() -> None:
    encoder = HighwaySeq2SeqEncoder(
        LstmSeq2SeqEncoder(
            input_size=8,
            hidden_size=8,
        ),
        LstmSeq2SeqEncoder(
            input_size=8,
            hidden_size=1,
        ),
        projection=False,
    )

    inputs = torch.randn((16, 4, 8))
    output = encoder(inputs)

    assert output.size() == (16, 4, 8)


def test_highway_seq2seq_encoder_with_projection() -> None:
    encoder = HighwaySeq2SeqEncoder(
        LstmSeq2SeqEncoder(
            input_size=8,
            hidden_size=8,
        ),
        LstmSeq2SeqEncoder(
            input_size=8,
            hidden_size=4,
        ),
        projection=True,
    )

    inputs = torch.randn((16, 4, 8))
    output = encoder(inputs)

    assert output.size() == (16, 4, 8)
