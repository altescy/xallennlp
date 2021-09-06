import torch
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders import FeedForwardEncoder, LstmSeq2SeqEncoder
from allennlp.nn.activations import SwishActivation
from xallennlp.modules.seq2seq_encoders import UnconstComposeEncoder


def test_unconst_compose_encoder() -> None:
    inputs = torch.rand(2, 3, 4)
    encoder = UnconstComposeEncoder(
        [
            LstmSeq2SeqEncoder(
                input_size=4,
                hidden_size=3,
                num_layers=1,
                bidirectional=True,
            ),
            FeedForwardEncoder(
                FeedForward(
                    input_dim=6,
                    hidden_dims=3,
                    num_layers=2,
                    activations=SwishActivation(),  # type: ignore
                )
            ),
        ]
    )

    assert encoder.is_bidirectional()
    assert encoder.get_input_dim() == 4
    assert encoder.get_output_dim() == 3

    output = encoder(inputs)
    assert output.size() == (2, 3, 3)
