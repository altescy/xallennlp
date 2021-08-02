import torch
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn.activations import SwishActivation

from xallennlp.modules.seq2vec_encoders import ComposeSeq2VecEncoder


def test_compose_seq2vec_encoder() -> None:
    inputs = torch.rand(4, 5, 8)
    encoder = ComposeSeq2VecEncoder(
        seq2seq_encoder=LstmSeq2SeqEncoder(input_size=8, hidden_size=8),
        seq2vec_encoder=BagOfEmbeddingsEncoder(embedding_dim=8),
        feedforward=FeedForward(input_dim=8, num_layers=2, hidden_dims=4, activations=SwishActivation()),
    )
    output = encoder(inputs)

    assert output.size() == (4, 4)
