import torch
from xallennlp.modules.seq2vec_encoders import PoolingSeq2VecEncoder


def test_pooling_seq2vec_encoder() -> None:
    inputs = torch.rand(4, 5, 8)
    encoder = PoolingSeq2VecEncoder(input_dim=8, method="max")
    output = encoder(inputs)

    assert output.size() == (4, 8)
