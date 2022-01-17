import torch
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from xallennlp.modules.seq2vec_encoders import ConcatSeq2VecEncoder


def test_concat_seq2vec_encoder() -> None:
    encoder = ConcatSeq2VecEncoder(
        encoders=[
            BagOfEmbeddingsEncoder(embedding_dim=4),
            BagOfEmbeddingsEncoder(embedding_dim=4),
        ]
    )

    inputs = torch.randn(2, 3, 4)
    output = encoder(inputs)

    assert output.size() == (2, 8)


def test_concat_seq2vec_encoder_with_combination() -> None:
    encoder = ConcatSeq2VecEncoder(
        encoders=[
            BagOfEmbeddingsEncoder(embedding_dim=4),
            BagOfEmbeddingsEncoder(embedding_dim=4),
        ],
        combination="1,2,1*2",
    )

    inputs = torch.randn(2, 3, 4)
    output = encoder(inputs)

    assert output.size() == (2, 12)
