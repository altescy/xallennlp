import numpy
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from xallennlp.data.token_indexers.minhash_token_indexer import MinHashTokenIndexer


def test_minhash_token_indexers() -> None:
    indexer = MinHashTokenIndexer(num_features=16, num_hashes=64)
    tokens = [Token(word) for word in "this is a test sentence".split()]
    field = TextField(tokens, token_indexers={"tokens": indexer})

    vocab = Vocabulary()
    field.index(vocab)

    output = indexer.tokens_to_indices(tokens, vocab)
    assert isinstance(output, dict)

    vectors = output["tokens"]
    assert len(vectors) == 5
    assert isinstance(vectors, list)

    vector = vectors[0]
    assert isinstance(vector, numpy.ndarray)
    assert vector.shape == (16,)
    assert vector.sum() == 64
