import numpy
from allennlp.data.fields.text_field import TextField
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from xallennlp.data.token_indexers.pretrained_transformer_minhash_indexer import PretrainedTransformerMinhashIndexer


def test_pretrained_transformer_token_indexer() -> None:
    indexer = PretrainedTransformerMinhashIndexer("bert-base-cased", num_features=32, num_hashes=64)

    tokens = [Token(text) for text in "this is a test sentence".split()]
    field = TextField(tokens, {"tokens": indexer})

    vocab = Vocabulary()
    field.index(vocab)

    output = indexer.tokens_to_indices(tokens, vocab)
    assert isinstance(output, dict)
    assert "tokens" in output

    vectors = output["tokens"]
    assert isinstance(vectors, list)
    assert len(vectors) == 5

    vector = vectors[0]
    assert isinstance(vector, numpy.ndarray)
    assert vector.sum() == 64
