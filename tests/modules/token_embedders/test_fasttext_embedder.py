from pathlib import Path

import fasttext
import numpy.testing
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from xallennlp.data.token_indexers.fasttext_subword_token_indexer import FastTextSubwordTokenIndexer
from xallennlp.modules.token_embedders.fasttext_embedder import FasttextEmbedder


def test_fasttext_embedder(tmp_path: Path) -> None:
    dataset_filename = tmp_path / "dataset.txt"
    pretrained_filename = tmp_path / "fasttext.model"

    with open(dataset_filename, "w") as fp:
        fp.write("\n".join(["this is a first sentence", "this is a second sentence"]))

    model = fasttext.train_unsupervised(
        str(dataset_filename),
        model="skipgram",
        dim=10,
        minCount=1,
    )
    model.save_model(str(pretrained_filename))

    vocab = Vocabulary()
    indexer = FastTextSubwordTokenIndexer(pretrained_filename=str(pretrained_filename))

    tokens = [Token(t) for t in "this is a first sentence".split()]
    indexed_tokens = indexer.tokens_to_indices(tokens, vocab)
    padding_lengths = indexer.get_padding_lengths(indexed_tokens)
    tensor_dict = indexer.as_padded_tensor_dict(indexed_tokens, padding_lengths)

    embedder = FasttextEmbedder(pretrained_filename=str(pretrained_filename))
    output = embedder(**tensor_dict)

    assert embedder.get_output_dim() == 10
    assert output.size() == (5, 10)
    numpy.testing.assert_almost_equal(model["this"], output[0].detach().cpu().numpy())
