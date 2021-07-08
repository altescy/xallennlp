import tempfile

import fasttext
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary

from xallennlp.data.token_indexers import FastTextTokenIndexer


class TestFastTextTokenIndexer:
    @staticmethod
    def test_as_array_produces_token_array() -> None:

        with tempfile.TemporaryDirectory() as tempdir:
            dataset_filename = f"{tempdir}/dataset.txt"
            pretrained_filename = f"{tempdir}/fasttext.model"

            with open(dataset_filename, "w") as fp:
                fp.write("\n".join(["this is a first sentence", "this is a second sentence"]))

            model = fasttext.train_unsupervised(
                dataset_filename,
                model="skipgram",
                dim=10,
                minCount=1,
            )
            model.save_model(pretrained_filename)

            indexer = FastTextTokenIndexer(pretrained_filename=pretrained_filename)
            tokens = [Token(word) for word in "this is a test sentence".split()]
            field = TextField(tokens, token_indexers={"tokens": indexer})

            vocab = Vocabulary()
            field.index(vocab)

            array_dict = indexer.tokens_to_indices(tokens, vocab)
            assert len(array_dict["tokens"]) == 5
            assert len(array_dict["tokens"][0]) == 10
