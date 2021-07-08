from typing import Dict, List, cast

import fasttext
import numpy
import torch
from allennlp.common.file_utils import cached_path
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import IndexedTokenList, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary

_DEFAULT_VALUE = "THIS IS A REALLY UNLIKELY VALUE THAT HAS TO BE A STRING"


@TokenIndexer.register("fasttext")
class FastTextTokenIndexer(TokenIndexer):
    def __init__(
        self,
        pretrained_filename: str,
        token_min_padding_length: int = 0,
        feature_name: str = "text",
        default_value: str = _DEFAULT_VALUE,
        normalize: bool = True,
    ) -> None:
        super().__init__(token_min_padding_length)
        pretrained_filename = cached_path(pretrained_filename)

        self._fasttext = fasttext.load_model(pretrained_filename)
        self._hidden_dim = self._fasttext.get_dimension()
        self._feature_name = feature_name
        self._default_value = default_value
        self._normalize = normalize

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
        return

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[numpy.ndarray]]:
        indices: List[numpy.ndarray] = [self._get_token_embedding(token) for token in tokens]
        return {"tokens": indices}

    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        def padding_token() -> numpy.ndarray:
            return numpy.zeros(self._hidden_dim, dtype=numpy.float32)

        tensor = torch.FloatTensor(
            pad_sequence_to_length(tokens["tokens"], padding_lengths["tokens"], default_value=padding_token)
        )
        return {"tokens": tensor}

    def get_empty_token_list(self) -> IndexedTokenList:
        return {"tokens": []}

    def _get_token_embedding(self, token: Token) -> numpy.ndarray:
        text = self._get_feature_value(token)
        embedding = cast(numpy.ndarray, self._fasttext[text])

        if self._normalize:
            embedding = embedding / (numpy.linalg.norm(embedding) + 1e-10)

        return embedding

    def _get_feature_value(self, token: Token) -> str:
        text = getattr(token, self._feature_name)
        if text is None:
            if self._default_value is not _DEFAULT_VALUE:
                text = self._default_value
            else:
                raise ValueError(
                    f"{token} did not have attribute {self._feature_name}. If you "
                    "want to ignore this kind of error, give a default value in the "
                    "constructor of this indexer."
                )
        return str(text)
