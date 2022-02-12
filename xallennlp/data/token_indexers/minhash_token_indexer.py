from typing import Dict, Iterator, List

import numpy
import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import IndexedTokenList, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides

_DEFAULT_VALUE = "THIS IS A REALLY UNLIKELY VALUE THAT HAS TO BE A STRING"


@TokenIndexer.register("minhash")
class MinHashTokenIndexer(TokenIndexer):
    def __init__(
        self,
        num_features: int,
        num_hashes: int = 64,
        n_grams: int = 3,
        token_min_padding_length: int = 0,
        feature_name: str = "text",
        default_value: str = _DEFAULT_VALUE,
    ) -> None:
        super().__init__(token_min_padding_length)

        self._num_features = num_features
        self._num_hashes = num_hashes
        self._n_grams = n_grams
        self._salts = [hash(str(salt)) for salt in range(num_hashes)]

        self._feature_name = feature_name
        self._default_value = default_value

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
        return

    @overrides
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[numpy.ndarray]]:
        indices: List[numpy.ndarray] = [self._get_token_embedding(token) for token in tokens]
        return {"tokens": indices}

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        def padding_token() -> numpy.ndarray:
            return numpy.zeros(self._num_features, dtype=numpy.float32)

        tensor = torch.FloatTensor(
            pad_sequence_to_length(tokens["tokens"], padding_lengths["tokens"], default_value=padding_token)
        )
        return {"tokens": tensor}

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        return {"tokens": []}

    def _get_token_embedding(self, token: Token) -> numpy.ndarray:
        vector = numpy.zeros(self._num_features)
        fingerprint = self._compute_token_fingerprint(token)
        for value in fingerprint:
            vector[value % self._num_features] += 1
        return vector

    def _compute_token_fingerprint(self, token: Token) -> List[int]:
        text = self._get_feature_value(token)
        text = f"<{text}>"
        ngrams = set(ngram for n in range(self._n_grams) for ngram in self._get_ngrams(text, n + 1))
        fingerprint = [min(hash(f"{ngram}:{salt}") for ngram in ngrams for salt in self._salts)]
        return fingerprint

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

    @staticmethod
    def _get_ngrams(text: str, n: int) -> Iterator[str]:
        length = len(text)
        for start, end in zip(range(0, length), range(n, length + 1)):
            yield text[start:end]
