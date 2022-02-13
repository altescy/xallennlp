from typing import Any, Dict, Iterator, List, Optional

import numpy
import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.token_indexer import IndexedTokenList
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.vocabulary import Vocabulary

_DEFAULT_VALUE = "THIS IS A REALLY UNLIKELY VALUE THAT HAS TO BE A STRING"


@TokenIndexer.register("pretrained_transformer_minhash")
class PretrainedTransformerMinhashIndexer(TokenIndexer):
    def __init__(
        self,
        model_name: str,
        num_features: int,
        num_hashes: int = 64,
        n_grams: int = 3,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        feature_name: str = "text",
        continuation_prefix: str = "##",
        default_value: str = _DEFAULT_VALUE,
        token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)

        self._num_features = num_features
        self._num_hashes = num_hashes
        self._n_grams = n_grams

        self._feature_name = feature_name
        self._default_value = default_value
        self._continuation_prefix = continuation_prefix

        self._salts = [hash(str(salt)) for salt in range(num_hashes)]
        self._tokenizer = PretrainedTransformerTokenizer(model_name, tokenizer_kwargs=tokenizer_kwargs)

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
        return

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[numpy.ndarray]]:
        wordpieces, offsets = self._tokenizer.intra_word_tokenize([self._get_feature_value(token) for token in tokens])
        vectors: List[numpy.ndarray] = []
        for token, offset in zip(tokens, offsets):
            if offset is None:
                token_wordpieces = [token]
            else:
                token_wordpieces = wordpieces[offset[0] : offset[1] + 1]
            vector = self._get_token_embedding(token_wordpieces)
            vectors.append(vector)
        return {"tokens": vectors}

    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        def padding_token() -> numpy.ndarray:
            return numpy.zeros(self._num_features, dtype=numpy.float32)

        tensor = torch.FloatTensor(
            pad_sequence_to_length(tokens["tokens"], padding_lengths["tokens"], default_value=padding_token)
        )
        return {"tokens": tensor}

    def get_empty_token_list(self) -> IndexedTokenList:
        return {"tokens": []}

    def _get_token_embedding(self, wordpieces: List[Token]) -> numpy.ndarray:
        fingerprints = numpy.zeros((len(wordpieces), self._num_hashes), dtype=int)
        for i, token in enumerate(wordpieces):
            fingerprints[i, :] = self._compute_token_fingerprint(token)
        fingerprint = fingerprints.min(0)
        vector = numpy.zeros(self._num_features)
        for value in fingerprint:
            vector[value % self._num_features] += 1
        return vector

    def _compute_token_fingerprint(self, token: Token) -> List[int]:
        text = token.ensure_text()
        if text.startswith(self._continuation_prefix):
            ngrams = {text}
        else:
            ngrams = set(self._get_ngrams(text, self._n_grams))
        fingerprint = [min(hash(f"{ngram}:{salt}") for ngram in ngrams) for salt in self._salts]
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
        if length < n:
            yield text
        for start, end in zip(range(0, length), range(n, length + 1)):
            yield text[start:end]
