import itertools
import shutil
import tempfile
from typing import Dict, List

import fasttext
import torch
from allennlp.common.file_utils import open_compressed
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import IndexedTokenList, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary

_DEFAULT_VALUE = "THIS IS A REALLY UNLIKELY VALUE THAT HAS TO BE A STRING"


@TokenIndexer.register("fasttext_subword")
class FastTextSubwordTokenIndexer(TokenIndexer):
    def __init__(
        self,
        pretrained_filename: str,
        feature_name: str = "text",
        default_value: str = _DEFAULT_VALUE,
        min_padding_length: int = 0,
        token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)

        self._fasttext = self._load_pretrained_model(pretrained_filename)
        self._hidden_dim = self._fasttext.get_dimension()
        self._feature_name = feature_name
        self._default_value = default_value
        self._min_padding_length = min_padding_length

    @staticmethod
    def _load_pretrained_model(filename: str) -> fasttext.FastText:
        with tempfile.NamedTemporaryFile("wb") as dc_file:
            with open_compressed(filename, "rb", encoding=None) as fp:
                shutil.copyfileobj(fp, dc_file)
            model = fasttext.load_model(dc_file.name)
        return model

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
        return

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[List[int]]]:
        def _get_indices(token: Token) -> List[int]:
            _, indices = self._fasttext.get_subwords(self._get_feature_value(token))
            indices += 1  # Add 1 to use 0 for padding index
            return [int(i) for i in indices]

        return {"token_subwords": [_get_indices(token) for token in tokens]}

    def get_padding_lengths(self, indexed_tokens: IndexedTokenList) -> Dict[str, int]:
        padding_lengths: Dict[str, int] = {}
        padding_lengths["token_subwords"] = max(len(indexed_tokens["token_subwords"]), self._token_min_padding_length)
        max_num_subwords = self._min_padding_length
        for token in indexed_tokens["token_subwords"]:
            max_num_subwords = max(len(token), max_num_subwords)  # type: ignore
        padding_lengths["num_token_subwords"] = max_num_subwords
        return padding_lengths

    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        # Pad the tokens.
        padded_tokens = pad_sequence_to_length(
            tokens["token_subwords"],
            padding_lengths["token_subwords"],
            default_value=lambda: [],
        )

        # Pad the characters within the tokens.
        desired_token_length = padding_lengths["num_token_subwords"]
        longest_token: List[int] = max(tokens["token_subwords"], key=len, default=[])  # type: ignore
        padding_value = 0
        if desired_token_length > len(longest_token):
            # Since we want to pad to greater than the longest token, we add a
            # "dummy token" so we can take advantage of the fast implementation of itertools.zip_longest.
            padded_tokens.append([padding_value] * desired_token_length)
        # pad the list of lists to the longest sublist, appending 0's
        padded_tokens = list(zip(*itertools.zip_longest(*padded_tokens, fillvalue=padding_value)))
        if desired_token_length > len(longest_token):
            # Removes the "dummy token".
            padded_tokens.pop()
        # Truncates all the tokens to the desired length, and return the result.
        return {"token_subwords": torch.LongTensor([list(token[:desired_token_length]) for token in padded_tokens])}

    def get_empty_token_list(self) -> IndexedTokenList:
        return {"tokens": []}

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
