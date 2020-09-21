import json
import logging
from typing import Any, Dict, Iterable, List, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import ArrayField, Field, ListField, MetadataField, SpanField, TextField
from allennlp.data.tokenizers import SpacyTokenizer, Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
import numpy as np
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("mrc_for_ner")
class MrcForNerDatasetReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        context_tokenizer: Tokenizer = None,
        query_tokenizer: Tokenizer = None,
        start_token: str = "[CLS]",
        separate_token: str = "[SEP]",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {
            "token": SingleIdTokenIndexer()
        }
        self._context_tokenizer = context_tokenizer or WhitespaceTokenizer()
        self._query_tokenizer = query_tokenizer or SpacyTokenizer()
        self._start_token = start_token
        self._separate_token = separate_token

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)

        with open(file_path, "r") as f:
            for line in f:
                example = json.loads(line)
                context: str = example.pop("context")
                query: str = example.pop("query")
                metadata: Dict[str, Any] = example

                yield self.text_to_instance(
                    context,
                    query,
                    metadata,
                )

    def text_to_instance(
        self,
        context: str,
        query: str,
        metadata: Dict[str, Any],
    ) -> Instance:
        # pylint: disable=arguments-differ
        context_tokens = self._context_tokenizer.tokenize(context)
        query_tokens = self._query_tokenizer.tokenize(query)

        tokens, span = self._concat_context_and_query(context_tokens,
                                                      query_tokens)

        text_field = TextField(
            tokens,
            self._token_indexers,
        )
        span_field = SpanField(span[0], span[1], text_field)

        fields: Dict[str, Field] = {}
        fields["text"] = text_field
        fields["span"] = ListField([span_field])

        if "start_position" in metadata:
            assert "end_position" in metadata
            assert "span_position" in metadata

            start_position = metadata.pop("start_position")
            end_position = metadata.pop("end_position")
            span_position = metadata.pop("span_position")

            start_position_array = self._position_to_array(
                start_position, len(context_tokens))
            end_position_array = self._position_to_array(
                end_position, len(context_tokens))
            span_position_array = self._span_position_to_array(
                span_position, len(context_tokens))

            fields["start_position"] = ArrayField(start_position_array,
                                                  dtype=np.int)
            fields["end_position"] = ArrayField(end_position_array,
                                                dtype=np.int)
            fields["span_position"] = ArrayField(span_position_array,
                                                 dtype=np.int)

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def _concat_context_and_query(
            self, context: List[Token],
            query: List[Token]) -> Tuple[List[Token], Tuple[int, int]]:
        tokens = [Token(self._start_token)] if self._start_token else []
        tokens.extend(query)
        tokens.append(Token(self._separate_token))
        tokens.extend(context)
        context_span = (len(query) + 2, len(tokens) - 1)
        return tokens, context_span

    @staticmethod
    def _position_to_array(position: List[int], length: int) -> np.ndarray:
        ret = np.zeros(length, dtype=np.int)
        ret[position] = 1
        return ret

    @staticmethod
    def _parse_span_position(span_position_str: str) -> Tuple[int, int]:
        """
        Convert a span position string "start,end" to a tuple (start, end).
        """
        splitted_position = span_position_str.split(",")
        assert len(splitted_position) == 2

        left_str, right_str = splitted_position

        left_index = int(left_str.strip())
        right_index = int(right_str.strip())

        return (left_index, right_index)

    def _span_position_to_array(
        self,
        span_position: List[str],
        length: int,
    ) -> np.ndarray:
        ret = np.zeros((length, length), dtype=np.int)
        for span_position_str in span_position:
            left, right = self._parse_span_position(span_position_str)
            ret[left, right] = 1
        return ret
