import json
from typing import Any, Dict, Iterator, List, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import (AdjacencyField, Field, ListField,
                                  MetadataField, SequenceLabelField, SpanField,
                                  TextField)


class SciERCReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_span_width: int = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(),
        }
        self._max_span_width = max_span_width

    def _read(self, file_path) -> Iterator[Instance]:
        file_path = cached_path(file_path)

        with open(file_path, "r") as f:
            for line in f:
                yield self.text_to_instance(**json.loads(line))

    def text_to_instance(
        self,
        sentences: List[List[str]],
        ner: List[List[Tuple[int, int, str]]] = None,
        clusters: List[List[Tuple[int, int]]] = None,
        relations: List[List[Tuple[int, int, int, int, str]]] = None,
        doc_key: str = None,
    ) -> Instance:
        # pylint: disable=arguments-differ

        metadata: Dict[str, Any] = {
            "original_sentences": sentences,
            "doc_key": doc_key,
        }

        flattened_sentences = [
            word for sentence in sentences for word in sentence
        ]

        text_field = TextField(
            [Token(word) for word in flattened_sentences],
            self._token_indexers,
        )

        spans: List[Field] = []
        span_dict: Dict[Tuple[int, int], int] = {}

        sentence_offset = 0
        for sentence in sentences:
            for start, end in enumerate_spans(
                    sentence,
                    offset=sentence_offset,
                    max_span_width=self._max_span_width,
            ):
                spans.append(SpanField(start, end, text_field))
                span_dict[(start, end)] = len(span_dict)

            sentence_offset += len(sentence)

        span_fields = ListField(spans)
        meta_fields = MetadataField(metadata)

        fields: Dict[str, Field] = {
            "text": text_field,
            "spans": spans,
            "metadata": meta_fields,
        }

        if ner is not None:
            ner_labels = self._get_ner_labels(ner, span_dict)
            fields["ner"] = SequenceLabelField(
                ner_labels,
                span_fields,
                label_namespace="ner",
            )

        if clusters is not None:
            coref_labels = self._get_coref_labels(clusters, span_dict)
            fields["coref"] = SequenceLabelField(
                coref_labels,
                span_fields,
                label_namespace="coref",
            )

        if relations is not None:
            relation_labels = self._get_relation_labels(relations, span_dict)
            fields["relation"] = AdjacencyField(
                list(relation_labels.keys()),
                span_fields,
                labels=list(relation_labels.values()),
                label_namespace="relation",
            )

        return Instance(fields)

    @staticmethod
    def _get_ner_labels(
        ner: List[List[Tuple[int, int, str]]],
        span_dict: Dict[Tuple[int, int], int],
    ) -> List[str]:
        ner_dict = {}
        if ner is not None:
            flattened_ner = [
                entity for ner_in_sentence in ner for entity in ner_in_sentence
            ]
            for start, end, ner_label in flattened_ner:
                ner_dict[(start, end)] = ner_label

        ner_labels: List[str] = []
        for mention in span_dict:
            ner_labels.append(ner_dict.get(mention, "@@NONE@@"))

        return ner_labels

    @staticmethod
    def _get_coref_labels(
        clusters: List[List[Tuple[int, int]]],
        span_dict: Dict[Tuple[int, int], int],
    ) -> List[int]:
        cluster_dict = {}
        if clusters is not None:
            for cluster_id, cluster in enumerate(clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        coref_labels: List[int] = []
        for mention in span_dict:
            if coref_labels is not None:
                coref_labels.append(cluster_dict.get(mention, -1))

        return coref_labels

    @staticmethod
    def _get_relation_labels(
        relations: List[List[Tuple[int, int, int, int, str]]],
        span_dict: Dict[Tuple[int, int], int],
    ) -> Dict[Tuple[int, int], str]:
        flattened_relations = [
            relation for relations_in_sentence in relations
            for relation in relations_in_sentence
        ]

        relation_labels: Dict[Tuple[int, int], str] = {}
        for (
                source_start,
                source_end,
                target_start,
                target_end,
                relation_label,
        ) in flattened_relations:
            source_span_id = span_dict[(source_start, source_end)]
            target_span_id = span_dict[(target_start, target_end)]
            span_pair = (source_span_id, target_span_id)
            relation_labels[span_pair] = relation_label
        return relation_labels
