from typing import Any, Dict, List, Optional, Tuple

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, TimeDistributed
from allennlp.modules.seq2seq_encoders import PassThroughEncoder, Seq2SeqEncoder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, F1Measure
import numpy as np
import torch


class PassThroughFeedForward(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self._input_dim = input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return inputs

    def get_output_dim(self) -> int:
        return self._input_dim


@Model.register("mrc_for_ner")
class MrcForNer(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder = None,
        start_feedforward: FeedForward = None,
        end_feedforward: FeedForward = None,
        span_feedforward: FeedForward = None,
        start_loss_weight: float = 1.0,
        end_loss_weight: float = 1.0,
        span_loss_weight: float = 1.0,
        lexical_dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: RegularizerApplicator = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder

        embedding_dim = self._text_field_embedder.get_output_dim()
        self._context_layer = context_layer \
            or PassThroughEncoder(embedding_dim)
        self._start_feedforward = start_feedforward \
            or PassThroughFeedForward(embedding_dim)
        self._end_feedforward = end_feedforward \
            or PassThroughFeedForward(embedding_dim)
        self._span_feedforward = span_feedforward \
            or PassThroughFeedForward(embedding_dim)
        self._start_loss_weight = start_loss_weight
        self._end_loss_weight = end_loss_weight
        self._span_loss_weight = span_loss_weight

        self._start_output = torch.nn.Linear(
            self._start_feedforward.get_output_dim(), 2)
        self._end_output = torch.nn.Linear(
            self._end_feedforward.get_output_dim(), 2)
        self._span_output = torch.nn.Linear(
            2 * self._span_feedforward.get_output_dim(), 1)

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x  # type: ignore

        self._span_loss = torch.nn.BCEWithLogitsLoss()

        self._accuracy = BooleanAccuracy()
        self._f1measure = F1Measure(positive_label=1)

        initializer(self)

    def forward(
        self,
        text: Dict[str, torch.LongTensor],
        span: torch.LongTensor,
        start_position: torch.IntTensor = None,
        end_position: torch.IntTensor = None,
        span_position: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
            text: ``Dict[str, torch.LongTensor]``, required.
                The output of a ``TextField`` representing the context-query
                concatenated text.
            span: ``torch.LongTensor``, required.
                A tensor of shape (batch_size, 1, 2), represeiting the span of
                context in the text.
            start_position: ``torch.IntTensor``, optional.
                A tensor of shape (batch_size, context_length), representing the
                start positions of entities.
            end_position: ``torch.IntTensor``, optional.
                A tensor of shape (batch_size, context_length), representing the
                end positions of entities.
            span_position: ``torch.IntTensor``, optional.
                A tensor of shape (batch_size, context_length, context_length),
                representing the start-end matching of spans.
            metadata: ``List[Dict[str, Any]]``, optional.
                A metadata dictionary for each instance in the batch. We use the
                "label" key from this dictionary, which is the label of query.
        """
        # Shape: (batch_size, sequence_length, embedding_size)
        text_embeddings = self._lexical_dropout(
            self._text_field_embedder(text))
        text_mask = util.get_text_field_mask(text).float()

        # Shape: (batch_size, 1, context_length, embedding_size)
        # Shape: (batch_size, 1, context_length)
        context_embeddings, context_mask = util.batched_span_select(
            self._context_layer(text_embeddings, text_mask),
            span,
        )

        # Shape: (batch_size, context_length, embedding_size)
        context_embeddings = context_embeddings.squeeze(1)
        # Shape: (batch_size, context_length)
        context_mask = context_mask.squeeze(1)

        # Shape: (batch_size, context_length, 2)
        start_logits = self._start_output(
            self._start_feedforward(context_embeddings))
        # Shape: (batch_size, context_length, 2)
        end_logits = self._end_output(
            self._end_feedforward(context_embeddings))

        # Shape: (batch_size, context_length, context_length, 2 * embedding_size)
        # Shape: (batch_size, context_length, context_length)
        span_embeddings, span_mask = \
            self._compute_span_embeddings(context_embeddings, context_mask)

        # Shape: (batch_size, sequence_length, sequence_length)
        span_logits = self._span_output(
            self._span_feedforward(span_embeddings)).squeeze()

        output_dict = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "span_logits": span_logits,
            "context_mask": context_mask,
            "span_mask": span_mask,
        }

        if (start_position is not None and end_position is not None
                and span_position is not None):
            start_loss = util.sequence_cross_entropy_with_logits(
                start_logits,
                start_position,
                context_mask,
            )
            end_loss = util.sequence_cross_entropy_with_logits(
                end_logits,
                end_position,
                context_mask,
            )
            span_loss = self._span_loss(
                span_logits.masked_select(span_mask),
                span_position.masked_select(span_mask).float(),
            )
            loss = (self._start_loss_weight * start_loss +
                    self._end_loss_weight * end_loss +
                    self._span_loss_weight * span_loss)

            start_preds = start_logits.argmax(-1)
            end_preds = end_logits.argmax(-1)
            span_preds = (span_logits.sigmoid() > 0.5).long()

            self._accuracy(
                span_preds *
                torch.einsum("bi,bj->bij", start_preds, end_preds),
                span_position,
                span_mask,
            )
            self._f1measure(
                torch.cat([
                    span_logits.sigmoid().unsqueeze(-1),
                    span_logits.new_zeros(span_logits.size()).unsqueeze(-1),
                ],
                          dim=-1),
                span_position,
                span_mask,
            )

            output_dict["loss"] = loss
            output_dict["start_loss"] = start_loss
            output_dict["end_loss"] = end_loss
            output_dict["span_loss"] = span_loss

        if metadata and "label" in metadata[0]:
            output_dict["label"] = [x["label"]
                                    for x in metadata]  # type: ignore

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self._accuracy.get_metric(reset)
        precision, recall, f1 = self._f1measure.get_metric(reset)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _compute_span_embeddings(
        self,
        context_embeddings: torch.Tensor,
        context_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, context_length, _ = context_embeddings.size()

        # Shape: (batch_size, context_length, context_length, 2 * embedding_size)
        span_embeddings = torch.cat([
            context_embeddings.unsqueeze(2).expand(batch_size, context_length,
                                                   context_length, -1),
            context_embeddings.unsqueeze(1).expand(batch_size, context_length,
                                                   context_length, -1),
        ], -1)

        # Shape: (batch_size, context_length, context_length)
        span_mask = \
            context_mask.unsqueeze(2).expand(batch_size, context_length, context_length) * \
            context_mask.unsqueeze(1).expand(batch_size, context_length, context_length)

        span_embeddings = self._span_feedforward(span_embeddings)

        return span_embeddings, span_mask
