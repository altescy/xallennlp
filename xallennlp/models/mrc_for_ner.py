from typing import Any, Dict, List, Tuple

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, TimeDistributed
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, FBetaMeasure
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
        context_layer: Seq2SeqEncoder,
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

        self._context_layer = context_layer

        self._start_scorer = torch.nn.Sequential(
            TimeDistributed(start_feedforward),
            TimeDistributed(
                torch.nn.Linear(
                    start_feedforward.get_output_dim(),
                    2,
                )),
        ) if start_feedforward is not None else TimeDistributed(
            torch.nn.Linear(
                context_layer.get_output_dim(),
                2,
            ))

        self._end_scorer = torch.nn.Sequential(
            TimeDistributed(end_feedforward),
            TimeDistributed(
                torch.nn.Linear(
                    end_feedforward.get_output_dim(),
                    2,
                )),
        ) if end_feedforward is not None else TimeDistributed(
            torch.nn.Linear(
                context_layer.get_output_dim(),
                2,
            ))

        self._span_scorer = torch.nn.Sequential(
            TimeDistributed(span_feedforward),
            TimeDistributed(
                torch.nn.Linear(
                    span_feedforward.get_output_dim(),
                    2,
                )),
        ) if span_feedforward is not None else TimeDistributed(
            torch.nn.Linear(
                2 * context_layer.get_output_dim(),
                1,
            ))

        self._start_loss_weight = start_loss_weight
        self._end_loss_weight = end_loss_weight
        self._span_loss_weight = span_loss_weight

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x  # type: ignore

        self._endpoint_loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self._span_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")

        self._accuracy = BooleanAccuracy()
        self._f1measure = FBetaMeasure(beta=1.0, average="micro")

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
        # Shape: (batch_size, sequence_length)
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
        start_logits = self._start_scorer(context_embeddings)
        # Shape: (batch_size, context_length, 2)
        end_logits = self._end_scorer(context_embeddings)

        # Shape: (batch_size, context_length, context_length, 1)
        # Shape: (batch_size, context_length, context_length)
        span_logits, span_mask = \
            self._compute_span_scores(context_embeddings, context_mask)

        # Shape: (batch_size, context_length)
        start_predictions = start_logits.argmax(-1)
        # Shape: (batch_size, context_length)
        end_predictions = end_logits.argmax(-1)
        # Shape: (batch_size, context_length, context_length)
        span_predictions = (span_logits.squeeze(-1) > 0).long()

        output_dict = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "span_logits": span_logits,
            "start_predictions": start_predictions,
            "end_predictions": end_predictions,
            "span_predictions": span_predictions,
            "context_mask": context_mask,
            "span_mask": span_mask,
        }

        if (start_position is not None and end_position is not None
                and span_position is not None):
            context_mask = context_mask.bool()
            start_loss = self._endpoint_loss(
                start_logits[context_mask],
                start_position[context_mask],
            )
            end_loss = self._endpoint_loss(
                end_logits[context_mask],
                end_position[context_mask],
            )
            span_loss = self._span_loss(
                span_logits.squeeze(-1)[span_mask],
                span_position[span_mask].float(),
            )
            loss = (self._start_loss_weight * start_loss +
                    self._end_loss_weight * end_loss +
                    self._span_loss_weight * span_loss)

            self._accuracy(
                span_predictions *
                torch.einsum("bi,bj->bij", start_predictions, end_predictions),
                span_position,
                span_mask,
            )
            # F1 measure computed here is not a common metrics as known as
            # span-based F1 used for named entity recognition.
            self._f1measure(
                torch.cat(
                    [
                        span_logits.sigmoid(),
                        span_logits.new_zeros(span_logits.size()),
                    ],
                    dim=-1,
                ),
                span_position,
                span_mask,
            )

            output_dict["loss"] = loss

        if metadata and "label" in metadata[0]:
            output_dict["label"] = [x["label"]
                                    for x in metadata]  # type: ignore

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self._accuracy.get_metric(reset)
        f1metrics = self._f1measure.get_metric(reset)
        return {
            "accuracy": accuracy,
            **f1metrics,
        }

    def _compute_span_scores(
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

        # Shape: (batch_size, context_length, context_length, 1)
        span_embeddings = self._span_scorer(span_embeddings)

        return span_embeddings, span_mask
