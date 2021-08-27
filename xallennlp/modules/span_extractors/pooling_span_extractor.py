from typing import Optional, cast

import torch
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.span_extractors.span_extractor_with_span_width_embedding import (
    SpanExtractorWithSpanWidthEmbedding,
)
from allennlp.nn.util import batched_span_select
from xallennlp.utils import masked_pool


@SpanExtractor.register("pooling")
class PoolingSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    def __init__(
        self,
        input_dim: int,
        method: str = "average",
        num_width_embeddings: Optional[int] = None,
        span_width_embedding_dim: Optional[int] = None,
        bucket_widths: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,  # type: ignore[arg-type]
            span_width_embedding_dim=span_width_embedding_dim,  # type: ignore[arg-type]
            bucket_widths=bucket_widths,
        )
        self._method = method

    def get_output_dim(self) -> int:
        return self._input_dim

    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: Optional[torch.BoolTensor] = None,
        span_indices_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        # Shape: (batch_size, max_spans, max_span_width, embedding_dim)
        # Shape: (batch_size, max_spans, max_span_width)
        embeddings, span_mask = batched_span_select(sequence_tensor, span_indices)

        # Shape: (batch_size, max_spans, max_span_width, 1)
        span_mask = cast(torch.BoolTensor, span_mask.unsqueeze(-1))

        # Shape: (batch_size, max_spans, embedding_dim)
        span_embeddings = masked_pool(embeddings, span_mask, self._method, dim=2)

        return span_embeddings
