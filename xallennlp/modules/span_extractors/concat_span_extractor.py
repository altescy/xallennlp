from typing import List, Optional

import torch
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.span_extractors.span_extractor_with_span_width_embedding import (
    SpanExtractorWithSpanWidthEmbedding,
)


@SpanExtractor.register("concat")
class ConcatSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    def __init__(
        self,
        span_extractors: List[SpanExtractor],
        num_width_embeddings: Optional[int] = None,
        span_width_embedding_dim: Optional[int] = None,
        bucket_widths: bool = False,
    ) -> None:
        if not span_extractors:
            raise ValueError("Need at least one span extractor.")
        if len(set(extractor.get_input_dim() for extractor in span_extractors)) > 1:
            raise ValueError("Span extractors' input dims must be the same.")

        input_dim = span_extractors[0].get_input_dim()

        super().__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,  # type: ignore[arg-type]
            span_width_embedding_dim=span_width_embedding_dim,  # type: ignore[arg-type]
            bucket_widths=bucket_widths,
        )
        self._span_extractors = torch.nn.ModuleList(span_extractors)
        self._combined_dim = sum(extractor.get_output_dim() for extractor in span_extractors)

    def get_output_dim(self) -> int:
        if self._span_width_embedding is not None:
            return int(self._combined_dim + self._span_width_embedding.get_output_dim())
        return self._combined_dim

    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: Optional[torch.BoolTensor] = None,
        span_indices_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        output = torch.cat(
            [
                extractor(
                    sequence_tensor,
                    span_indices,
                    sequence_mask,
                    span_indices_mask,
                )
                for extractor in self._span_extractors
            ],
            dim=-1,
        )
        return output
