import torch
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from xallennlp.modules.span_extractors import ConcatSpanExtractor


def test_concat_span_extractor() -> None:
    inputs = torch.rand((2, 4, 5))
    spans = torch.LongTensor([[[0, 2], [1, 1]], [[1, 2], [2, 3]]])

    extractor = ConcatSpanExtractor(
        span_extractors=[
            EndpointSpanExtractor(input_dim=5, combination="x,y"),
            SelfAttentiveSpanExtractor(input_dim=5),
        ],
        num_width_embeddings=4,
        span_width_embedding_dim=3,
    )
    assert extractor.get_input_dim() == 5
    assert extractor.get_output_dim() == 18

    output = extractor(inputs, spans)
    assert output.size() == (2, 2, 18)
