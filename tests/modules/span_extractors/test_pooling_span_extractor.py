import torch
from xallennlp.modules.span_extractors import PoolingSpanExtractor


def test_pooling_span_extractor() -> None:
    inputs = torch.rand((2, 4, 5))
    spans = torch.LongTensor([[[0, 2], [1, 1]], [[1, 2], [2, 3]]])
    extractor = PoolingSpanExtractor(input_dim=5)
    output = extractor(inputs, spans)

    assert output.size() == (2, 2, 5)

    inputs = torch.rand((2, 4, 5))
    spans = torch.LongTensor([[[0, 2], [1, 1]], [[1, 2], [2, 3]]])
    extractor = PoolingSpanExtractor(input_dim=5, num_width_embeddings=4, span_width_embedding_dim=3)
    output = extractor(inputs, spans)

    assert output.size() == (2, 2, 5 + 3)
