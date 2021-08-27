import torch
from xallennlp.modules.span_extractors import PoolingSpanExtractor


def test_pooling_span_extractor() -> None:
    inputs = torch.rand((2, 4, 5))
    spans = torch.LongTensor([[[0, 2], [1, 1]], [[1, 2], [2, 3]]])
    extractor = PoolingSpanExtractor(input_dim=5)
    output = extractor(inputs, spans)

    assert output.size() == (2, 2, 5)
