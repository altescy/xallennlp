import torch
from xallennlp.modules.distances import CosineDistance, EuclideanDistance


def test_euclidean_distance() -> None:
    dist = EuclideanDistance()
    source = torch.rand(3, 4)
    target = torch.rand(3, 4)
    output = dist(source, target)
    assert output.size() == (3,)


def test_cosine_distance() -> None:
    dist = CosineDistance()
    source = torch.rand(3, 4)
    target = torch.rand(3, 4)
    output = dist(source, target)
    assert output.size() == (3,)
