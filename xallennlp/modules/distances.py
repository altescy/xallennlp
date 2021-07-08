from typing import cast

import torch
from allennlp.common.registrable import Registrable
from allennlp.nn.util import tiny_value_of_dtype


class Distance(torch.nn.Module, Registrable):
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


@Distance.register("euclidean")
class EuclideanDistance(Distance):
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (source - target).norm(p=2, dim=-1)  # type: ignore


@Distance.register("cosine")
class CosineDistance(Distance):
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Shape: (batch_size, embedding_dim)
        source_norm = source / (
            source.norm(p=2, dim=-1, keepdim=True) + tiny_value_of_dtype(source.dtype)  # type: ignore
        )
        # Shape: (batch_size, embedding_dim)
        target_norm = target / (
            target.norm(p=2, dim=-1, keepdim=True) + tiny_value_of_dtype(target.dtype)  # type: ignore
        )
        # Shape: (batch_size, )
        similarity = (source_norm * target_norm).sum(-1)
        distances = 0.5 * (1 - similarity)

        return cast(torch.Tensor, distances)
