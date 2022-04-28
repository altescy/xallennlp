import shutil
import tempfile
from typing import Optional, cast

import fasttext
import torch
from allennlp.common.file_utils import open_compressed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn import util
from torch.nn.functional import embedding


@TokenEmbedder.register("fasttext_embedder")
class FasttextEmbedder(TokenEmbedder):
    def __init__(
        self,
        pretrained_filename: str,
        trainable: bool = True,
        projection_dim: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(self._load_pretrained_weight(pretrained_filename), requires_grad=trainable)
        self._max_norm = max_norm
        self._norm_type = norm_type
        self._scale_grad_by_freq = scale_grad_by_freq
        self._sparse = sparse
        self._projection = torch.nn.Linear(self._weight.size(1), projection_dim) if projection_dim else None
        self._output_dim = projection_dim or self._weight.size(1)

    @staticmethod
    def _load_pretrained_weight(filename: str) -> torch.Tensor:
        with tempfile.NamedTemporaryFile("wb") as dc_file:
            with open_compressed(filename, "rb", encoding=None) as fp:
                shutil.copyfileobj(fp, dc_file)
            model = fasttext.load_model(dc_file.name)
        pretrained_weight = model.get_input_matrix()
        vocab_size, embedding_dim = pretrained_weight.shape
        weight = torch.zeros((vocab_size + 1), embedding_dim)
        weight[1:] = torch.FloatTensor(pretrained_weight)
        return weight

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, token_subwords: torch.Tensor) -> torch.Tensor:
        mask = cast(torch.BoolTensor, (token_subwords != 0).unsqueeze(-1))
        subword_embeddings = embedding(
            token_subwords,
            self._weight,
            padding_idx=0,
            max_norm=self._max_norm,
            norm_type=self._norm_type,
            scale_grad_by_freq=self._scale_grad_by_freq,
            sparse=self._sparse,
        )
        embeddings = util.masked_mean(subword_embeddings, mask, dim=-2)
        if self._projection:
            embeddings = self._projection(embeddings)
        return embeddings
