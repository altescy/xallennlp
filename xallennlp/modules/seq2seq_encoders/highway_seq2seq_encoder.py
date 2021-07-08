from typing import Optional, cast

import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.time_distributed import TimeDistributed


@Seq2SeqEncoder.register("highway")
class HighwaySeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        highway_encoder: Seq2SeqEncoder,
        transform_gate_encoder: Seq2SeqEncoder,
        carry_gate_encoder: Optional[Seq2SeqEncoder] = None,
        projection: bool = True,
    ) -> None:
        stateful = highway_encoder.stateful or transform_gate_encoder.stateful
        check_dimensions_match(
            highway_encoder.get_input_dim(),
            transform_gate_encoder.get_input_dim(),
            "highway_encoder input dim",
            "transform_gate_encoder input dim",
        )
        if carry_gate_encoder is not None:
            stateful = stateful or carry_gate_encoder.stateful
            check_dimensions_match(
                highway_encoder.get_input_dim(),
                carry_gate_encoder.get_input_dim(),
                "highway_encoder input dim",
                "carry_gate_encoder input dim",
            )

        super().__init__(stateful=stateful)

        self._input_dim = highway_encoder.get_input_dim()
        self._highway_encoder = highway_encoder
        self._transform_gate_encoder = transform_gate_encoder
        self._carry_gate_encoder = carry_gate_encoder
        self._highway_projection: Optional[torch.nn.Module] = None
        self._transform_gate_projection: Optional[torch.nn.Module] = None
        self._carry_gate_projection: Optional[torch.nn.Module] = None
        if projection:
            self._highway_projection = TimeDistributed(  # type: ignore
                torch.nn.Linear(
                    highway_encoder.get_output_dim(),
                    highway_encoder.get_input_dim(),
                )
            )
            self._transform_gate_projection = TimeDistributed(  # type: ignore
                torch.nn.Linear(
                    transform_gate_encoder.get_output_dim(),
                    transform_gate_encoder.get_input_dim(),
                ),
            )
            if carry_gate_encoder is not None:
                self._carry_gate_projection = TimeDistributed(  # type: ignore
                    torch.nn.Linear(
                        carry_gate_encoder.get_output_dim(),
                        carry_gate_encoder.get_input_dim(),
                    ),
                )
        else:
            assert highway_encoder.get_output_dim() in (self._input_dim, 1)
            assert transform_gate_encoder.get_output_dim() in (self._input_dim, 1)
            if carry_gate_encoder is not None:
                assert carry_gate_encoder.get_output_dim() in (self._input_dim, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        # Shape: (batch_size, max_length, embedding_size)
        highway_encodings = self._highway_encoder(inputs, mask)
        if self._highway_projection is not None:
            highway_encodings = self._highway_projection(highway_encodings)

        # Shape: (batch_size, max_length, embedding_size)
        transform_gate_logits = self._transform_gate_encoder(inputs, mask).sigmoid()
        if self._transform_gate_projection is not None:
            transform_gate_logits = self._transform_gate_projection(transform_gate_logits)
        transform_gate_scores = transform_gate_logits.sigmoid()

        # Shape: (batch_size, max_length, embedding_size)
        if self._carry_gate_encoder is None:
            carry_gate_scores = 1 - transform_gate_scores
        else:
            carry_gate_logits = self._carry_gate_encoder(inputs, mask).sigmoid()
            if self._carry_gate_projection is not None:
                carry_gate_logits = self._carry_gate_projection(carry_gate_logits)
            carry_gate_scores = carry_gate_logits.sigmoid()

        output = (carry_gate_scores * inputs) + (transform_gate_scores * highway_encodings)

        return cast(torch.Tensor, output)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self.get_input_dim()
