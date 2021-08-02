import datetime
import re
from typing import Any, Dict, Optional, cast

import flatten_dict
import numpy
import torch
from allennlp.nn.util import tiny_value_of_dtype

REGEX_TIMEDELTA = re.compile(r"(?:(\d+) days?, )?(\d+):(\d+):(\d+)(?:\.(\d+))?")


def flatten_dict_for_mlflow_log(data: Dict[str, Any]) -> Dict[str, Any]:
    flattened_data: Dict[str, Any] = flatten_dict.flatten(data, reducer="dot", enumerate_types=(list, tuple))
    flattened_data = {str(key): value for key, value in flattened_data.items()}
    return flattened_data


def str_to_timedelta(delta_str: str) -> datetime.timedelta:
    match = re.match(REGEX_TIMEDELTA, delta_str)

    if not match:
        raise ValueError(f"invalid timedelta format: {delta_str}")

    nums = tuple(int(x or 0) for x in match.groups())
    days, hours, minutes, seconds, micros = nums

    seconds += 3600 * hours + 60 * minutes

    return datetime.timedelta(days=days, seconds=seconds, microseconds=micros)


def masked_fourier_transform(
    inputs: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    if mask is None:
        return cast(torch.Tensor, torch.fft.fft(inputs, dim=1))

    batch_size, max_length, embedding_dim = inputs.size()

    # Shape: (batch_size * embedding_dim, max_length)
    flattened_inputs = inputs.transpose(1, 2).reshape(batch_size * embedding_dim, max_length)
    flattened_inputs = torch.complex(flattened_inputs, flattened_inputs.new_zeros(flattened_inputs.size()))
    # Shape: (batch_size * embedding_dim, max_length)
    flattened_mask = mask.repeat_interleave(embedding_dim, dim=0).float()

    # Shape: (batch_size * embedding_dim, 1, 1)
    lenghts = flattened_mask.sum(dim=1, keepdim=True).unsqueeze(-1)

    # Shape: (batch_size * embedding_dim, max_length, max_length)
    inputs_indice = torch.arange(max_length).unsqueeze(0).unsqueeze(1).expand(batch_size * embedding_dim, 1, max_length)
    # Shape: (batch_size * embedding_dim, max_length, max_length)
    output_indice = torch.arange(max_length).unsqueeze(0).unsqueeze(2).expand(batch_size * embedding_dim, max_length, 1)
    # Shape: (batch_size * embedding_dim, max_length, max_length)
    weights = torch.exp(-2j * numpy.pi * inputs_indice * output_indice / (lenghts + tiny_value_of_dtype(torch.float)))
    weights = weights.to(inputs.device)
    weights = weights * flattened_mask.unsqueeze(1)

    # Shape: (batch_size max_length, embedding_dim)
    output = (
        torch.bmm(weights, flattened_inputs.unsqueeze(-1))
        .squeeze(-1)
        .view(batch_size, embedding_dim, max_length)
        .transpose(1, 2)
    )

    return output
