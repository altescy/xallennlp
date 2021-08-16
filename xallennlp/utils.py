import datetime
import re
from typing import Any, Dict, Optional, cast

import flatten_dict
import torch

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


def masked_fft(
    inputs: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    if mask is None:
        return cast(torch.Tensor, torch.fft.fft(inputs, dim=1))

    batch_size, max_length, embedding_dim = inputs.size()

    lengths = mask.long().sum(dim=1)
    output = torch.zeros((batch_size, max_length, embedding_dim), dtype=torch.complex64).to(inputs.device)

    for i, (x, l) in enumerate(zip(inputs, lengths)):
        output[i, :l, :] = torch.fft.fft(x[:l, :], dim=0)

    return output
