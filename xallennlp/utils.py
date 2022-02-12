import datetime
import os
import re
from typing import Any, Dict, Optional, cast
from urllib.parse import urlparse

import flatten_dict
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import masked_max, masked_mean, replace_masked_values

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


def get_serialization_dir(path: str = "outputs") -> str:
    try:
        import mlflow

        active_run = mlflow.active_run()
    except ModuleNotFoundError:
        active_run = None

    if active_run is not None:
        run_info = active_run.info
        artifact_uri = urlparse(run_info.artifact_uri)

        if artifact_uri.scheme == "file":
            return str(artifact_uri.path)

        return str(
            os.path.join(
                path,
                run_info.experiment_id,
                run_info.run_id,
            )
        )

    return str(
        os.path.join(
            path,
            datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S"),
        )
    )


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


def masked_pool(
    inputs: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
    method: str = "average",
    dim: int = 1,
    keepdim: bool = False,
) -> torch.Tensor:
    if mask is None:
        mask = cast(torch.BoolTensor, inputs.new_ones(inputs.size()).bool())

    if method == "average":
        return masked_mean(inputs, mask, dim=dim, keepdim=keepdim)
    if method == "max":
        return masked_max(inputs, mask, dim=dim, keepdim=keepdim)
    if method == "sum":
        return replace_masked_values(inputs, mask, 0.0).sum(dim=dim, keepdim=keepdim)

    raise ConfigurationError(f"Invalid pooling method: {method}")


def convert_to_toeplitz(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.dim() != 1:
        raise ValueError(f"Number of dimensions of inputs must be equal to 1 (actual={inputs.dim()}).")

    num_elements = inputs.size(0)
    if num_elements % 2 != 1:
        raise ValueError(f"Size of inputs must be a odd number. (actual={num_elements})")

    n = (num_elements + 1) // 2
    r = num_elements // 2

    output = torch.nn.functional.pad(inputs, (0, n))
    output = output.tile(n)
    output = output[:-n]
    output = output.reshape(n, -1)
    output = output[:, r:-r]

    return output
