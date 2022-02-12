import datetime
from typing import cast

import numpy.testing
import torch
from xallennlp.utils import convert_to_toeplitz, flatten_dict_for_mlflow_log, masked_fft, masked_pool, str_to_timedelta


def test_flatten_dict_for_mlflow_log() -> None:
    data = {"x": {"y": "a"}, "z": 123}
    flattened_data = flatten_dict_for_mlflow_log(data)

    assert flattened_data["x.y"] == "a"
    assert flattened_data["z"] == 123


def test_str_to_timedelta() -> None:
    delta = datetime.timedelta(1, 2, 0)
    delta_str = str(delta)

    recon_delta = str_to_timedelta(delta_str)

    assert recon_delta == delta


def test_masked_fft() -> None:
    inputs = torch.rand(3, 4, 5)
    mask = cast(torch.BoolTensor, torch.ones(3, 4).bool())
    output = masked_fft(inputs, mask).numpy()
    desired = torch.fft.fft(inputs, dim=1).numpy()

    numpy.testing.assert_allclose(output, desired, rtol=1e-5)  # type: ignore


def test_masked_pool() -> None:
    inputs = torch.rand(3, 4, 5)
    mask = cast(torch.BoolTensor, torch.ones(3, 4, 1).bool())
    output = masked_pool(inputs, mask, method="max").numpy()
    desired = torch.max(inputs, dim=1)[0].numpy()

    numpy.testing.assert_equal(output, desired)  # type: ignore


def test_convert_to_toeplitz() -> None:
    inputs = torch.randn(9)
    output = convert_to_toeplitz(inputs)

    assert output.size() == (5, 5)
    assert all(output[i, j] == output[i - 1, j - 1] for i in range(1, 5) for j in range(1, 5))
