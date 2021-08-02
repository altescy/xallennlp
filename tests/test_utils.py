import datetime
from typing import cast

import numpy.testing
import torch
from xallennlp.utils import flatten_dict_for_mlflow_log, masked_fourier_transform, str_to_timedelta


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


def test_masked_fourier_transform() -> None:
    inputs = torch.rand(3, 4, 5)
    mask = cast(torch.BoolTensor, torch.ones(3, 4).bool())
    output = masked_fourier_transform(inputs, mask).numpy()
    desired = torch.fft.fft(inputs, dim=1).numpy()

    numpy.testing.assert_allclose(output, desired, rtol=1e-5)  # type: ignore
