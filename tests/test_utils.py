import datetime

from xallennlp.utils import flatten_dict_for_mlflow_log, str_to_timedelta


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
