from typing import Dict, TypeVar
import datetime
import re

import flatten_dict

REGEX_TIMEDELTA = re.compile(
    r"(?:(\d+) days?, )?(\d+):(\d+):(\d+)(?:\.(\d+))?")

T = TypeVar("T")


def flatten_dict_for_mlflow_log(data: Dict[str, T]) -> Dict[str, T]:
    flattened_data: Dict[str, T] = {
        str(key): value
        for key, value in flatten_dict.flatten(data, reducer="dot").items()
    }
    return flattened_data


def str_to_timedelta(delta_str: str) -> datetime.timedelta:
    match = re.match(REGEX_TIMEDELTA, delta_str)

    if not match:
        raise ValueError(f"invalid timedelta format: {delta_str}")

    nums = tuple(int(x or 0) for x in match.groups())
    days, hours, minutes, seconds, micros = nums

    seconds += 3600 * hours + 60 * minutes

    return datetime.timedelta(days=days, seconds=seconds, microseconds=micros)
