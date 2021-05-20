import datetime
import re
from typing import Any, Dict

import flatten_dict

REGEX_TIMEDELTA = re.compile(r"(?:(\d+) days?, )?(\d+):(\d+):(\d+)(?:\.(\d+))?")


def flatten_dict_for_mlflow_log(data: Dict[str, Any]) -> Dict[str, Any]:
    flattened_data: Dict[str, Any] = flatten_dict.flatten(data, reducer="dot")
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
