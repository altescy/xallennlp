from typing import Any, Dict

import flatten_dict


def flatten_dict_for_mlflow_log(data: Dict[str, Any]) -> Dict[str, Any]:
    flattened_data = flatten_dict.flatten(data, reducer="dot")
    flattened_data = {str(key): value for key, value in flattened_data.items()}
    return flattened_data
