import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, Optional

import mlflow
from allennlp.training.callbacks import TrainerCallback

if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

from xallennlp.utils import flatten_dict_for_mlflow_log, str_to_timedelta

logger = logging.getLogger(__name__)


@TrainerCallback.register("mlflow")
class MLflowMetrics(TrainerCallback):
    def __init__(
        self,
        serialization_dir: str,
    ) -> None:
        super().__init__(serialization_dir=serialization_dir)

    def on_start(
        self,
        trainer: "GradientDescentTrainer",
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        with open(os.path.join(self.serialization_dir, "config.json")) as json_file:
            config = flatten_dict_for_mlflow_log(json.load(json_file))

        mlflow.log_params(config)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        if "training_duration" in metrics:
            trainig_duration = str_to_timedelta(metrics["training_duration"])
            metrics["training_duration"] = trainig_duration.total_seconds()

        flattened_metrics = flatten_dict_for_mlflow_log(metrics)

        step = trainer._total_batches_completed

        for key, value in flattened_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value), step=step)
            else:
                log_nonnumerical_metric(key, value, step)

    def on_end(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        mlflow.log_artifacts(self.serialization_dir)


def log_nonnumerical_metric(key: str, value: Any, step: int) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        temppath = os.path.join(tempdir, key)

        with open(temppath, "w") as f:
            f.write(repr(value))

        mlflow.log_artifact(temppath, f"metrics/step_{step}")
