import logging
import os
import tempfile
from typing import Any, Dict

from allennlp.training.trainer import EpochCallback, GradientDescentTrainer
import mlflow

from xallennlp.utils import flatten_dict_for_mlflow_log, str_to_timedelta

logger = logging.getLogger(__name__)


@EpochCallback.register("mlflow_metrics")
class MLflowMetrics(EpochCallback):
    def __call__(
            self,
            trainer: GradientDescentTrainer,
            metrics: Dict[str, Any],
            epoch: int,
            is_master: bool,
    ) -> None:
        if mlflow.active_run() is None:
            logger.warning("MLflow active run not found."
                           " Recommend to use 'train-with-mlflow' command.")

        if "trainig_duration" in metrics:
            trainig_duration = str_to_timedelta(metrics["trainig_duration"])
            metrics["trainig_duration"] = trainig_duration.total_seconds()

        flattened_metrics = flatten_dict_for_mlflow_log(metrics)

        for key, value in flattened_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
            else:
                log_nonnumerical_metric(key, value, epoch)


def log_nonnumerical_metric(key: str, value: Any, epoch: int):
    with tempfile.TemporaryDirectory() as tempdir:
        temppath = os.path.join(tempdir, key)

        with open(temppath, "w") as f:
            f.write(repr(value))

        mlflow.log_artifact(temppath, f"metrics/epoch_{epoch}/{key}")
