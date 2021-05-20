import logging
from typing import Union

import mlflow
from allennlp.training import Checkpointer, Trainer

logger = logging.getLogger(__name__)


@Checkpointer.register("mlflow")
class MLflowCheckpointer(Checkpointer):
    def save_checkpoint(
        self,
        epoch: Union[int, str],
        trainer: Trainer,
        is_best_so_far: bool = False,
        save_model_only=False,
    ) -> None:
        if self._serialization_dir is None:
            return

        super().save_checkpoint(
            epoch,
            trainer,
            is_best_so_far,
            save_model_only,
        )

        if mlflow.active_run() is None:
            logger.warning("MLflow active run not found." " Recommend to use 'train-with-mlflow' command.")

        mlflow.log_artifacts(self._serialization_dir)
