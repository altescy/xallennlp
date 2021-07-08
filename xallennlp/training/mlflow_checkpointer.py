import logging

import mlflow
from allennlp.training import Checkpointer, Trainer

logger = logging.getLogger(__name__)


@Checkpointer.register("mlflow")
class MLflowCheckpointer(Checkpointer):
    def save_checkpoint(
        self,
        trainer: Trainer,
    ) -> None:
        super().save_checkpoint(trainer)

        if mlflow.active_run() is None:
            logger.warning("MLflow active run not found." " Recommend to use 'train-with-mlflow' command.")

        mlflow.log_artifacts(self._serialization_dir)
