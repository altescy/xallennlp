import argparse
import tempfile
from pathlib import Path

import mlflow
import yaml
from allennlp.common.util import import_module_and_submodules
from xallennlp.commands.train_with_mlflow import TrainWithMLflow, train_model_from_args

import_module_and_submodules("xallennlp")


class TestTrainWithMLflow:
    def setup(self) -> None:
        self.parser = argparse.ArgumentParser(description="Testing")
        subparsers = self.parser.add_subparsers(title="Commands", metavar="")
        TrainWithMLflow().add_subparser(subparsers)

    def test_train_with_mlflow_from_args(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)

            mlflow.set_tracking_uri(f"file://{tempdir}")
            mlflow.set_experiment("test")

            args = self.parser.parse_args(
                [
                    "train-with-mlflow",
                    "configs/basic_classifier.jsonnet",
                ]
            )
            args.include_package = []
            train_model_from_args(args)

            with open(tempdir / "0" / "meta.yaml") as f:
                meta = yaml.safe_load(f)

            assert meta["name"] == "test"
