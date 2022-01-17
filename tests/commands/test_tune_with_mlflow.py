import argparse
from pathlib import Path

import mlflow
import yaml
from allennlp.common.util import import_module_and_submodules
from xallennlp.commands.tune_with_mlflow import TuneWithMlflow, tune_from_args

import_module_and_submodules("xallennlp")


class TestTuneWithMlflow:
    def setup(self) -> None:
        self.parser = argparse.ArgumentParser(description="Testing")
        subparsers = self.parser.add_subparsers(title="Commands", metavar="")
        TuneWithMlflow().add_subparser(subparsers)

    def test_train_with_mlflow_from_args(self, tmp_path: Path) -> None:
        mlflow.set_tracking_uri(f"file://{tmp_path}")
        mlflow.set_experiment("test")

        args = self.parser.parse_args(
            [
                "tune-with-mlflow",
                "tests/fixtures/configs/basic_classifier_experiment.jsonnet",
                "tests/fixtures/configs/basic_classifier_hparams.json",
                "--n-trials",
                "5",
            ]
        )
        args.include_package = []
        tune_from_args(args)

        with open(tmp_path / "0" / "meta.yaml") as f:
            meta = yaml.safe_load(f)

        assert meta["name"] == "test"
