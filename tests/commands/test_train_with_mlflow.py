import argparse
import os
from pathlib import Path
import tempfile

from allennlp.common.util import import_module_and_submodules
from allennlp.commands import create_parser

from xallennlp.commands.train_with_mlflow import TrainWithMLflow, train_model_from_args

import_module_and_submodules("xallennlp")


class TestTrainWithMLflow:
    def setup(self):
        self.parser = create_parser()

    def test_train_with_mlflow_from_args(self):
        config_dir = Path.cwd() / "configs"
        data_dir = Path.cwd() / "data"

        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            os.chdir(tempdir)

            (tempdir / "data").symlink_to(data_dir)
            (tempdir / "configs").symlink_to(config_dir)

            args = self.parser.parse_args([
                "train-with-mlflow",
                "configs/basic_classifier.jsonnet",
            ])
            train_model_from_args(args)

            assert (tempdir / "mlruns" / "0").exists()
