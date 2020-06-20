import argparse
import os
from urllib.parse import urlparse

from allennlp.commands.subcommand import Subcommand
from allennlp.commands.train import train_model
from allennlp.common import Params
import mlflow
from overrides import overrides

from xallennlp.utils import flatten_dict_for_mlflow_log


@Subcommand.register("train-with-mlflow")
class TrainWithMLflow(Subcommand):
    @overrides
    def add_subparser(
            self,
            parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Train the model with MLflow Tracking."""
        subparser = parser.add_parser(self.name,
                                      description=description,
                                      help="Train a model with MLflow.")

        subparser.add_argument(
            "param_path",
            type=str,
            help="path to parameter file describing the model to be trained")

        subparser.add_argument(
            "-e",
            "--experiment",
            type=str,
            default="",
            help="name of mlflow experiment",
        )

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            type=str,
            default="out",
            help="directory in which to save the model and its logs",
        )

        subparser.add_argument(
            "-r",
            "--recover",
            action="store_true",
            default=False,
            help="recover training from the state in serialization_dir",
        )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=
            "a JSON structure used to override the experiment configuration",
        )

        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help=
            "outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.add_argument(
            "--node-rank",
            type=int,
            default=0,
            help="rank of this node in the distributed setup")

        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help=
            "do not train a model, but create a vocabulary, show dataset statistics and "
            "other training information",
        )

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    params = Params.from_file(args.param_path, args.overrides)

    params_dict = params.as_flat_dict()
    params_dict.update({"args": vars(args)})
    flattened_params = flatten_dict_for_mlflow_log(params_dict)

    if args.experiment:
        mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_params(flattened_params)

        serialization_dir = get_serialization_dir(args)

        train_model(
            params=params,
            serialization_dir=serialization_dir,
            file_friendly_logging=args.file_friendly_logging,
            recover=args.recover,
            force=args.force,
            node_rank=args.node_rank,
            include_package=args.include_package,
            dry_run=args.dry_run,
        )


def get_serialization_dir(args: argparse.Namespace) -> str:
    run_info = mlflow.active_run().info
    artifact_uri = urlparse(run_info.artifact_uri)

    if args.recover:
        return str(args.serialization_dir)

    if artifact_uri.scheme == "file":
        return str(artifact_uri.path)

    return str(
        os.path.join(
            args.serialization_dir,
            run_info.experiment_id,
            run_info.run_id,
        ))
