import argparse
import json
import logging
import os
import tempfile
from functools import partial

import mlflow
import optuna
from allennlp.commands.subcommand import Subcommand
from optuna import Trial
from optuna.integration import AllenNLPExecutor
from xallennlp.utils import flatten_dict_for_mlflow_log, get_serialization_dir

logger = logging.getLogger(__name__)


@Subcommand.register("tune-with-mlflow")
class TuneWithMlflow(Subcommand):
    """
    This command is based on allennlp-optuna.
    """

    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Train the specified model on the specified dataset."""
        subparser = parser.add_parser(self.name, description=description, help="Optimize hyperparameter of a model.")

        subparser.add_argument(
            "param_path",
            type=str,
            help="path to parameter file describing the model to be trained",
        )

        subparser.add_argument(
            "hparam_path",
            type=str,
            help="path to hyperparameter file",
            default="hyper_params.json",
        )

        subparser.add_argument(
            "--optuna-param-path",
            type=str,
            help="path to Optuna config",
        )

        subparser.add_argument(
            "--serialization-dir",
            type=str,
            help="directory in which to save the model and its logs",
        )

        # ---- Optuna -----

        subparser.add_argument(
            "--load-if-exists",
            default=False,
            action="store_true",
            help="If specified, the creation of the study is skipped "
            "and load existing one when the study name is duplicated.",
        )

        subparser.add_argument(
            "--direction",
            type=str,
            choices=("minimize", "maximize"),
            default="minimize",
            help="Set direction of optimization to a new study. Set 'minimize' "
            "for minimization and 'maximize' for maximization.",
        )

        subparser.add_argument(
            "--n-trials",
            type=int,
            help="The number of trials. If this argument is not given, as many " "trials run as possible.",
            default=50,
        )

        subparser.add_argument(
            "--timeout",
            type=float,
            help="Stop study after the given number of second(s). If this argument"
            " is not given, as many trials run as possible.",
        )

        subparser.add_argument(
            "--study-name",
            default=None,
            help="The name of the study to start optimization on.",
        )

        subparser.add_argument(
            "--storage",
            type=str,
            help="The path to storage.",
            default="sqlite:///allennlp_optuna.db",
        )

        subparser.add_argument(
            "--metrics",
            type=str,
            help="The metrics you want to optimize.",
            default="best_validation_loss",
        )

        subparser.set_defaults(func=tune_from_args)
        return subparser


def tune_from_args(args: argparse.Namespace) -> None:
    config_file = args.param_path
    hparam_path = args.hparam_path
    optuna_param_path = args.optuna_param_path
    serialization_dir = args.serialization_dir
    include_package = args.include_package

    load_if_exists = args.load_if_exists
    direction = args.direction
    n_trials = args.n_trials
    timeout = args.timeout
    study_name = args.study_name
    storage = args.storage
    metrics = args.metrics

    def _objective(
        trial: Trial,
        hparam_path: str,
    ) -> float:

        for hparam in json.load(open(hparam_path)):
            attr_type = hparam["type"]
            suggest = getattr(trial, "suggest_{}".format(attr_type))
            suggest(**hparam["attributes"])

        with mlflow.start_run(nested=True):
            optuna_serialization_dir = get_serialization_dir(serialization_dir)

            os.makedirs(optuna_serialization_dir, exist_ok=True)

            mlflow.set_tag("optuna.study_name", trial.study.study_name)
            mlflow.set_tag("optuna.trial_number", trial.number)

            executor = AllenNLPExecutor(
                trial,
                config_file,
                optuna_serialization_dir,
                metrics=metrics,
                include_package=include_package,
                force=True,
            )
            score = executor.run()

        return score

    if optuna_param_path is not None and os.path.isfile(optuna_param_path):
        optuna_config = json.load(open(optuna_param_path))
    else:
        optuna_config = {}

    if "pruner" in optuna_config:
        pruner_class = getattr(optuna.pruners, optuna_config["pruner"]["type"])
        pruner = pruner_class(**optuna_config["pruner"].get("attributes", {}))
    else:
        pruner = None

    if "sampler" in optuna_config:
        sampler_class = getattr(optuna.samplers, optuna_config["sampler"]["type"])
        sampler = sampler_class(optuna_config["sampler"].get("attributes", {}))
    else:
        sampler = None

    with mlflow.start_run() as mlrun:
        study_name = study_name or mlrun.info.run_id

        mlflow.set_tag("optuna.study_name", study_name)
        mlflow.log_params(flatten_dict_for_mlflow_log(vars(args)))
        mlflow.log_artifact(config_file)
        mlflow.log_artifact(hparam_path)

        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=load_if_exists,
        )

        objective = partial(
            _objective,
            hparam_path=hparam_path,
        )

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_trial = study.best_trial

        mlflow.log_metric("best_value", best_trial.value)
        mlflow.log_metric("best_trial", best_trial.number)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "best_params.json")
            with open(filename, "w") as fp:
                json.dump(best_trial.params, fp)
            mlflow.log_artifact(filename)
