import json
import os
import re
import sys
import logging

import click
from click import UsageError
from datetime import timedelta

import mlflow.db
import mlflow.experiments
import mlflow.deployments.cli
import mlflow.pipelines.cli
from mlflow import projects
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
import mlflow.runs
import mlflow.store.artifact.cli
from mlflow import version
from mlflow import tracking
from mlflow.store.tracking import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH, DEFAULT_ARTIFACTS_URI
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking import _get_store
from mlflow.utils import cli_args
from mlflow.utils.logging_utils import eprint
from mlflow.utils.process import ShellCommandException
from mlflow.utils.server_cli_utils import (
    resolve_default_artifact_root,
    artifacts_only_config_validation,
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from newron.exceptions import NewronException

_logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=version.VERSION)
def cli():
    pass

@cli.command()
@click.argument("uri")
@click.option(
    "--entry-point",
    "-e",
    metavar="NAME",
    default="main",
    help="Entry point within project. [default: main]. If the entry point is not found, "
    "attempts to run the project file with the specified name as a script, "
    "using 'python' to run .py files and the default shell (specified by "
    "environment variable $SHELL) to run .sh files",
)
@click.option(
    "--version",
    "-v",
    metavar="VERSION",
    help="Version of the project to run, as a Git commit reference for Git projects.",
)
@click.option(
    "--param-list",
    "-P",
    metavar="NAME=VALUE",
    multiple=True,
    help="A parameter for the run, of the form -P name=value. Provided parameters that "
    "are not in the list of parameters for an entry point will be passed to the "
    "corresponding entry point as command-line arguments in the form `--name value`",
)
@click.option(
    "--docker-args",
    "-A",
    metavar="NAME=VALUE",
    multiple=True,
    help="A `docker run` argument or flag, of the form -A name=value (e.g. -A gpus=all) "
    "or -A name (e.g. -A t). The argument will then be passed as "
    "`docker run --name value` or `docker run --name` respectively. ",
)
@click.option(
    "--experiment-name",
    envvar=tracking._EXPERIMENT_NAME_ENV_VAR,
    help="Name of the experiment under which to launch the run. If not "
    "specified, 'experiment-id' option will be used to launch run.",
)
@click.option(
    "--experiment-id",
    envvar=tracking._EXPERIMENT_ID_ENV_VAR,
    type=click.STRING,
    help="ID of the experiment under which to launch the run.",
)
# TODO: Add tracking server argument once we have it working.
@click.option(
    "--backend",
    "-b",
    metavar="BACKEND",
    default="local",
    help="Execution backend to use for run. Supported values: 'local', 'databricks', "
    "kubernetes (experimental). Defaults to 'local'. If running against "
    "Databricks, will run against a Databricks workspace determined as follows: "
    "if a Databricks tracking URI of the form 'databricks://profile' has been set "
    "(e.g. by setting the MLFLOW_TRACKING_URI environment variable), will run "
    "against the workspace specified by <profile>. Otherwise, runs against the "
    "workspace specified by the default Databricks CLI profile. See "
    "https://github.com/databricks/databricks-cli for more info on configuring a "
    "Databricks CLI profile.",
)
@click.option(
    "--backend-config",
    "-c",
    metavar="FILE",
    help="Path to JSON file (must end in '.json') or JSON string which will be passed "
    "as config to the backend. The exact content which should be "
    "provided is different for each execution backend and is documented "
    "at https://www.mlflow.org/docs/latest/projects.html.",
)
@cli_args.NO_CONDA
@cli_args.ENV_MANAGER
@click.option(
    "--storage-dir",
    envvar="MLFLOW_TMP_DIR",
    help="Only valid when ``backend`` is local. "
    "MLflow downloads artifacts from distributed URIs passed to parameters of "
    "type 'path' to subdirectories of storage_dir.",
)
@click.option(
    "--run-id",
    metavar="RUN_ID",
    help="If specified, the given run ID will be used instead of creating a new run. "
    "Note: this argument is used internally by the MLflow project APIs "
    "and should not be specified.",
)
@click.option(
    "--run-name",
    metavar="RUN_NAME",
    help="The name to give the MLflow Run associated with the project execution. If not specified, "
    "the MLflow Run name is left unset.",
)
def run(
    uri,
    entry_point,
    version,
    param_list,
    docker_args,
    experiment_name,
    experiment_id,
    backend,
    backend_config,
    no_conda,  # pylint: disable=unused-argument
    env_manager,
    storage_dir,
    run_id,
    run_name,
):
    if experiment_id is not None and experiment_name is not None:
        eprint("Specify only one of 'experiment-name' or 'experiment-id' options.")
        sys.exit(1)
    
    if backend_config is not None and os.path.splitext(backend_config)[-1] != ".json":
        try:
            backend_config = json.loads(backend_config)
        except ValueError as e:
            eprint("Invalid backend config JSON. Parse error: %s" % e)
            raise
    if backend == "kubernetes":
        if backend_config is None:
            eprint("Specify 'backend_config' when using kubernetes mode.")
            sys.exit(1)
    try:
        projects.run(
            uri,
            entry_point,
            version,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            parameters=param_dict,
            docker_args=args_dict,
            backend=backend,
            backend_config=backend_config,
            env_manager=env_manager,
            storage_dir=storage_dir,
            synchronous=backend in ("local", "kubernetes") or backend is None,
            run_id=run_id,
            run_name=run_name,
        )
    except projects.ExecutionException as e:
        _logger.error("=== %s ===", e)
        sys.exit(1)