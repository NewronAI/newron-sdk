# pylint: disable=wrong-import-position
"""
The ``newron`` module provides a high-level API for starting and managing Newron runs.
For example:

.. code:: python

    import newron

    newron.init()

    newron.start_run()
    newron.log_param("my", "param")
    newron.log_metric("score", 100)
    newron.end_run()

You can also use the context manager syntax like this:

.. code:: python

    import newron

    newron.init()
    
    with newron.start_run() as run:
        newron.log_param("my", "param")
        newron.log_metric("score", 100)

which automatically terminates the run at the end of the ``with`` block.
"""
from newron.version import VERSION as __version__  # pylint: disable=unused-import
from mlflow.utils.logging_utils import _configure_mlflow_loggers


# Filter annoying Cython warnings that serve no good purpose, and so before
# importing other modules.
# See: https://github.com/numpy/numpy/pull/432/commits/170ed4e33d6196d7
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from newron import tracking
from newron import models
from newron import auth
from mlflow import projects
import newron.artifacts

# model flavors
_model_flavors_supported = []
    # pylint: disable=unused-import
from newron import catboost
from newron import fastai
from newron import gluon
from newron import keras
from newron import lightgbm
from mlflow import pyfunc
from newron import pytorch
from newron import sklearn
from newron import spacy
from newron import tensorflow
from mlflow import xgboost
from mlflow import shap
from mlflow import pyspark
from mlflow import paddle
from newron import prophet
from newron import diviner

_model_flavors_supported = [
    "catboost",
    "fastai",
    "gluon",
    "keras",
    "lightgbm",
    "pyfunc",
    "pytorch",
    "sklearn",
    "spacy",
    "spark",
    "tensorflow",
    "xgboost",
    "shap",
    "paddle",
    "prophet",
    "diviner",
]

_configure_mlflow_loggers(root_module_name=__name__)

# TODO: Comment out this block when we deprecate support for python 3.7.
# _major = 3
# _minor = 7
# _deprecated_version = (_major, _minor)
# _min_supported_version = (_major, _minor + 1)

# if sys.version_info[:2] == _deprecated_version:
#     warnings.warn(
#         "MLflow support for Python {dep_ver} is deprecated and will be dropped in "
#         "an upcoming release. At that point, existing Python {dep_ver} workflows "
#         "that use MLflow will continue to work without modification, but Python {dep_ver} "
#         "users will no longer get access to the latest MLflow features and bugfixes. "
#         "We recommend that you upgrade to Python {min_ver} or newer.".format(
#             dep_ver=".".join(map(str, _deprecated_version)),
#             min_ver=".".join(map(str, _min_supported_version)),
#         ),
#         FutureWarning,
#         stacklevel=2,
#     )


#search_experiments = mlflow.tracking.search_experiments 
#Marked as experimental

get_tracking_uri = tracking.get_tracking_uri
get_registry_uri = tracking.get_registry_uri
create_experiment = tracking.create_experiment
set_experiment = tracking.set_experiment
log_params = tracking.log_params
log_metrics = tracking.log_metrics
set_experiment_tags = tracking.set_experiment_tags
set_experiment_tag = tracking.set_experiment_tag
set_tags = tracking.set_tags
delete_experiment = tracking.delete_experiment  
delete_run = tracking.delete_run
register_model = tracking.register_model
autolog = tracking.autolog
evaluate = models.evaluate
last_active_run = tracking.last_active_run
NewronClient = tracking.NewronClient
ActiveRun = tracking.ActiveRun
log_param = tracking.log_param
log_metric = tracking.log_metric
set_tag = tracking.set_tag
delete_tag = tracking.delete_tag
log_artifacts = tracking.log_artifacts
log_artifact = tracking.log_artifact
log_text = tracking.log_text
log_dict = tracking.log_dict
log_image = tracking.log_image
log_figure = tracking.log_figure
active_run = tracking.active_run
get_run = tracking.get_run
start_run = tracking.start_run
end_run = tracking.end_run
search_runs = tracking.search_runs
list_run_infos = tracking.list_run_infos
get_artifact_uri = tracking.get_artifact_uri
set_tracking_uri = tracking.set_tracking_uri
set_registry_uri = tracking.set_registry_uri
get_experiment = tracking.get_experiment
get_experiment_by_name = tracking.get_experiment_by_name
list_experiments = tracking.list_experiments
run = projects.run
init = tracking.init


__all__ = [
    "ActiveRun",
    "log_param",
    "log_params",
    "log_metric",
    "log_metrics",
    "set_experiment_tags",
    "set_experiment_tag",
    "set_tag",
    "set_tags",
    "delete_tag",
    "log_artifacts",
    "log_artifact",
    "log_text",
    "log_dict",
    "log_figure",
    "log_image",
    "active_run",
    "start_run",
    "end_run",
    "search_runs",
    "get_artifact_uri",
    "get_tracking_uri",
    "set_tracking_uri",
    "get_experiment",
    "get_experiment_by_name",
    "list_experiments",
    "search_experiments",
    "create_experiment",
    "set_experiment",
    "delete_experiment",
    "get_run",
    "delete_run",
    "run",
    "register_model",
    "get_registry_uri",
    "set_registry_uri",
    "list_run_infos",
    "autolog",
    "evaluate",
    "last_active_run",
    "NewronClient",
] + _model_flavors_supported
