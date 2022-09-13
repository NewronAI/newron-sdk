"""
The ``newron.tensorflow`` module provides an API for logging and loading TensorFlow models.
"""

import mlflow.tensorflow
from packaging.version import Version
from newron.models import ModelSignature, ModelInputExample
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from collections import namedtuple
from threading import RLock
import inspect
FLAVOR_NAME = "tensorflow"

DEFAULT_AWAIT_MAX_SLEEP_SECONDS = 5 * 60

_TensorBoardLogDir = namedtuple("_TensorBoardLogDir", ["location", "is_temp"])

_load_pyfunc = mlflow.tensorflow._load_pyfunc
_validate_saved_model = mlflow.tensorflow._validate_saved_model
_load_tensorflow_saved_model = mlflow.tensorflow._load_tensorflow_saved_model
_parse_flavor_configuration = mlflow.tensorflow._parse_flavor_configuration
_TF2Wrapper = mlflow.tensorflow._TF2Wrapper
_assoc_list_to_map = mlflow.tensorflow._assoc_list_to_map
_flush_queue = mlflow.tensorflow._flush_queue
_add_to_queue = mlflow.tensorflow._add_to_queue
_add_to_queue = mlflow.tensorflow._add_to_queue
_get_tensorboard_callback = mlflow.tensorflow._get_tensorboard_callback
_setup_callbacks = mlflow.tensorflow._setup_callbacks
_metric_queue_lock = RLock()
_metric_queue = []


save_model = mlflow.tensorflow.save_model
log_model = mlflow.tensorflow.log_model
load_model = mlflow.tensorflow.load_model
get_default_pip_requirements = mlflow.tensorflow.get_default_pip_requirements
get_default_conda_env = mlflow.tensorflow.get_default_conda_env
autolog = mlflow.tensorflow.autolog
