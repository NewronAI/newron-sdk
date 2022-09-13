import mlflow.lightgbm
from mlflow.utils.requirements_utils import _get_pinned_requirement
from newron.models import ModelSignature, ModelInputExample
import inspect
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
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
FLAVOR_NAME = "lightgbm"
DEFAULT_AWAIT_MAX_SLEEP_SECONDS = 5 * 60

_save_model = mlflow.lightgbm._save_model
_patch_metric_names = mlflow.lightgbm._patch_metric_names
_autolog_callback = mlflow.lightgbm._autolog_callback
_load_model = mlflow.lightgbm._load_model
_LGBModelWrapper = mlflow.lightgbm._LGBModelWrapper
_load_pyfunc = mlflow.lightgbm._load_pyfunc

def get_default_pip_requirements(include_cloudpickle=False):
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("lightgbm")]
    if include_cloudpickle:
        pip_deps.append(_get_pinned_requirement("cloudpickle"))
    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))

save_model = mlflow.lightgbm.save_model
log_model = mlflow.lightgbm.log_model
load_model = mlflow.lightgbm.load_model
