import mlflow.sklearn
import inspect
from mlflow.utils.requirements_utils import _get_pinned_requirement
from newron.models import ModelSignature, ModelInputExample
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,)
import logging
_apis_autologging_disabled = [
    "cross_validate",
    "cross_val_predict",
    "cross_val_score",
    "learning_curve",
    "permutation_test_score",
    "validation_curve",
]

FLAVOR_NAME = "sklearn"
DEFAULT_AWAIT_MAX_SLEEP_SECONDS = 5 * 60
SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"

_logger = logging.getLogger(__name__)
_gen_estimators_to_patch = mlflow.sklearn._gen_estimators_to_patch
_load_model_from_local_file = mlflow.sklearn._load_model_from_local_file
_load_pyfunc = mlflow.sklearn._load_pyfunc
_SklearnCustomModelPicklingError = mlflow.sklearn._SklearnCustomModelPicklingError
_dump_model = mlflow.sklearn._dump_model
_save_model = mlflow.sklearn._save_model
_AutologgingMetricsManager = mlflow.sklearn._AutologgingMetricsManager
_autolog = mlflow.sklearn._autolog
_eval_and_log_metrics_impl = mlflow.sklearn._eval_and_log_metrics_impl 
_get_metric_name_list = mlflow.sklearn._get_metric_name_list
log_model = mlflow.sklearn.log_model
eval_and_log_metrics = mlflow.sklearn.eval_and_log_metrics
save_model = mlflow.sklearn.save_model
autolog = mlflow.sklearn.autolog

def get_default_pip_requirements(include_cloudpickle=False):
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("scikit-learn", module="sklearn")]
    if include_cloudpickle:
        pip_deps += [_get_pinned_requirement("cloudpickle")]

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))




