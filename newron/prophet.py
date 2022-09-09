import mlflow.prophet
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
FLAVOR_NAME = "prophet"
DEFAULT_AWAIT_MAX_SLEEP_SECONDS = 5 * 60

_load_model = mlflow.prophet._load_model
_ProphetModelWrapper = mlflow.prophet._ProphetModelWrapper
_load_pyfunc = mlflow.prophet._load_pyfunc
_save_model = mlflow.prophet._save_model
#autolog = mlflow.prophet.autolog

def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at a minimum, contains these requirements.
    """
    # Note: Prophet's whl build process will fail due to missing dependencies, defaulting
    # to setup.py installation process.
    # If a pystan installation error occurs, ensure gcc>=8 is installed in your environment.
    # See: https://gcc.gnu.org/install/
    return [_get_pinned_requirement("prophet")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())

def save_model(
    pr_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Save a Prophet model to a path on the local file system.
    
    Args:
        pr_model: Prophet model (an instance of Prophet() forecaster that has been fit
                     on a temporal series.
        path: Local path where the serialized model (as JSON) is to be saved.
        conda_env: conda_env
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:
                      .. code-block:: python
                        from mlflow.models.signature import infer_signature
                        model = Prophet().fit(df)
                        train = model.history
                        predictions = model.predict(model.make_future_dataframe(30))
                        signature = infer_signature(train, predictions)
        input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
        pip_requirements: pip_requirements
        extra_pip_requirements: extra_pip_requirements
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    mlflow.prophet.save_model(**values)

def log_model(
    pr_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Log a Prophet model as an MLflow artifact for the current run.
        pr_model: Prophet model to be saved.
        artifact_path: Run-relative artifact path.
        conda_env: conda_env
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
        registered_model_name: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output
                      :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred
                      <mlflow.models.infer_signature>` from datasets with valid model input
                      (e.g. the training dataset with target column omitted) and valid model
                      output (e.g. model predictions generated on the training dataset),
                      for example:
                      .. code-block:: python
                        from mlflow.models.signature import infer_signature
                        model = Prophet().fit(df)
                        train = model.history
                        predictions = model.predict(model.make_future_dataframe(30))
                        signature = infer_signature(train, predictions)
        input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to
                          feed the model. The given example will be converted to a
                          Pandas DataFrame and then serialized to json using the
                          Pandas split-oriented format. Bytes are base64-encoded.
        await_registration_for: Number of seconds to wait for the model version
                        to finish being created and is in ``READY`` status.
                        By default, the function waits for five minutes.
                        Specify 0 or None to skip waiting.
        pip_requirements: pip_requirements
        extra_pip_requirements: extra_pip_requirements
    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    return mlflow.prophet.log_model(**values)

def load_model(model_uri, dst_path=None):
    """
    Load a Prophet model from a local file or a run.
    Args:
        model_uri: The location, in URI format, of the MLflow model. 
                    For example:
            * ``Users/me/path/to/local/model``
            * ``relative/path/to/local/model``
            * ``s3://my_bucket/path/to/model``
            * ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                        
            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#artifact-locations>`.
        dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.
    Returns:
        A Prophet model instance
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    return mlflow.prophet.load_model(**values)
