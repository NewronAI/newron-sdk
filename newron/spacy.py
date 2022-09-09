import mlflow.spacy
import inspect
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
from newron.models import ModelSignature, ModelInputExample
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
FLAVOR_NAME = "spacy"
DEFAULT_AWAIT_MAX_SLEEP_SECONDS = 5 * 60

_load_model = mlflow.spacy._load_model
_SpacyModelWrapper = mlflow.spacy._SpacyModelWrapper
_load_pyfunc = mlflow.spacy._load_pyfunc


def get_default_pip_requirements():
    """
    
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("spacy")]


def get_default_conda_env():
    """
    
    Returns:
        The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())

def save_model(
    spacy_model,
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
    Save a spaCy model to a path on the local file system.

    Args:
        spacy_model: spaCy model to be saved.
        path: Local path where the model is to be saved.
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
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
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
    mlflow.spacy.save_model(**values)

def log_model(
    spacy_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Log a spaCy model as an MLflow artifact for the current run.

    Args:
        spacy_model: spaCy model to be saved.
        artifact_path: Run-relative artifact path.
        conda_env: conda_env
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
        registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:
                      .. code-block:: python
                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
        input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
        pip_requirements: pip_requirements
        extra_pip_requirements: extra_pip_requirements
        kwargs: kwargs to pass to ``spacy.save_model`` method.
        
    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    del values["kwargs"]
    return mlflow.spacy.log_model(**values)

def load_model(model_uri, dst_path=None):
    """
    Load a spaCy model from a local file (if ``run_id`` is ``None``) or a run.
    Args:    
        model_uri: The location, in URI format, of the MLflow model. For example:
            * ``Users/me/path/to/local/model``
            * ``relative/path/to/local/model``
            * ``s3://my_bucket/path/to/model``
            * ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            * ``models:/<model_name>/<model_version>``
            * ``models:/<model_name>/<stage>``
                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`.
        dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.
                     
    Returns:
        A spaCy loaded model
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    return mlflow.spacy.load_model(**values)