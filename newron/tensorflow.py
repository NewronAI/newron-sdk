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


def get_default_pip_requirements():
    """
    Returns: 
            A list of default pip requirements for Models produced by Tensorflow.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    import tensorflow as tf

    pip_deps = [_get_pinned_requirement("tensorflow")]

    # tensorflow >= 2.6.0 requires keras:
    # https://github.com/tensorflow/tensorflow/blob/v2.6.0/tensorflow/tools/pip_package/setup.py#L106
    # To prevent a different version of keras from being installed by tensorflow when creating
    # a serving environment, add a pinned requirement for keras
    if Version(tf.__version__) >= Version("2.6.0"):
        pip_deps.append(_get_pinned_requirement("keras"))

    return pip_deps

def get_default_conda_env():
    """
    Returns: 
            The default Conda environment for Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())

def log_model(
    tf_saved_model_dir,
    tf_meta_graph_tags,
    tf_signature_def_key,
    artifact_path,
    conda_env=None,
    code_paths=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    registered_model_name=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Log a *serialized* collection of TensorFlow graphs and variables as an Newron model
    for the current run. This method operates on TensorFlow variables and graphs that have been
    serialized in TensorFlow's ``SavedModel`` format. For more information about ``SavedModel``
    format, see the TensorFlow documentation:
    [Tensorflow: Save and Restore Models](https://www.tensorflow.org/guide/saved_model#save_and_restore_models.)
    This method saves a model with both ``python_function`` and ``tensorflow`` flavors.
    If loaded back using the ``python_function`` flavor, the model can be used to predict on
    pandas DataFrames, producing a pandas DataFrame whose output columns correspond to the
    TensorFlow model's outputs. The python_function model will flatten outputs that are length-one,
    one-dimensional tensors of a single scalar value (e.g.
    ``{"predictions": [[1.0], [2.0], [3.0]]}``) into the scalar values (e.g.
    ``{"predictions": [1, 2, 3]}``), so that the resulting output column is a column of scalars
    rather than lists of length one. All other model output types are included as-is in the output
    DataFrame.
    Note that this method should not be used to log a ``tf.keras`` model. Use
    :py:func:`mlflow.keras.log_model` instead.

    Args:
        tf_saved_model_dir: Path to the directory containing serialized TensorFlow variables and
                               graphs in ``SavedModel`` format.
        tf_meta_graph_tags: A list of tags identifying the model's metagraph within the
                               serialized ``SavedModel`` object. For more information, see the
                               ``tags`` parameter of the
                               ``tf.saved_model.builder.SavedModelBuilder`` method.
        tf_signature_def_key: A string identifying the input/output signature associated with the
                                 model. This is a key within the serialized ``SavedModel`` signature
                                 definition mapping. For more information, see the
                                 ``signature_def_map`` parameter of the
                                 ``tf.saved_model.builder.SavedModelBuilder`` method.
        artifact_path: The run-relative path to which to log model artifacts.
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
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
        await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: pip_requirements
        extra_pip_requirements: extra_pip_requirements

    Returns: 
            A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    return mlflow.tensorflow.log_model(**values)
    
def save_model(
    tf_saved_model_dir,
    tf_meta_graph_tags,
    tf_signature_def_key,
    path,
    mlflow_model=None,
    conda_env=None,
    code_paths=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Save a *serialized* collection of TensorFlow graphs and variables as an MLflow model
    to a local path. This method operates on TensorFlow variables and graphs that have been
    serialized in TensorFlow's ``SavedModel`` format. For more information about ``SavedModel``
    format, see the TensorFlow documentation:
    [Tensorflow: Save and Restore Models](https://www.tensorflow.org/guide/saved_model#save_and_restore_models.)

    Args:[source][source][source]
        tf_saved_model_dir: Path to the directory containing serialized TensorFlow variables and
                               graphs in ``SavedModel`` format.
        tf_meta_graph_tags: A list of tags identifying the model's metagraph within the
                               serialized ``SavedModel`` object. For more information, see the
                               ``tags`` parameter of the
                               ``tf.saved_model.builder.savedmodelbuilder`` method.
        tf_signature_def_key: A string identifying the input/output signature associated with the
                                 model. This is a key within the serialized ``savedmodel``
                                 signature definition mapping. For more information, see the
                                 ``signature_def_map`` parameter of the
                                 ``tf.saved_model.builder.savedmodelbuilder`` method.
        path: Local path where the MLflow model is to be saved.
        mlflow_model: MLflow model configuration to which to add the ``tensorflow`` flavor.
        conda_env: conda_env
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:
                      .. code-block:: python
                        from newron.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
        input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
        pip_requirements: pip_requirements
        extra_pip_requirements: extra_pip_requirements
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    mlflow.tensorflow.save_model(**values)

def load_model(model_uri, dst_path=None):
    """
    Load an Newron model that contains the TensorFlow flavor from the specified path.
    ```python
        import newron.tensorflow
        import tensorflow as tf
        tf_graph = tf.Graph()
        tf_sess = tf.Session(graph=tf_graph)
        with tf_graph.as_default():
            signature_definition = newron.tensorflow.load_model(model_uri="model_uri",
                                    tf_sess=tf_sess)
            input_tensors = [tf_graph.get_tensor_by_name(input_signature.name)
                                for _, input_signature in signature_definition.inputs.items()]
            output_tensors = [tf_graph.get_tensor_by_name(output_signature.name)
                                for _, output_signature in signature_definition.outputs.items()]
    ```
    Args:
        model_uri: The location, in URI format, of the Newron model. For example:
            * ``Users/me/path/to/local/model``
            * ``relative/path/to/local/model``
            * ``s3://my_bucket/path/to/model``
            * ``models:/<model_name>/<model_version>``
            * ``models:/<model_name>/<stage>``
                    For more information about supported URI schemes, see
                    `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                    artifact-locations>`.
        dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.
    Returns: 
        A callable graph (tf.function) that takes inputs and returns inferences.   
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    return mlflow.tensorflow.load_model(**values)

def autolog(
    every_n_iter=1,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    log_input_examples=False,
    log_model_signatures=False,
):  # pylint: disable=unused-argument
    # pylint: disable=E0611
    """
    Enables automatic logging from TensorFlow to Newron.
    Note that autologging for ``tf.keras`` is handled by :py:func:`mlflow.tensorflow.autolog`,
    not :py:func:`mlflow.keras.autolog`.
    As an example, try running the
    `TensorFlow examples <https://github.com/mlflow/mlflow/tree/master/examples/tensorflow>`.
    For each TensorFlow module, autologging captures the following information:
    **tf.keras**
     - **Metrics** and **Parameters**
      - Training loss; validation loss; user-specified metrics
      - ``fit()`` or ``fit_generator()`` parameters; optimizer name; learning rate; epsilon
     - **Artifacts**
      - Model summary on training start
      - `MLflow Model <https://mlflow.org/docs/latest/models.html>` (Keras model)
      - TensorBoard logs on training end
    **tf.keras.callbacks.EarlyStopping**
     - **Metrics** and **Parameters**
      - Metrics from the ``EarlyStopping`` callbacks: ``stopped_epoch``, ``restored_epoch``,``restore_best_weight``, etc
      - ``fit()`` or ``fit_generator()`` parameters associated with ``EarlyStopping``:
        ``min_delta``, ``patience``, ``baseline``, ``restore_best_weights``, etc
    **tf.estimator**
     - **Metrics** and **Parameters**
      - TensorBoard metrics: ``average_loss``, ``loss``, etc
      - Parameters ``steps`` and ``max_steps``
     - **Artifacts**
      - `Newron Model <https://newron.org/docs/latest/models.html>` (TF saved model) on call
        to ``tf.estimator.export_saved_model``
    **TensorFlow Core**
     - **Metrics**
      - All ``tf.summary.scalar`` calls
    Refer to the autologging tracking documentation for more
    information on `TensorFlow workflows <https://www.mlflow.org/docs/latest/tracking.html#tensorflow-and-keras-experimental>`.
    
    Args:
        every_n_iter: The frequency with which metrics should be logged. For example, a value of
                         100 will log metrics at step 0, 100, 200, etc.
        log_models: If ``True``, trained models are logged as Newron model artifacts.
                       If ``False``, trained models are not logged.
        disable: If ``True``, disables the TensorFlow autologging integration. If ``False``,
                    enables the TensorFlow integration autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      tensorflow that have not been tested against this version of the Newron
                      client or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from Newron during TensorFlow
                   autologging. If ``False``, show all events and warnings during TensorFlow
                   autologging.
        registered_model_name: If given, each time a model is trained, it is registered as a
                                  new model version of the registered model with this name.
                                  The registered model is created if it does not already exist.
        log_input_examples: If ``True``, input examples from training datasets are collected and
                               logged along with tf/keras model artifacts during training. If
                               ``False``, input examples are not logged.
        log_model_signatures: If ``True``,
                                 :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
                                 describing model inputs and outputs are collected and logged along
                                 with tf/keras model artifacts during training. If ``False``,
                                 signatures are not logged. ``False`` by default because
                                 logging TensorFlow models with signatures changes their pyfunc
                                 inference behavior when Pandas DataFrames are passed to
                                 ``predict()``: when a signature is present, an ``np.ndarray``
                                 (for single-output models) or a mapping from
                                 ``str`` -> ``np.ndarray`` (for multi-output models) is returned;
                                 when a signature is not present, a Pandas DataFrame is returned.
    """

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    mlflow.tensorflow.autolog(**values)