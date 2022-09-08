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

def save_model(
    sk_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    pyfunc_predict_fn="predict",
):
    """
    Save a scikit-learn model to a path on the local file system. Produces an MLflow Model
    containing the following flavors:
        - :py:mod:`mlflow.sklearn`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for scikit-learn models
          that define `predict()`, since `predict()` is required for pyfunc model inference.
    :param sk_model: scikit-learn model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the formats listed in
                                 ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
                                 format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
                                 provides better cross-system compatibility by identifying and
                                 packaging code dependencies with the serialized model.
    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param pyfunc_predict_fn: The name of the prediction function to use for inference with the
           pyfunc representation of the resulting MLflow Model; e.g. ``"predict_proba"``.
    .. code-block:: python
        :caption: Example
        import mlflow.sklearn
        from sklearn.datasets import load_iris
        from sklearn import tree
        iris = load_iris()
        sk_model = tree.DecisionTreeClassifier()
        sk_model = sk_model.fit(iris.data, iris.target)
        # Save the model in cloudpickle format
        # set path to location for persistence
        sk_path_dir_1 = ...
        mlflow.sklearn.save_model(
                sk_model, sk_path_dir_1,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        # save the model in pickle format
        # set path to location for persistence
        sk_path_dir_2 = ...
        mlflow.sklearn.save_model(sk_model, sk_path_dir_2,
                                  serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    mlflow.sklearn.save_model(**values)

def log_model(
    sk_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    pyfunc_predict_fn="predict",
):
    """
    Log a scikit-learn model as an MLflow artifact for the current run. Produces an MLflow Model
    containing the following flavors:
        - :py:mod:`mlflow.sklearn`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for scikit-learn models
          that define `predict()`, since `predict()` is required for pyfunc model inference.
    :param sk_model: scikit-learn model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the formats listed in
                                 ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
                                 format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
                                 provides better cross-system compatibility by identifying and
                                 packaging code dependencies with the serialized model.
    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.
    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param pyfunc_predict_fn: The name of the prediction function to use for inference with the
           pyfunc representation of the resulting MLflow Model; e.g. ``"predict_proba"``.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    .. code-block:: python
        :caption: Example
        import mlflow
        import mlflow.sklearn
        from sklearn.datasets import load_iris
        from sklearn import tree
        iris = load_iris()
        sk_model = tree.DecisionTreeClassifier()
        sk_model = sk_model.fit(iris.data, iris.target)
        # set the artifact_path to location where experiment artifacts will be saved
        #log model params
        mlflow.log_param("criterion", sk_model.criterion)
        mlflow.log_param("splitter", sk_model.splitter)
        # log model
        mlflow.sklearn.log_model(sk_model, "sk_models")
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    return mlflow.sklearn.log_model(**values)

def load_model(model_uri, dst_path=None):
    """
    Load a scikit-learn model from a local file or a run.
    :param model_uri: The location, in URI format, of the MLflow model, for example:
                      - ``Users\me\path\to\local\model``
                      - ``relative\path\to\local\model``
                      - ``s3:\\my_bucket\path\to\model``
                      - ``runs:\<mlflow_run_id>\run-relative\path\to\model``
                      - ``models:\<model_name>\<model_version>``
                      - ``models:\<model_name>\<stage>``
                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https:\\www.mlflow.org\docs\latest\concepts.html#
                      artifact-locations>`.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.
    :return: A scikit-learn model.
    .. code-block:: python
        :caption: Example
        import mlflow.sklearn
        sk_model = mlflow.sklearn.load_model("runs:\96771d893a5e46159d9f3b49bf9013e2\sk_models")
        # use Pandas DataFrame to make predictions
        pandas_df = ...
        predictions = sk_model.predict(pandas_df)
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    return mlflow.sklearn.load_model(**values)

def eval_and_log_metrics(model, X, y_true, *, prefix, sample_weight=None, pos_label=None):
    """
    Computes and logs metrics (and artifacts) for the given model and labeled dataset.
    The metrics\artifacts mirror what is auto-logged when training a model
    (see mlflow.sklearn.autolog).
    :param model: The model to be evaluated.
    :param X: The features for the evaluation dataset.
    :param y_true: The labels for the evaluation dataset.
    :param prefix: Prefix used to name metrics and artifacts.
    :param sample_weight: Per-sample weights to apply in the computation of metrics\artifacts.
    :param pos_label: The positive label used to compute binary classification metrics such as
        precision, recall, f1, etc. This parameter is only used for binary classification model
        - if used on multi-label model, the evaluation will fail;
        - if used for regression model, the parameter will be ignored.
        For multi-label classification, keep `pos_label` unset (or set to `None`), and the
        function will calculate metrics for each label and find their average weighted by support
        (number of true instances for each label).
    :return: The dict of logged metrics. Artifacts can be retrieved by inspecting the run.
    ** Example **
    .. code-block:: python
        from sklearn.linear_model import LinearRegression
        import mlflow
        # enable autologging
        mlflow.sklearn.autolog()
        # prepare training data
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        # prepare evaluation data
        X_eval = np.array([[3, 3], [3, 4]])
        y_eval = np.dot(X_eval, np.array([1,2])) + 3
        # train a model
        model = LinearRegression()
        with mlflow.start_run() as run:
            model.fit(X, y)
            metrics = mlflow.sklearn.eval_and_log_metrics(model, X_eval, y_eval, prefix="val_")
    Each metric's and artifact's name is prefixed with `prefix`, e.g., in the previous example the
    metrics and artifacts are named 'val_XXXXX'. Note that training-time metrics are auto-logged
    as 'training_XXXXX'. Metrics and artifacts are logged under the currently active run if one
    exists, otherwise a new run is started and left active.
    Raises an error if:
      - prefix is empty
      - model is not an sklearn estimator or does not support the 'predict' method
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    return mlflow.sklearn.eval_and_log_metrics(**values)

def autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    max_tuning_runs=5,
    log_post_training_metrics=True,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    registered_model_name=None,
    pos_label=None,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging for scikit-learn estimators.
    **When is autologging performed?**
      Autologging is performed when you call:
      - ``estimator.fit()``
      - ``estimator.fit_predict()``
      - ``estimator.fit_transform()``
    **Logged information**
      **Parameters**
        - Parameters obtained by ``estimator.get_params(deep=True)``. Note that ``get_params``
          is called with ``deep=True``. This means when you fit a meta estimator that chains
          a series of estimators, the parameters of these child estimators are also logged.
      **Training metrics**
        - A training score obtained by ``estimator.score``. Note that the training score is
          computed using parameters given to ``fit()``.
        - Common metrics for classifier:
          - `precision score`
          .. _precision score:
              https:\\scikit-learn.org\stable\modules\generated\sklearn.metrics.precision_score.html
          - `recall score`
          .. _recall score:
              https:\\scikit-learn.org\stable\modules\generated\sklearn.metrics.recall_score.html
          - `f1 score`
          .. _f1 score:
              https:\\scikit-learn.org\stable\modules\generated\sklearn.metrics.f1_score.html
          - `accuracy score`
          .. _accuracy score:
              https:\\scikit-learn.org\stable\modules\generated\sklearn.metrics.accuracy_score.html
          If the classifier has method ``predict_proba``, we additionally log:
          - `log loss`
          .. _log loss:
              https:\\scikit-learn.org\stable\modules\generated\sklearn.metrics.log_loss.html
          - `roc auc score`
          .. _roc auc score:
              https:\\scikit-learn.org\stable\modules\generated\sklearn.metrics.roc_auc_score.html
        - Common metrics for regressor:
          - `mean squared error`
          .. _mean squared error:
              https:\\scikit-learn.org\stable\modules\generated\sklearn.metrics.mean_squared_error.html
          - root mean squared error
          - `mean absolute error`
          .. _mean absolute error:
              https:\\scikit-learn.org\stable\modules\generated\sklearn.metrics.mean_absolute_error.html
          - `r2 score`
          .. _r2 score:
              https:\\scikit-learn.org\stable\modules\generated\sklearn.metrics.r2_score.html
      .. _post training metrics:
      **Post training metrics**
        When users call metric APIs after model training, MLflow tries to capture the metric API
        results and log them as MLflow metrics to the Run associated with the model. The following
        types of scikit-learn metric APIs are supported:
        - model.score
        - metric APIs defined in the `sklearn.metrics` module
        For post training metrics autologging, the metric key format is:
        "{metric_name}[-{call_index}]_{dataset_name}"
        - If the metric function is from `sklearn.metrics`, the MLflow "metric_name" is the
          metric function name. If the metric function is `model.score`, then "metric_name" is
          "{model_class_name}_score".
        - If multiple calls are made to the same scikit-learn metric API, each subsequent call
          adds a "call_index" (starting from 2) to the metric key.
        - MLflow uses the prediction input dataset variable name as the "dataset_name" in the
          metric key. The "prediction input dataset variable" refers to the variable which was
          used as the first argument of the associated `model.predict` or `model.score` call.
          Note: MLflow captures the "prediction input dataset" instance in the outermost call
          frame and fetches the variable name in the outermost call frame. If the "prediction
          input dataset" instance is an intermediate expression without a defined variable
          name, the dataset name is set to "unknown_dataset". If multiple "prediction input
          dataset" instances have the same variable name, then subsequent ones will append an
          index (starting from 2) to the inspected dataset name.
        **Limitations**
           - MLflow can only map the original prediction result object returned by a model
             prediction API (including predict \ predict_proba \ predict_log_proba \ transform,
             but excluding fit_predict \ fit_transform.) to an MLflow run.
             MLflow cannot find run information
             for other objects derived from a given prediction result (e.g. by copying or selecting
             a subset of the prediction result). scikit-learn metric APIs invoked on derived objects
             do not log metrics to MLflow.
           - Autologging must be enabled before scikit-learn metric APIs are imported from
             `sklearn.metrics`. Metric APIs imported before autologging is enabled do not log
             metrics to MLflow runs.
           - If user define a scorer which is not based on metric APIs in `sklearn.metrics`, then
             then post training metric autologging for the scorer is invalid.
        **Tags**
          - An estimator class name (e.g. "LinearRegression").
          - A fully qualified estimator class name
            (e.g. "sklearn.linear_model._base.LinearRegression").
        **Artifacts**
          - An MLflow Model with the :py:mod:`mlflow.sklearn` flavor containing a fitted estimator
            (logged by :py:func:`mlflow.sklearn.log_model()`). The Model also contains the
            :py:mod:`mlflow.pyfunc` flavor when the scikit-learn estimator defines `predict()`.
          - For post training metrics API calls, a "metric_info.json" artifact is logged. This is a
            JSON object whose keys are MLflow post training metric names
            (see "Post training metrics" section for the key format) and whose values are the
            corresponding metric call commands that produced the metrics, e.g.
            ``accuracy_score(y_true=test_iris_y, y_pred=pred_iris_y, normalize=False)``.
    **How does autologging work for meta estimators?**
      When a meta estimator (e.g. `Pipeline`, `GridSearchCV`) calls ``fit()``, it internally calls
      ``fit()`` on its child estimators. Autologging does NOT perform logging on these constituent
      ``fit()`` calls.
      **Parameter search**
          In addition to recording the information discussed above, autologging for parameter
          search meta estimators (`GridSearchCV` and `RandomizedSearchCV`) records child runs
          with metrics for each set of explored parameters, as well as artifacts and parameters
          for the best model (if available).
    **Supported estimators**
      - All estimators obtained by `sklearn.utils.all_estimators` (including meta estimators).
      - `Pipeline`
      - Parameter search estimators (`GridSearchCV` and `RandomizedSearchCV`)
    .. _sklearn.utils.all_estimators:
        https:\\scikit-learn.org\stable\modules\generated\sklearn.utils.all_estimators.html
    .. _Pipeline:
        https:\\scikit-learn.org\stable\modules\generated\sklearn.pipeline.Pipeline.html
    .. _GridSearchCV:
        https:\\scikit-learn.org\stable\modules\generated\sklearn.model_selection.GridSearchCV.html
    .. _RandomizedSearchCV:
        https:\\scikit-learn.org\stable\modules\generated\sklearn.model_selection.RandomizedSearchCV.html
    **Example**
    `See more examples <https:\\github.com\mlflow\mlflow\blob\master\examples\sklearn_autolog>`
    .. code-block:: python
        from pprint import pprint
        import numpy as np
        from sklearn.linear_model import LinearRegression
        import mlflow
        from mlflow import MlflowClient
        def fetch_logged_data(run_id):
            client = MlflowClient()
            data = client.get_run(run_id).data
            tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
            return data.params, data.metrics, tags, artifacts
        # enable autologging
        mlflow.sklearn.autolog()
        # prepare training data
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        # train a model
        model = LinearRegression()
        with mlflow.start_run() as run:
            model.fit(X, y)
        # fetch logged data
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
        pprint(params)
        # {'copy_X': 'True',
        #  'fit_intercept': 'True',
        #  'n_jobs': 'None',
        #  'normalize': 'False'}
        pprint(metrics)
        # {'training_score': 1.0,
           'training_mae': 2.220446049250313e-16,
           'training_mse': 1.9721522630525295e-31,
           'training_r2_score': 1.0,
           'training_rmse': 4.440892098500626e-16}
        pprint(tags)
        # {'estimator_class': 'sklearn.linear_model._base.LinearRegression',
        #  'estimator_name': 'LinearRegression'}
        pprint(artifacts)
        # ['model\MLmodel', 'model\conda.yaml', 'model\model.pkl']
    :param log_input_examples: If ``True``, input examples from training datasets are collected and
                               logged along with scikit-learn model artifacts during training. If
                               ``False``, input examples are not logged.
                               Note: Input examples are MLflow model attributes
                               and are only collected if ``log_models`` is also ``True``.
    :param log_model_signatures: If ``True``,
                                 :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
                                 describing model inputs and outputs are collected and logged along
                                 with scikit-learn model artifacts during training. If ``False``,
                                 signatures are not logged.
                                 Note: Model signatures are MLflow model attributes
                                 and are only collected if ``log_models`` is also ``True``.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
                       Input examples and model signatures, which are attributes of MLflow models,
                       are also omitted when ``log_models`` is ``False``.
    :param disable: If ``True``, disables the scikit-learn autologging integration. If ``False``,
                    enables the scikit-learn autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      scikit-learn that have not been tested against this version of the MLflow
                      client or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during scikit-learn
                   autologging. If ``False``, show all events and warnings during scikit-learn
                   autologging.
    :param max_tuning_runs: The maximum number of child Mlflow runs created for hyperparameter
                            search estimators. To create child runs for the best `k` results from
                            the search, set `max_tuning_runs` to `k`. The default value is to track
                            the best 5 search parameter sets. If `max_tuning_runs=None`, then
                            a child run is created for each search parameter set. Note: The best k
                            results is based on ordering in `rank_test_score`. In the case of
                            multi-metric evaluation with a custom scorer, the first scorer’s
                            `rank_test_score_<scorer_name>` will be used to select the best k
                            results. To change metric used for selecting best k results, change
                            ordering of dict passed as `scoring` parameter for estimator.
    :param log_post_training_metrics: If ``True``, post training metrics are logged. Defaults to
                                      ``True``. See the `post training metrics` section for more
                                      details.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
                                 ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    :param registered_model_name: If given, each time a model is trained, it is registered as a
                                  new model version of the registered model with this name.
                                  The registered model is created if it does not already exist.
    :param pos_label: If given, used as the positive label to compute binary classification
                      training metrics such as precision, recall, f1, etc. This parameter should
                      only be set for binary classification model. If used for multi-label model,
                      the training metrics calculation will fail and the training metrics won't
                      be logged. If used for regression model, the parameter will be ignored.
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    mlflow.sklearn.autolog(**values)