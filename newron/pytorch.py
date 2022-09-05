import mlflow.pytorch
import inspect
from mlflow.utils.requirements_utils import _get_pinned_requirement
from newron.models import ModelSignature, ModelInputExample
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
FLAVOR_NAME = "pytorch"
DEFAULT_AWAIT_MAX_SLEEP_SECONDS = 5 * 60

_SERIALIZED_TORCH_MODEL_FILE_NAME = "model.pth"
_TORCH_STATE_DICT_FILE_NAME = "state_dict.pth"
_PICKLE_MODULE_INFO_FILE_NAME = "pickle_module_info.txt"
_EXTRA_FILES_KEY = "extra_files"
_REQUIREMENTS_FILE_KEY = "requirements_file"

_load_model = mlflow.pytorch._load_model
_PyTorchWrapper = mlflow.pytorch._PyTorchWrapper
_load_pyfunc = mlflow.pytorch._load_pyfunc


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return list(
        map(
            _get_pinned_requirement,
            [
                "torch",
                # We include CloudPickle in the default environment because
                # it's required by the default pickle module used by `save_model()`
                # and `log_model()`: `mlflow.pytorch.pickle_module`.
                "cloudpickle",
            ],
        )
    )


def get_default_conda_env():
    """
    :return: The default Conda environment as a dictionary for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    .. code-block:: python
        :caption: Example
        import mlflow.pytorch
        # Log PyTorch model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")
        # Fetch the associated conda environment
        env = mlflow.pytorch.get_default_conda_env()
        print("conda env: {}".format(env))
    .. code-block:: text
        :caption: Output
        conda env {'name': 'mlflow-env',
                   'channels': ['conda-forge'],
                   'dependencies': ['python=3.7.5',
                                    {'pip': ['torch==1.5.1',
                                             'mlflow',
                                             'cloudpickle==1.6.0']}]}
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())

def log_model(
    pytorch_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    pickle_module=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    requirements_file=None,
    extra_files=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Log a PyTorch model as an MLflow artifact for the current run.
        .. warning::
            Log the model with a signature to avoid inference errors.
            If the model is logged without a signature, the MLflow Model Server relies on the
            default inferred data type from NumPy. However, PyTorch often expects different
            defaults, particularly when parsing floats. You must include the signature to ensure
            that the model is logged with the correct data type so that the MLflow model server
            can correctly provide valid input.
    :param pytorch_model: PyTorch model to be saved. Can be either an eager model (subclass of
                          ``torch.nn.Module``) or scripted model prepared via ``torch.jit.script``
                          or ``torch.jit.trace``.
                          The model accept a single ``torch.FloatTensor`` as
                          input and produce a single output tensor.
                          If saving an eager model, any code dependencies of the
                          model's class, including the class definition itself, should be
                          included in one of the following locations:
                          - The package(s) listed in the model's Conda environment, specified
                            by the ``conda_env`` parameter.
                          - One or more of the files specified by the ``code_paths`` parameter.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param pickle_module: The module that PyTorch should use to serialize ("pickle") the specified
                          ``pytorch_model``. This is passed as the ``pickle_module`` parameter
                          to ``torch.save()``. By default, this module is also used to
                          deserialize ("unpickle") the PyTorch model at load time.
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
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :param requirements_file:
        .. warning::
            ``requirements_file`` has been deprecated. Please use ``pip_requirements`` instead.
        A string containing the path to requirements file. Remote URIs are resolved to absolute
        filesystem paths. For example, consider the following ``requirements_file`` string:
        .. code-block:: python
            requirements_file = "s3://my-bucket/path/to/my_file"
        In this case, the ``"my_file"`` requirements file is downloaded from S3. If ``None``,
        no requirements file is added to the model.
    :param extra_files: A list containing the paths to corresponding extra files. Remote URIs
                      are resolved to absolute filesystem paths.
                      For example, consider the following ``extra_files`` list -
                      extra_files = ["s3://my-bucket/path/to/my_file1",
                                    "s3://my-bucket/path/to/my_file2"]
                      In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3.
                      If ``None``, no extra files are added to the model.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param kwargs: kwargs to pass to ``torch.save`` method.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    .. code-block:: python
        :caption: Example
        import numpy as np
        import torch
        import mlflow.pytorch
        class LinearNNModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)  # One in and one out
            def forward(self, x):
                y_pred = self.linear(x)
                return y_pred
        def gen_data():
            # Example linear model modified to use y = 2x
            # from https://github.com/hunkim/PyTorchZeroToAll
            # X training data, y labels
            X = torch.arange(1.0, 25.0).view(-1, 1)
            y = torch.from_numpy(np.array([x * 2 for x in X])).view(-1, 1)
            return X, y
        # Define model, loss, and optimizer
        model = LinearNNModel()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        # Training loop
        epochs = 250
        X, y = gen_data()
        for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing X to the model
            y_pred = model(X)
            # Compute the loss
            loss = criterion(y_pred, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Log the model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")
            # convert to scripted model and log the model
            scripted_pytorch_model = torch.jit.script(model)
            mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")
        # Fetch the logged model artifacts
        print("run_id: {}".format(run.info.run_id))
        for artifact_path in ["model/data", "scripted_model/data"]:
            artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id,
                        artifact_path)]
            print("artifacts: {}".format(artifacts))
    .. code-block:: text
        :caption: Output
        run_id: 1a1ec9e413ce48e9abf9aec20efd6f71
        artifacts: ['model/data/model.pth',
                    'model/data/pickle_module_info.txt']
        artifacts: ['scripted_model/data/model.pth',
                    'scripted_model/data/pickle_module_info.txt']
    .. figure:: ../_static/images/pytorch_logged_models.png
        PyTorch logged models
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    return mlflow.pytorch.log_model(**values)

def save_model(
    pytorch_model,
    path,
    conda_env=None,
    mlflow_model=None,
    code_paths=None,
    pickle_module=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    requirements_file=None,
    extra_files=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Save a PyTorch model to a path on the local file system.
    :param pytorch_model: PyTorch model to be saved. Can be either an eager model (subclass of
                          ``torch.nn.Module``) or scripted model prepared via ``torch.jit.script``
                          or ``torch.jit.trace``.
                          The model accept a single ``torch.FloatTensor`` as
                          input and produce a single output tensor.
                          If saving an eager model, any code dependencies of the
                          model's class, including the class definition itself, should be
                          included in one of the following locations:
                          - The package(s) listed in the model's Conda environment, specified
                            by the ``conda_env`` parameter.
                          - One or more of the files specified by the ``code_paths`` parameter.
    :param path: Local path where the model is to be saved.
    :param conda_env: {{ conda_env }}
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param pickle_module: The module that PyTorch should use to serialize ("pickle") the specified
                          ``pytorch_model``. This is passed as the ``pickle_module`` parameter
                          to ``torch.save()``. By default, this module is also used to
                          deserialize ("unpickle") the PyTorch model at load time.
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
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param requirements_file:
        .. warning::
            ``requirements_file`` has been deprecated. Please use ``pip_requirements`` instead.
        A string containing the path to requirements file. Remote URIs are resolved to absolute
        filesystem paths. For example, consider the following ``requirements_file`` string:
        .. code-block:: python
            requirements_file = "s3://my-bucket/path/to/my_file"
        In this case, the ``"my_file"`` requirements file is downloaded from S3. If ``None``,
        no requirements file is added to the model.
    :param extra_files: A list containing the paths to corresponding extra files. Remote URIs
                      are resolved to absolute filesystem paths.
                      For example, consider the following ``extra_files`` list -
                      extra_files = ["s3://my-bucket/path/to/my_file1",
                                    "s3://my-bucket/path/to/my_file2"]
                      In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3.
                      If ``None``, no extra files are added to the model.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param kwargs: kwargs to pass to ``torch.save`` method.
    .. code-block:: python
        :caption: Example
        import os
        import torch
        import mlflow.pytorch
        # Class defined here
        class LinearNNModel(torch.nn.Module):
            ...
        # Initialize our model, criterion and optimizer
        ...
        # Training loop
        ...
        # Save PyTorch models to current working directory
        with mlflow.start_run() as run:
            mlflow.pytorch.save_model(model, "model")
            # Convert to a scripted model and save it
            scripted_pytorch_model = torch.jit.script(model)
            mlflow.pytorch.save_model(scripted_pytorch_model, "scripted_model")
        # Load each saved model for inference
        for model_path in ["model", "scripted_model"]:
            model_uri = "{}/{}".format(os.getcwd(), model_path)
            loaded_model = mlflow.pytorch.load_model(model_uri)
            print("Loaded {}:".format(model_path))
            for x in [6.0, 8.0, 12.0, 30.0]:
                X = torch.Tensor([[x]])
                y_pred = loaded_model(X)
                print("predict X: {}, y_pred: {:.2f}".format(x, y_pred.data.item()))
            print("--")
    .. code-block:: text
        :caption: Output
        Loaded model:
        predict X: 6.0, y_pred: 11.90
        predict X: 8.0, y_pred: 15.92
        predict X: 12.0, y_pred: 23.96
        predict X: 30.0, y_pred: 60.13
        --
        Loaded scripted_model:
        predict X: 6.0, y_pred: 11.90
        predict X: 8.0, y_pred: 15.92
        predict X: 12.0, y_pred: 23.96
        predict X: 30.0, y_pred: 60.13
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    mlflow.pytorch.save_model(**values)

def load_model(model_uri, dst_path=None, **kwargs):
    """
    Load a PyTorch model from a local file or a run.
    :param model_uri: The location, in URI format, of the MLflow model, for example:
                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``
                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.
    :param kwargs: kwargs to pass to ``torch.load`` method.
    :return: A PyTorch model.
    .. code-block:: python
        :caption: Example
        import torch
        import mlflow.pytorch
        # Class defined here
        class LinearNNModel(torch.nn.Module):
            ...
        # Initialize our model, criterion and optimizer
        ...
        # Training loop
        ...
        # Log the model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")
        # Inference after loading the logged model
        model_uri = "runs:/{}/model".format(run.info.run_id)
        loaded_model = mlflow.pytorch.load_model(model_uri)
        for x in [4.0, 6.0, 30.0]:
            X = torch.Tensor([[x]])
            y_pred = loaded_model(X)
            print("predict X: {}, y_pred: {:.2f}".format(x, y_pred.data.item()))
    .. code-block:: text
        :caption: Output
        predict X: 4.0, y_pred: 7.57
        predict X: 6.0, y_pred: 11.64
        predict X: 30.0, y_pred: 60.48
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    return mlflow.pytorch.load_model(**values)

def autolog(
    log_every_n_epoch=1,
    log_every_n_step=None,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging from `PyTorch Lightning
    <https://pytorch-lightning.readthedocs.io/en/latest>`_ to MLflow.
    Autologging is performed when you call the `fit` method of
    `pytorch_lightning.Trainer() \
    <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#>`_.
    Explore the complete `PyTorch MNIST \
    <https://github.com/mlflow/mlflow/tree/master/examples/pytorch/MNIST>`_ for
    an expansive example with implementation of additional lightening steps.
    **Note**: Autologging is only supported for PyTorch Lightning models,
    i.e., models that subclass
    `pytorch_lightning.LightningModule \
    <https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html>`_.
    In particular, autologging support for vanilla PyTorch models that only subclass
    `torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_
    is not yet available.
    :param log_every_n_epoch: If specified, logs metrics once every `n` epochs. By default, metrics
                       are logged after every epoch.
    :param log_every_n_step: If specified, logs batch metrics once every `n` global step.
                       By default, metrics are not logged for steps. Note that setting this to 1 can
                       cause performance issues and is not recommended.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param disable: If ``True``, disables the PyTorch Lightning autologging integration.
                    If ``False``, enables the PyTorch Lightning autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      pytorch and pytorch-lightning that have not been tested against this version
                      of the MLflow client or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during PyTorch
                   Lightning autologging. If ``False``, show all events and warnings during
                   PyTorch Lightning autologging.
    :param registered_model_name: If given, each time a model is trained, it is registered as a
                                  new model version of the registered model with this name.
                                  The registered model is created if it does not already exist.
    .. code-block:: python
        :caption: Example
        import os
        import pytorch_lightning as pl
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.datasets import MNIST
        try:
            from torchmetrics.functional import accuracy
        except ImportError:
            from pytorch_lightning.metrics.functional import accuracy
        import mlflow.pytorch
        from mlflow import MlflowClient
        # For brevity, here is the simplest most minimal example with just a training
        # loop step, (no validation, no testing). It illustrates how you can use MLflow
        # to auto log parameters, metrics, and models.
        class MNISTModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(28 * 28, 10)
            def forward(self, x):
                return torch.relu(self.l1(x.view(x.size(0), -1)))
            def training_step(self, batch, batch_nb):
                x, y = batch
                logits = self(x)
                loss = F.cross_entropy(logits, y)
                pred = logits.argmax(dim=1)
                acc = accuracy(pred, y)
                # Use the current of PyTorch logger
                self.log("train_loss", loss, on_epoch=True)
                self.log("acc", acc, on_epoch=True)
                return loss
            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.02)
        def print_auto_logged_info(r):
            tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
            print("run_id: {}".format(r.info.run_id))
            print("artifacts: {}".format(artifacts))
            print("params: {}".format(r.data.params))
            print("metrics: {}".format(r.data.metrics))
            print("tags: {}".format(tags))
        # Initialize our model
        mnist_model = MNISTModel()
        # Initialize DataLoader from MNIST Dataset
        train_ds = MNIST(os.getcwd(), train=True,
            download=True, transform=transforms.ToTensor())
        train_loader = DataLoader(train_ds, batch_size=32)
        # Initialize a trainer
        trainer = pl.Trainer(max_epochs=20, progress_bar_refresh_rate=20)
        # Auto log all MLflow entities
        mlflow.pytorch.autolog()
        # Train the model
        with mlflow.start_run() as run:
            trainer.fit(mnist_model, train_loader)
        # fetch the auto logged parameters and metrics
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    .. code-block:: text
        :caption: Output
        run_id: 42caa17b60cb489c8083900fb52506a7
        artifacts: ['model/MLmodel', 'model/conda.yaml', 'model/data']
        params: {'betas': '(0.9, 0.999)',
                 'weight_decay': '0',
                 'epochs': '20',
                 'eps': '1e-08',
                 'lr': '0.02',
                 'optimizer_name': 'Adam', '
                 amsgrad': 'False'}
        metrics: {'acc_step': 0.0,
                  'train_loss_epoch': 1.0917967557907104,
                  'train_loss_step': 1.0794280767440796,
                  'train_loss': 1.0794280767440796,
                  'acc_epoch': 0.0033333334140479565,
                  'acc': 0.0}
        tags: {'Mode': 'training'}
    .. figure:: ../_static/images/pytorch_lightening_autolog.png
        PyTorch autologged MLflow entities
    """
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    mlflow.pytorch.autolog(**values)