import mlflow.tensorflow
FLAVOR_NAME = "tensorflow"

load_model = mlflow.tensorflow.load_model
_load_pyfunc = mlflow.tensorflow._load_pyfunc
save_model = mlflow.tensorflow.save_model
_validate_saved_model = mlflow.tensorflow._validate_saved_model
_load_tensorflow_saved_model = mlflow.tensorflow._load_tensorflow_saved_model
_parse_flavor_configuration = mlflow.tensorflow._parse_flavor_configuration
get_default_pip_requirements = mlflow.tensorflow.get_default_pip_requirements
get_default_conda_env = mlflow.tensorflow.get_default_conda_env
log_model = mlflow.tensorflow.log_model
autolog = mlflow.tensorflow.autolog
_TF2Wrapper = mlflow.tensorflow._TF2Wrapper
_assoc_list_to_map = mlflow.tensorflow._assoc_list_to_map
_flush_queue = mlflow.tensorflow._flush_queue
_add_to_queue = mlflow.tensorflow._add_to_queue
_add_to_queue = mlflow.tensorflow._add_to_queue
_get_tensorboard_callback = mlflow.tensorflow._get_tensorboard_callback
_setup_callbacks = mlflow.tensorflow._setup_callbacks