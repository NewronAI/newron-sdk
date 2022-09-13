import mlflow.prophet

_load_model = mlflow.prophet._load_model
_ProphetModelWrapper = mlflow.prophet._ProphetModelWrapper
_load_pyfunc = mlflow.prophet._load_pyfunc
_save_model = mlflow.prophet._save_model


save_model = mlflow.prophet.save_model
log_model = mlflow.prophet.log_model
load_model = mlflow.prophet.load_model
get_default_pip_requirements = mlflow.prophet.get_default_pip_requirements
get_default_conda_env = mlflow.prophet.get_default_conda_env

