import mlflow.fastai

get_default_pip_requirements = mlflow.fastai.get_default_pip_requirements
get_default_conda_env = mlflow.fastai.get_default_conda_env
save_model = mlflow.fastai.save_model
log_model = mlflow.fastai.log_model
_load_model = mlflow.fastai._load_model
_FastaiModelWrapper = mlflow.fastai._FastaiModelWrapper
_load_pyfunc = mlflow.fastai._load_pyfunc
load_model = mlflow.fastai.load_model
autolog = mlflow.fastai.autolog