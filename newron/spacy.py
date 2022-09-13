import mlflow.spacy

FLAVOR_NAME = "spacy"
DEFAULT_AWAIT_MAX_SLEEP_SECONDS = 5 * 60

_load_model = mlflow.spacy._load_model
_SpacyModelWrapper = mlflow.spacy._SpacyModelWrapper
_load_pyfunc = mlflow.spacy._load_pyfunc

save_model = mlflow.spacy.save_model
log_model = mlflow.spacy.log_model
load_model = mlflow.spacy.load_model
get_default_pip_requirements = mlflow.spacy.get_default_pip_requirements
get_default_conda_env = mlflow.spacy.get_default_conda_env

