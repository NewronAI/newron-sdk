import mlflow.sklearn

_apis_autologging_disabled = [
    "cross_validate",
    "cross_val_predict",
    "cross_val_score",
    "learning_curve",
    "permutation_test_score",
    "validation_curve",
]

_gen_estimators_to_patch = mlflow.sklearn._gen_estimators_to_patch
get_default_pip_requirements = mlflow.sklearn.get_default_pip_requirements
get_default_conda_env = mlflow.sklearn.get_default_conda_env
save_model = mlflow.sklearn.save_model
log_model = mlflow.sklearn.log_model
_load_model_from_local_file = mlflow.sklearn._load_model_from_local_file
_load_pyfunc = mlflow.sklearn._load_pyfunc
_SklearnCustomModelPicklingError = mlflow.sklearn._SklearnCustomModelPicklingError
_dump_model = mlflow.sklearn._dump_model
_save_model = mlflow.sklearn._save_model
load_model = mlflow.sklearn.load_model
_AutologgingMetricsManager = mlflow.sklearn._AutologgingMetricsManager
autolog = mlflow.sklearn.autolog
_autolog = mlflow.sklearn._autolog
eval_and_log_metrics = mlflow.sklearn.eval_and_log_metrics
_eval_and_log_metrics_impl = mlflow.sklearn._eval_and_log_metrics_impl 