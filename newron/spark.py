import mlflow.spark
#### do we need classes to import e.g._PyFuncModelWrapper

FLAVOR_NAME = "spark"

# Default temporary directory on DFS. Used to write / read from Spark ML models.
DFS_TMP = "/tmp/newron"
_SPARK_MODEL_PATH_SUB = mlflow.spark._SPARK_MODEL_PATH_SUB
_MLFLOWDBFS_SCHEME = mlflow.spark._MLFLOWDBFS_SCHEME

get_default_pip_requirements = mlflow.spark.get_default_pip_requirements
get_default_conda_env = mlflow.spark.get_default_conda_env
log_model = mlflow.spark.log_model
_tmp_path = mlflow.spark._tmp_path
_mlflowdbfs_path = mlflow.spark._mlflowdbfs_path
_maybe_save_model = mlflow.spark._maybe_save_model
_should_use_mlflowdbfs = mlflow.spark._should_use_mlflowdbfs
_save_model_metadata = mlflow.spark._save_model_metadata
_validate_model = mlflow.spark._validate_model
save_model = mlflow.spark.save_model
_shutil_copytree_without_file_permissions = mlflow.spark._shutil_copytree_without_file_permissions
_load_model_databricks = mlflow.spark._load_model_databricks
_load_model = mlflow.spark._load_model
load_model = mlflow.spark.load_model
_load_pyfunc = mlflow.spark._load_pyfunc
_find_and_set_features_col_as_vector_if_needed = mlflow.spark._find_and_set_features_col_as_vector_if_needed
autolog = mlflow.spark.autolog