import mlflow.tracking._model_registry.fluent
import mlflow.tracking.fluent
from mlflow import tracking
from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
from mlflow.entities import Experiment
from auth import Auth0

get_tracking_uri = tracking.get_tracking_uri
get_registry_uri = tracking.get_registry_uri
create_experiment = mlflow.tracking.fluent.create_experiment
set_experiment = mlflow.tracking.fluent.set_experiment
log_params = mlflow.tracking.fluent.log_params
log_metrics = mlflow.tracking.fluent.log_metrics
set_experiment_tags = mlflow.tracking.fluent.set_experiment_tags
set_experiment_tag = mlflow.tracking.fluent.set_experiment_tag
set_tags = mlflow.tracking.fluent.set_tags
delete_experiment = mlflow.tracking.fluent.delete_experiment
delete_run = mlflow.tracking.fluent.delete_run
register_model = mlflow.tracking._model_registry.fluent.register_model
autolog = mlflow.tracking.fluent.autolog
evaluate = mlflow.models.evaluate
last_active_run = mlflow.tracking.fluent.last_active_run
NewronClient = tracking.MlflowClient
ActiveRun = mlflow.tracking.fluent.ActiveRun
log_param = mlflow.tracking.fluent.log_param
log_metric = mlflow.tracking.fluent.log_metric
set_tag = mlflow.tracking.fluent.set_tag
delete_tag = mlflow.tracking.fluent.delete_tag
log_artifacts = mlflow.tracking.fluent.log_artifacts
log_artifact = mlflow.tracking.fluent.log_artifact
log_text = mlflow.tracking.fluent.log_text
log_dict = mlflow.tracking.fluent.log_dict
log_image = mlflow.tracking.fluent.log_image
log_figure = mlflow.tracking.fluent.log_figure
active_run = mlflow.tracking.fluent.active_run
get_run = mlflow.tracking.fluent.get_run
start_run = mlflow.tracking.fluent.start_run
end_run = mlflow.tracking.fluent.end_run
search_runs = mlflow.tracking.fluent.search_runs
list_run_infos = mlflow.tracking.fluent.list_run_infos
get_artifact_uri = mlflow.tracking.fluent.get_artifact_uri
set_tracking_uri = tracking.set_tracking_uri
set_registry_uri = tracking.set_registry_uri
get_experiment = mlflow.tracking.fluent.get_experiment
get_experiment_by_name = mlflow.tracking.fluent.get_experiment_by_name
list_experiments = mlflow.tracking.fluent.list_experiments

SERVER_URI = "https://mlflow-tracking-server-zx44gn5asa-uc.a.run.app"

class NewronPluginRequestHeaderProvider(RequestHeaderProvider):
    """RequestHeaderProvider provided through plugin system"""

    def in_context(self):
        return True

    def request_headers(self):
        return {"project_id": Experiment.experiment_id}

def init(experiment_name:str = None):
    _auth = Auth0()
    print("init method")
    if _auth.authenticate():
        set_tracking_uri(SERVER_URI)
        print(SERVER_URI)
        set_experiment(experiment_name)
    else:
        raise Exception("Authentication failed")