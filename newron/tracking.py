import mlflow.tracking._model_registry.fluent
import mlflow.tracking.fluent
from mlflow import tracking
from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
from mlflow.entities import Experiment
from newron.auth import Auth0
import requests

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

PROJECT_URI = "https://api.newron.ai/v1/project"
DEFAULT_AWAIT_MAX_SLEEP_SECONDS = 5 * 60

import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)

class NewronPluginRequestHeaderProvider(RequestHeaderProvider):
    """RequestHeaderProvider provided through plugin system"""

    def in_context(self):
        return True

    def request_headers(self):
        return {"project_id": Experiment.experiment_id}

def init(project_name, experiment_name = 'Default', description=None):
    """
    Function to initialise a tracking experiment with Newron. The function authenticates
    the user against the Newron server and allows the user to activate an experiment under 
    a project. 
    Args:
 
        project_name: Case sensitive name of the project under which the experimentis
                            to be activated.
        experiment_name: Case sensitive name of the experiment to be activated. If an experiment
                            with this name does not exist, a new experiment wth this name is
                            created.
        framework: proide name of framework from the list of supported frameworks by newron
        exp_desc: Description of the experiment being activated. In case the experiment had a 
                            description previously it would be overwritten by the new description.
    """
    
    _auth = Auth0()
    auth_response = _auth.authenticate()
    if auth_response:
        set_tracking_uri(SERVER_URI)
        import requests        
        url = PROJECT_URI
        payload = {}
        payload["accountId"] = auth_response["email"]
        payload["userId"] = auth_response["email"]
        ##auth_response["sub"].split("|")[1]
        payload["projectName"] = project_name
        payload["experimentName"] = experiment_name
        if description:
            payload["desc"] = description

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + auth_response["access_token"]
        }
        gateway_response = requests.request("PUT", url, json=payload, headers=headers)

        set_experiment(experiment_id = gateway_response.json()["mlflow"]["experimentId"])


        #if framework in ['sklearn','keras','tensorflow','pytorch','xgboost','fastai']:
        #  eval('newron.{framework}.autolog()')
        #else:
        #  newron.autolog()

        mlflow.autolog()
        
    else:
        raise Exception("Authentication failed")
       

