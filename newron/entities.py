


from mlflow.entities import Experiment, ExperimentTag, FileInfo,Metric,LifecycleStage
from mlflow.entities import Param, Run, RunData, RunInfo, RunStatus, RunTag, SourceType,ViewType
# should it be inherited from experiment
class Project():
    """
    Projecr object.
    """

    DEFAULT_PROJECT_NAME = "Default"

    def __init__(self, project_id, name, description, team,privacy_rule, tags=None):
        #super().__init__()
        self._project_id = project_id
        self._name = name
        self._description = description
        self._team = team
        self._privacy_rule = privacy_rule

        self._tags = {tag.key: tag.value for tag in (tags or [])}

    @property
    def project_id(self):
        """String ID of the experiment."""
        return self._project_id

    @property
    def name(self):
        """String name of the experiment."""
        return self._name

    def _set_name(self, new_name):
        self._name = new_name

class Sweeps(Experiment):
    """
    Sweeps object
    """
    pass
    # define how would sweep be used