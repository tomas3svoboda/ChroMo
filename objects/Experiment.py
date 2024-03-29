from objects.Metadata import Metadata
from objects.ExperimentCondition import ExperimentCondition
from objects.ExperimentComponent import ExperimentComponent

class Experiment:
    """Class representing experiments
    Contains:
    metadata (Metadata) - additional information
    experiment_condition (Experiment_condition) - describing conditions of experiment
    experiment_components (list(Experiment_component)) - list of components and their concentration in time
    """
    def __init__(self):
        self.metadata = Metadata()
        self.experimentCondition = ExperimentCondition()
        self.experimentComponents = list()
        self.shift = -1
