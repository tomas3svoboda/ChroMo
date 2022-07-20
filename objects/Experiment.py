from Metadata import Metadata
from objects.ExperimentCondition import ExperimentCondition
from objects.ExperimentComponent import ExperimentComponent

"""
Class representing experiments
Contains:
    metadata (Metadata) - additional information
    experiment_condition (Experiment_condition) - describing conditions of experiment
    experiment_components (list(Experiment_component)) - list of components and their concentration in time
"""
class Experiment:
    metadata = Metadata()
    experiment_condition = ExperimentCondition()
    experiment_components = list()
