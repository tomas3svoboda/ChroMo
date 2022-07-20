from Metadata import Metadata
from objects.Experiment_condition import Experiment_condition
from objects.Experiment_component import Experiment_component

"""
Class representing experiments
Contains:
    metadata (Metadata) - additional information
    experiment_condition (Experiment_condition) - describing conditions of experiment
    experiment_components (list(Experiment_component)) - list of components and their concentration in time
"""
class Experiment:
    metadata = Metadata()
    experiment_condition = Experiment_condition()
    experiment_components = list()
