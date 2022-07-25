from objects.Metadata import Metadata

"""
List of clusters of components
Contains:
    cluster (list(Experiment_component)) - cluster of components
    metadata - additional information
"""
class ExperimentClusters:
    def __init__(self):
        self.clusters = dict()
        self.metadata = Metadata()
