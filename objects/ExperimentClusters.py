from objects.Metadata import Metadata

class ExperimentClusters:
    """List of clusters of components
    Contains:
    cluster (list(Experiment_component)) - cluster of components
    metadata - additional information
    """
    def __init__(self):
        self.clusters = dict()
        self.metadata = Metadata()
