from objects.Metadata import Metadata

"""
Set of experiments for analysis
Contains:
    metadata (Metadata) - additional information
    experiments (set(Experiment)) - set of experiments
"""
class Experiment_set:
    metadata = Metadata()
    experiments = set()