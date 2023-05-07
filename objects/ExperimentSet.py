from objects.Metadata import Metadata
import os

"""
Set of experiments for analysis
Contains:
    metadata (Metadata) - additional information
    experiments (set(Experiment)) - set of experiments
"""
class ExperimentSet:
    def __init__(self):
        self.metadata = Metadata()
        self.experiments = list()

    def get_exp_by_name(self, name):
        for exp in self.experiments:
            head, tail = os.path.split(exp.metadata.path)
            if name == tail:
                return exp
        return False

    def get_exps_by_names(self, names):
        expList = []
        for exp in self.experiments:
            head, tail = os.path.split(exp.metadata.path)
            if tail in names:
                expList.append(exp)
        return expList
