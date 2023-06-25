from objects.ExperimentSet import ExperimentSet
from objects.Metadata import Metadata
import pandas as pd

# Retired
class Solution:
    def __init__(self):
        self.experimentSet = ExperimentSet()
        self.modelExperimentSet = ExperimentSet()
        self.metadata = Metadata()
        self.results = pd.DataFrame([], columns=['Component', 'Path_to_experiment', 'Henry Constant', 'Dispersion Coeficient', 'Porosity', 'Langmuir Constant', 'Freundlich Constant'])

    def Add_Result(self, comp, pathToExp, henryConst, disperCoef, porosity):
        newRow = pd.DataFrame([[comp, pathToExp, henryConst, disperCoef, porosity]], columns=['Component', 'Path_to_experiment', 'Henry Constant', 'Dispersion Coeficient', 'Porosity'])
        self.results = pd.concat([self.results, newRow], ignore_index=True)
        print(self.results)

    def Export_To_CSV(self, path):
        self.results.to_csv(path, index=False, compression=None)