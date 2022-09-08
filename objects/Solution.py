from objects.ExperimentSet import ExperimentSet
import pandas as pd

class Solution:
    def __init__(self):
        self.experimentSet = ExperimentSet()
        self.modelExperimentSet = ExperimentSet()
        self.results = pd.DataFrame([], columns=['Component', 'Path_to_experiment', 'Henry Constant', 'Dispersion Coeficient', 'Porosity'])

    def Add_Result(self, comp, pathToExp, henryConst, disperCoef, porosity):
        newRow = pd.DataFrame([[comp, pathToExp, henryConst, disperCoef, porosity]], columns=['Component', 'Path_to_experiment', 'Henry Constant', 'Dispersion Coeficient', 'Porosity'])
        self.results = pd.concat([self.results, newRow], ignore_index=True)
        print(self.results)

    def Export_To_CSV(self, path):
        self.results.to_csv(path, index=False, compression=None)