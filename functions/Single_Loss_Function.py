from functions.Lin_Solver import Lin_Solver
from scipy.interpolate import interp1d
# Calculates loss value of sigle particular solution and according time series. Serves for isotherm decision.

# PRO JEDNU SPOŽKU JEN JEDNA SUMA
def Single_Loss_Function(experiment):
    for comp in experiment.experimentComponents:
        df = comp.concentrationTime
        modelCurve = Lin_Solver([1, 2, 3])
        #TODO