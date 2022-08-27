from functions.Lin_Solver import Lin_Solver
from scipy.interpolate import interp1d
import numpy as np
# Calculates loss value of sigle particular solution and according time series. Serves for isotherm decision.

def Single_Loss_Function_1(params, experimentComp):
    errSum = 0
    df = experimentComp.concentrationTime
    modelCurve = Lin_Solver(experimentComp.experiment.experimentCondition.flowRate,
                            experimentComp.experiment.experimentCondition.columnLength,
                            experimentComp.experiment.experimentCondition.columnDiameter,
                            experimentComp.experiment.experimentCondition.feedVolume,
                            experimentComp.feedConcentration,
                            params[0],
                            params[1],
                            params[2],
                            debugPrint=False)[:, -1]
    # !remove hard wired time values
    time = np.linspace(0, 10800, 3000)
    f = interp1d(time, modelCurve)
    modelCurveInterpolated = f(df.iloc[:, 0].to_numpy())
    tmpErrSum = 0
    cmax = 0
    for a, b in zip(df.iloc[:, 1].to_numpy(), modelCurveInterpolated):
        err = abs(a-b)
        tmpErrSum += err
        if a > cmax:
            cmax = a
    errSum += tmpErrSum
    errSum = errSum/cmax
    return errSum
