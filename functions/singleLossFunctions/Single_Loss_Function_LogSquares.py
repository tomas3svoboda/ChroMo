from functions.solvers.Solver_Choice import Solver_Choice
from scipy.interpolate import interp1d
import math
import numpy as np
# Calculates loss value of sigle particular solution and according time series. Serves for isotherm decision.

def Single_Loss_Function_LogSquares(params, experimentComp, solver, factor):
    errSum = 0
    df = experimentComp.concentrationTime
    modelCurve = Solver_Choice(solver, params, experimentComp)[:, -1]
    # TODO remove hard wired time values
    minTime = df.iat[0, 0]
    maxTime = df.iat[-1, 0]
    time = np.linspace(minTime, maxTime, 2000)
    f = interp1d(time, modelCurve)
    modelCurveInterpolated = f(df.iloc[:, 0].to_numpy())
    tmpErrSum = 0
    max = 0
    for a, b in zip(df.iloc[:, 1].to_numpy(), modelCurveInterpolated):
        err = ((a-b)**2)
        tmpErrSum += err
        if (factor == 2 or factor == 3) and a > max:
            max = a
    errSum += tmpErrSum
    if factor == 1:
        errSum = errSum/1
    elif factor == 2:
        errSum = errSum/max
    elif factor == 3:
        errSum = errSum/(max**2)
    elif factor == 4:
        errSum = errSum/experimentComp.feedConcentration
    elif factor == 5:
        errSum = errSum/(experimentComp.feedConcentration**2)
    elif factor == 6:
        errSum = errSum/(experimentComp.experiment.experimentCondition.feedVolume * experimentComp.feedConcentration)
    elif factor == 7:
        errSum = errSum/((experimentComp.experiment.experimentCondition.feedVolume * experimentComp.feedConcentration)**2)
    errSum = math.log(errSum)
    return errSum
