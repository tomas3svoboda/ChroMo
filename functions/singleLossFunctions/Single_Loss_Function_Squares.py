from functions.solvers.Solver_Choice import Solver_Choice
from scipy.interpolate import interp1d
import numpy as np
# Calculates loss value of sigle particular solution and according time series. Serves for isotherm decision.

def Single_Loss_Function_Squares(params, experimentComp, solver):
    errSum = 0
    df = experimentComp.concentrationTime
    modelCurve = Solver_Choice(solver, params, experimentComp)[:, -1]
    # TODO remove hard wired time values (need to fix in solvers?)
    minTime = df.iat[0, 0]
    maxTime = df.iat[-1, 0]
    time = np.linspace(minTime, maxTime, 3000)
    f = interp1d(time, modelCurve)
    modelCurveInterpolated = f(df.iloc[:, 0].to_numpy())
    tmpErrSum = 0
    max = 0
    for a, b in zip(df.iloc[:, 1].to_numpy(), modelCurveInterpolated):
        err = ((a-b)**2)
        tmpErrSum += err
        if a > max:
            max = a
    errSum += tmpErrSum
    errSum = errSum/(max**2)
    return errSum
