from functions.solvers.Solver_Choice import Solver_Choice
from scipy.interpolate import interp1d
import math
import numpy as np

def Single_Loss_Function_LogSquares(params, experimentComp, solver, factor, spacialDiff, timeDiff, time):
    """Calculates a loss value, representing difference between experimental and model data with given parameters.
    Summed values are squared.
    Natural log is applied to final value."""
    errSum = 0
    df = experimentComp.concentrationTime
    model = Solver_Choice(solver, params, experimentComp, spacialDiff, timeDiff, time)
    modelCurve = model[0][:, -1]
    time = model[1]
    f = interp1d(time, modelCurve, fill_value="extrapolate")
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
