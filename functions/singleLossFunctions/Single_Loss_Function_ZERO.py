from functions.solvers.Solver_Choice import Solver_Choice
from scipy.interpolate import interp1d
import numpy as np
# Calculates loss value of sigle particular solution and according time series. Serves for isotherm decision.

def Single_Loss_Function_Squares(params, experimentComp, solver, factor, spacialDiff, timeDiff, time):
    """Calculates a loss value, representing difference between experimental and model data with given parameters.
    Summed values are squared."""
    errSum = 0
    df = experimentComp.concentrationTime
    model = Solver_Choice(solver, params, experimentComp, spacialDiff, timeDiff, time)
    modelCurve = model[0][:, -1]
    # replace model curve by ZEROS!!!!!!!!!!!!!!!!!!!!!!!!!!!
    modelCurve = np.zeros(modelCurve.shape)
    time = model[1]
    f = interp1d(time, modelCurve, fill_value="extrapolate")
    modelCurveInterpolated = f(df.iloc[:, 0].to_numpy())
    tmpErrSum = 0
    max_a = 0
    max_b = 0

    for a, b in zip(df.iloc[:, 1].to_numpy(), modelCurveInterpolated):
        err = ((a-b)**2)
        tmpErrSum += err
        '''if (factor == 2 or factor == 3) and b > max_b:
            max_b = b'''
        if (factor == 2 or factor == 3) and a > max_a:
            max_a = a

    '''if max_a > max_b:
        max = max_a
    else:
        max = max_b'''
    max = max_a
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

    # Division square root of the number of elements
    errSum = errSum / np.sqrt(len(df[experimentComp.name]))

    # WEIGHTING BY PREPROCESSING SCORE
    if experimentComp.preprocessingScore > 0:
        errSum = errSum / experimentComp.preprocessingScore

    return errSum
