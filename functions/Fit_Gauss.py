import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.optimize import leastsq
from scipy.special import erf
from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet

def Fit_Gauss(experimentSet):

    # defines a typical gaussian function, of independent variable x,
    # amplitude a, position b, width parameter c, and erf parameter d.
    def gaussian(x, a, b, c, d):
        amp = (a / (c * np.sqrt(2 * np.pi)))
        spread = np.exp((-(x - b) ** 2.0) / 2 * c ** 2.0)
        skew = (1 + erf((d * (x - b)) / (c * np.sqrt(2))))
        return amp * spread * skew

    # defines the expected resultant as a sum of intrinsic gaussian functions
    def GaussSum(x, p, n):
        gs = sum(gaussian(x, p[4*k], p[4*k+1], p[4*k+2], p[4*k+3])for k in range(n))
        return gs

    # defines a residual, which is the  reducing the square of the difference
    # between the data and the function
    def residuals(p, y, x, n):
        return y - GaussSum(x, p, n)

    for experiment in experimentSet.experiments:
        for component in experiment.experimentComponents:
            initials = [[6.5, 13.0, 1.0, 0.0]]
            n_value = len(initials)
            data_set = component.concentrationTime.to_numpy()

            # executes least-squares regression analysis to optimize initial parameters
            const = leastsq(residuals, initials, args=(data_set[:, 1], data_set[:, 0], n_value))[0]

            # integrates the gaussian functions through gauss quadrature and saves the
            # results to a dictionary.

            areas = dict()
            for i in range(n_value):
                areas[i] = quad(gaussian, data_set[0, 0], data_set[-1, 0], args=(const[4*i], const[4*i+1], const[4*i+2], const[4*i+3]))[0]

            result = pd.DataFrame()
            result.loc[:, 0] = np.linspace(0, max(data_set[:, 0]), 200)
            result.loc[:, 1] = GaussSum(result.iloc[:, 0], const, n_value)
            component.concentrationTime = result

    return experimentSet


