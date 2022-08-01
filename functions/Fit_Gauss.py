import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.optimize import leastsq
from scipy.special import erf
from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet

def Fit_Gauss(experimentSetGauss):
    print('Fitting Gauss started!')
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
    print('Fitting Gauss started!')
    for exp in experimentSetGauss.experiments:
        print('For Loop Started')
        for comp in exp.experimentComponents:
            print(comp.concentrationTime)
            data_set = comp.concentrationTime.to_numpy()
            print(data_set)
            max_time = data_set[-1, 0]
            max_conc = max(data_set[:, 1])
            max_conc_index = data_set[:, 1].index(max_conc)

            initials = [[max_conc, data_set[max_conc_index, 0], 1.0, 0.0]]
            n_value = len(initials)

            # executes least-squares regression analysis to optimize initial parameters
            const = leastsq(residuals, initials, args=(data_set[:, 1], data_set[:, 0], n_value))[0]

            # integrates the gaussian functions through gauss quadrature and saves the
            # results to a dictionary.

            areas = dict()
            for i in range(n_value):
                areas[i] = quad(gaussian, data_set[0, 0], data_set[-1, 0], args=(const[4*i], const[4*i+1], const[4*i+2], const[4*i+3]))[0]
            time = np.linspace(0, max_time, 200)
            result = pd.DataFrame({'time': time, comp.name: GaussSum(time, const, n_value)})
            comp.concentrationTime = result
            print(result)

    return experimentSetGauss


