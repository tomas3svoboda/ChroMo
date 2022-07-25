import numpy as np
from scipy.integrate import quad
from scipy.optimize import leastsq
from scipy.special import erf

# defines a typical gaussian function, of independent variable x,
# amplitude a, position b, width parameter c, and erf parameter d.
def gaussian(x, a, b, c, d):
    amp = (a / (c * np.sqrt(2 * np.pi)))
    spread = np.exp((-(x - b) ** 2.0) / 2 * c ** 2.0)
    skew = (1 + erf((d * (x - b)) / (c * np.sqrt(2))))
    return amp * spread * skew

# defines the expected resultant as a sum of intrinsic gaussian functions
def GaussSum(x, p, n):
    gs = sum(
        gaussian(x, p[4*k], p[4*k+1], p[4*k+2], p[4*k+3])
        for k in range(n)
    )
    return gs

# defines a residual, which is the  reducing the square of the difference
# between the data and the function
def residuals(p, y, x, n):
    return y - GaussSum(x,p,n)

# Main function body
def Fit_Gauss(experimentSetCor1):

    experimentSetGauss = experimentSetCor1

    initials = [4.5, 19, 1, 0]

    n_value = len(initials)

    for experiment in experimentSetCor1.experiments:
        for component in experiment.experimentComponents:
            data_set = component.concentrationTime.to_numpy()

        # executes least-squares regression analysis to optimize initial parameters
            cnst = leastsq(
                residuals,
                initials,
                args=(
                    data_set[:,1],
                    data_set[:,0],
                    n_value
                    )
            )[0]

            # integrates the gaussian functions through gauss quadrature and saves the
            # results to a dictionary.

            areas = dict()

            for i in range(n_value):
                areas[i] = quad(
                    gaussian,
                    data_set[0,0],      # lower integration bound
                    data_set[-1,0],     # upper integration bound
                    args=(
                        cnst[4*i],
                        cnst[4*i+1],
                        cnst[4*i+2],
                        cnst[4*i+3]
                        )
                    )[0]

            t = np.linspace(0,40,200)
            concentration = GaussSum(t,cnst, n_value)
            experimentSetGauss.experiment.component.concetrationTime = [t, concentration]

    return experimentSetGauss
