import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.optimize import leastsq
from scipy.special import erf
from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet

def Fit_Gauss(experimentSetGauss):
    print('Fitting Gauss started!')
    # ---------------------Start of external code-------------------------------
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
    # ---------------------External code---------------------------------------
    for exp in experimentSetGauss.experiments:
        for comp in exp.experimentComponents:
            data_set = comp.concentrationTime.to_numpy()
            data_set[:, 0] = data_set[:, 0]/60
            max_time = data_set[-1, 0]
            print(max_time)
            max_conc = max(data_set[:, 1])
            print(max_conc)
            max_conc_index = data_set[:, 1].tolist().index(max_conc)

            if max_conc < max_time / 5:
                mutiplier = 10
                data_set[:, 1] = 10 * data_set[:, 1]
                max_conc = 10 * max_conc
            elif max_conc < max_time / 50:
                mutiplier = 100
                data_set[:, 1] = 100 * data_set[:, 1]
                max_conc = 100 * max_conc
            elif max_conc < max_time / 500:
                mutiplier = 1000
                data_set[:, 1] = 1000 * data_set[:, 1]
                max_conc = 1000 * max_conc
            elif max_conc < max_time / 5000:
                mutiplier = 10000
                data_set[:, 1] = 10000 * data_set[:, 1]
                max_conc = 10000 * max_conc
            elif max_conc < max_time / 50000:
                mutiplier = 100000
                data_set[:, 1] = 100000 * data_set[:, 1]
                max_conc = 100000 * max_conc
            else:
                mutiplier = 1

            init = data_set[max_conc_index, 0] + ((data_set[max_conc_index, 0]-data_set[max_conc_index-1, 0])/3)
            initials = [[max_conc, init, 0.4, 0.0]]
            n_value = len(initials)

            # executes least-squares regression analysis to optimize initial parameters
            const = leastsq(residuals, initials, args=(data_set[:, 1], data_set[:, 0], n_value))[0]

            # integrates the gaussian functions through gauss quadrature and saves the
            # results to a dictionary.

            #---------------------------- Start of External code--------------------------
            areas = dict()
            for i in range(n_value):
                areas[i] = quad(gaussian, data_set[0, 0], data_set[-1, 0], args=(const[4*i], const[4*i+1], const[4*i+2], const[4*i+3]))[0]
            #---------------------------- End of External code--------------------------

            time = (np.linspace(0, max_time, 80))
            gauss_data = GaussSum(time, const, n_value)/mutiplier
            time_red = (np.linspace(0, max_time, 6))
            n = 0
            for i in gauss_data:
                if i > (max_conc/30):
                    time_red = np.append(time_red, (time[n]))
                n += 1
            time = np.sort(time)

            comp_name = comp.name
            result = pd.DataFrame({'time': time_red, comp_name: ((GaussSum(time_red, const, n_value))/mutiplier)})
            result = result.sort_values(by = ['time'])
            result['time'] *= 60

            #-----------------temporary solution---------------------
            if comp.name == "ManOH":
                result.drop(result[result['time'] < 0].index, inplace=True)



            # -----------------temporary solution---------------------
            #result = result.drop((result[result[comp_name] < (max_conc/30)].index))
            #result = result.drop((result[(result[comp_name]<(max_conc/30)) and (not ((result['time'] % 100) == 0))].index))
            #result.loc[0] = [0,0]
            #result.loc[len(result.index)] = [max_time,data_set[-1, 1]]
            #result = result.dropna()

            comp.concentrationTime = result

    return experimentSetGauss


