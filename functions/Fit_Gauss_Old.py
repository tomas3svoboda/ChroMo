import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.optimize import leastsq
from scipy.special import erf
from scipy.interpolate import interp1d
from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet

def Fit_Gauss(experimentSetGauss):
    """Defines a typical gaussian function, of independent variable x,
    amplitude a, position b, width parameter c, and erf parameter d.
    """
    #print('Fitting Gauss started!')

    # ---------------------Start of external code-------------------------------
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

    # ---------------------End of external code---------------------------------------

    for exp in experimentSetGauss.experiments:
        for comp in exp.experimentComponents:
            data_set = comp.concentrationTime.to_numpy()
            data_set[:, 0] = data_set[:, 0]/60
            max_time = data_set[-1, 0]
            max_conc = max(data_set[:, 1])
            max_conc_index = data_set[:, 1].tolist().index(max_conc)

            # Multiplier based on absolute values of concentrations ensures proper function of external code below

            # Initialize multiplier
            multiplier = 1

            # Define the scaling thresholds
            scaling_factors = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

            # Loop through scaling factors in reverse (to start with the largest)
            for factor in reversed(scaling_factors):
                if max_conc < max_time / (factor * 10):
                    multiplier = factor
                    print('MULTIPLIER: ' + str(multiplier))
                    break

            # Apply the multiplier to the dataset and max_conc
            data_set[:, 1] *= multiplier
            max_conc *= multiplier

            init = data_set[max_conc_index, 0] + ((data_set[max_conc_index, 0]-data_set[max_conc_index-1, 0])/3)
            initials = [[max_conc, init, 0.4, 0.0]]
            n_value = len(initials)

            #---------------------------- Start of External code--------------------------

            # executes least-squares regression analysis to optimize initial parameters
            const = leastsq(residuals, initials, args=(data_set[:, 1], data_set[:, 0], n_value), maxfev=10000)[0]
            #print ('For experiment: '+ str(exp) + '; and component: ' + str(comp.name) + '; Parameters are: ' + str(const))

            # integrates the gaussian functions through gauss quadrature and saves the
            # results to a dictionary.
            #areas = dict()
            #for i in range(n_value):
                #areas[i] = quad(gaussian, data_set[0, 0], data_set[-1, 0], args=(const[4*i], const[4*i+1], const[4*i+2], const[4*i+3]))[0]
            #---------------------------- End of External code--------------------------

            # Assuming GaussSum, const, n_value, and multiplier are defined
            # Assuming comp.name is defined and comp is an object that has a 'name' attribute

            # Initial time array
            time = np.linspace(0, max_time, 60)
            gauss_data = GaussSum(time, const, n_value) / multiplier

            # Identify the peak index and the value
            peak_index = gauss_data.argmax()
            peak_value = gauss_data[peak_index]

            # Thresholds for peak detection
            peak_start_threshold = 0.05 * peak_value  # Adjust this as needed for the start of the peak
            peak_end_threshold = 0.05 * peak_value    # Adjust this as needed for the end of the peak

            # Find where the peak starts and ends
            rising_edge_index = np.where(gauss_data > peak_start_threshold)[0][0]
            falling_edge_index = np.where(gauss_data > peak_end_threshold)[0][-1]

            # Allocate time points for three density levels
            densest_points = 20  # More points at the peak
            intermediate_points = 15  # Intermediate number of points for the falling edge
            less_dense_points = 5  # Few points for the baseline

            # Generate time points for each region
            time_densest = np.linspace(time[rising_edge_index], time[peak_index], densest_points)
            time_intermediate = np.linspace(time[peak_index], time[falling_edge_index], intermediate_points)
            time_less_dense_start = np.linspace(time[0], time[rising_edge_index], less_dense_points, endpoint=False)
            time_less_dense_end = np.linspace(time[falling_edge_index], time[-1], less_dense_points, endpoint=False)

            # Combine and sort the time arrays
            time_red = np.sort(np.concatenate((time_less_dense_start, time_densest, time_intermediate, time_less_dense_end)))

            # Remove potential duplicate time points
            time_red = np.unique(time_red)

            # Interpolate to find concentration values at new time points using the original GaussSum function
            comp_name = comp.name
            result = pd.DataFrame({
                'Time': time_red,
                comp_name: (GaussSum(time_red, const, n_value) / multiplier)
            })

            # Sort by time and adjust units if necessary
            result = result.sort_values(by=['Time'])
            result['Time'] *= 60  # Convert to seconds if needed

            # The DataFrame 'result' now has time points with the specified three levels of density

            #-----------------temporary solution---------------------

            if comp.name == "ManOH":
                result.drop(result[result['Time'] < 0].index, inplace=True)

            # -----------------temporary solution---------------------
            #result = result.drop((result[result[comp_name] < (max_conc/30)].index))
            #result = result.drop((result[(result[comp_name]<(max_conc/30)) and (not ((result['Time'] % 100) == 0))].index))
            #result.loc[0] = [0,0]
            #result.loc[len(result.index)] = [max_Time,data_set[-1, 1]]
            #result = result.dropna()

            comp.concentrationTime = result.reset_index(drop=True)

    return experimentSetGauss


