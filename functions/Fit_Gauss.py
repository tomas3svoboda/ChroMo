import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.optimize import leastsq
from scipy.special import erf
from scipy.interpolate import interp1d
from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet

'''def calculate_peak_width_at_inflections(comp_name, concentration_time_data):
    """
    Calculates the peak width based on significant inflection points, selecting the ones with the steepest slopes.

    Args:
    concentration_time_data (pd.DataFrame): DataFrame with 'Time' and 'Concentration' columns.

    Returns:
    float: Peak width based on significant inflection points, or 0 if not enough inflection points are found.
    """
    time = concentration_time_data['Time'].to_numpy()
    concentration = concentration_time_data[comp_name].to_numpy()

    # Calculate derivatives
    first_derivative = np.gradient(concentration, time)
    second_derivative = np.gradient(first_derivative, time)

    # Find inflection points based on the second derivative changing sign
    inflections = np.where(np.diff(np.sign(second_derivative)))[0]

    if len(inflections) < 2:
        print("Not enough inflection points found.")
        return 0

    # Determine the steepness of the slope at each inflection point
    slopes = np.abs(first_derivative[inflections])

    # Select indices of the two inflection points with the highest slope magnitude
    if len(slopes) > 1:
        significant_indices = np.argsort(-slopes)[:2]  # Sort descending and take top 2
        significant_inflections = inflections[significant_indices]
        # Ensure the lower index comes first
        significant_inflections = np.sort(significant_inflections)
        # Calculate the width as the time difference between these two points
        width = time[significant_inflections[1]] - time[significant_inflections[0]]
        return width
        print(str(comp_name) + ' width: ' + str(width))
    else:
        return 0  # Default to zero if not enough significant inflection points'''

'''def calculate_peak_width_at_inflections(comp_name, concentration_time_data):
    """
    Calculates the peak width based on significant inflection points, selecting the ones with the steepest slopes.

    Args:
    concentration_time_data (pd.DataFrame): DataFrame with 'Time' and 'Concentration' columns.

    Returns:
    float: Peak width based on significant inflection points, or 0 if not enough inflection points are found.
    """
    time = concentration_time_data['Time'].to_numpy()
    concentration = concentration_time_data[comp_name].to_numpy()

    # Calculate derivatives
    first_derivative = np.gradient(concentration, time)
    second_derivative = np.gradient(first_derivative, time)

    # Find inflection points based on the second derivative changing sign
    inflections = np.where(np.diff(np.sign(second_derivative)))[0]

    if len(inflections) < 2:
        print("Not enough inflection points found.")
        return 0

    # Determine the steepness of the slope at each inflection point
    slopes = np.abs(first_derivative[inflections])

    # Select indices of the two inflection points with the highest slope magnitude
    if len(slopes) > 1:
        significant_indices = np.argsort(-slopes)[:2]  # Sort descending and take top 2
        significant_inflections = inflections[significant_indices]
        # Ensure the lower index comes first
        significant_inflections = np.sort(significant_inflections)
        # Calculate the width as the time difference between these two points
        width = time[significant_inflections[1]] - time[significant_inflections[0]]
        print(str(comp_name) + ' width: ' + str(width))
        return width
    else:
        return 0  # Default to zero if not enough significant inflection points'''


def add_interpolated_points(comp, data, index, num_points=5):
    """
    Adds interpolated points between two rows of a DataFrame at a specified index.

    Args:
    data (DataFrame): The DataFrame containing time and concentration data.
    index (int): Index where the jump occurs.
    num_points (int): Number of interpolated points to add.

    Returns:
    DataFrame: Updated DataFrame with interpolated points added.
    """
    # Get the time and concentration at the jump
    time_start = data.at[index, 'Time']
    time_end = data.at[index + 1, 'Time']
    conc_start = data.at[index, comp.name]
    conc_end = data.at[index + 1, comp.name]

    # Create interpolated times and concentrations
    interp_times = np.linspace(time_start, time_end, num=num_points + 2)[1:-1]  # exclude the start/end points
    interp_concs = np.interp(interp_times, [time_start, time_end], [conc_start, conc_end])

    # Create a DataFrame of the interpolated points
    interp_data = pd.DataFrame({
        'Time': interp_times,
        comp.name: interp_concs
    })

    # Insert interpolated data into the original DataFrame
    part1 = data.iloc[:index+1]  # Up to and including the start point
    part2 = data.iloc[index+1:]  # From the end point onwards
    new_data = pd.concat([part1, interp_data, part2]).reset_index(drop=True)
    return new_data

def Fit_Gauss(experimentSetGauss):
    """Defines a typical gaussian function, of independent variable x,
    amplitude a, position b, width parameter c, and erf parameter d.
    """
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Fitting Gauss started!')

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
                    break

            # Apply the multiplier to the dataset and max_conc
            data_set[:, 1] *= multiplier
            max_conc *= multiplier

            init = data_set[max_conc_index, 0] + ((data_set[max_conc_index, 0]-data_set[max_conc_index-1, 0])/3)
            initials = [[max_conc, init, 0.4, 0.0]]
            n_value = len(initials)

            #---------------------------- Start of External code--------------------------

            # executes least-squares regression analysis to optimize initial parameters
            solution = leastsq(residuals, initials, args=(data_set[:, 1], data_set[:, 0], n_value), maxfev=10000, full_output=1)
            const = solution[0]
            #print ('For experiment: '+ str(exp) + '; and component: ' + str(comp.name) + '; Parameters are: ' + str(const))

            # integrates the gaussian functions through gauss quadrature and saves the
            # results to a dictionary.
            #areas = dict()
            #for i in range(n_value):
                #areas[i] = quad(gaussian, data_set[0, 0], data_set[-1, 0], args=(const[4*i], const[4*i+1], const[4*i+2], const[4*i+3]))[0]
            #---------------------------- End of External code--------------------------

            # Assuming GaussSum, const, n_value, and multiplier are defined
            # Assuming comp.name is defined and comp is an object that has a 'name' attribute

            high_res_time = np.linspace(0, max_time, 10000)
            high_res_gauss_data = GaussSum(high_res_time, const, n_value)

            # Identify the peak index and the value
            peak_index = high_res_gauss_data.argmax()
            peak_value = high_res_gauss_data[peak_index]
            peak_time = high_res_time[peak_index]

            # Thresholds for peak detection
            peak_start_threshold = 0.05 * peak_value  # Adjust this as needed for the start of the peak
            peak_end_threshold = 0.005 * peak_value    # Adjust this as needed for the end of the peak
            peak_top_threshold = peak_value - (0.05 * peak_value)
            peak_half = peak_value / 2

            # Find where the peak starts and ends
            peak_start = np.where(high_res_gauss_data > peak_start_threshold)[0][0]
            peak_end = np.where(high_res_gauss_data > peak_end_threshold)[0][-1]
            peak_top_start = np.where(high_res_gauss_data > peak_top_threshold)[0][0]
            peak_top_end = np.where(high_res_gauss_data > peak_top_threshold)[0][-1]
            peak_first_half = np.where(high_res_gauss_data > peak_half)[0][0]
            peak_second_half = np.where(high_res_gauss_data > peak_half)[0][-1]

            duration_start = (high_res_time[peak_start] - high_res_time[0])
            duration_rising = (high_res_time[peak_top_start] - high_res_time[peak_start])
            duration_top = (high_res_time[peak_top_end] - high_res_time[peak_top_start])
            duration_falling = (high_res_time[peak_end] - high_res_time[peak_top_end])
            duration_end = (high_res_time[-1] - high_res_time[peak_end])
            peak_half_width = high_res_time[peak_second_half] - high_res_time[peak_first_half]
            #print(peak_half_width)

            # Allocate time points for three density levels
            rising_points = 20  # Rising edge
            top_points = 4 # Around the top
            falling_points = 20  # Falling edge

            if high_res_time[peak_start] < (high_res_time[-1]/50):
                start_points = 2
            elif high_res_time[peak_start] < (high_res_time[-1]/25):
                start_points = 4
            elif high_res_time[peak_start] < (high_res_time[-1]/10):
                start_points = 6
            elif high_res_time[peak_start] < (high_res_time[-1]/5):
                start_points = 8
            elif high_res_time[peak_start] < (high_res_time[-1]/2):
                start_points = 10
            elif high_res_time[peak_start] < (high_res_time[-1]/1.5):
                start_points = 12
            else:
                start_points = 16

            if high_res_time[peak_end] > (high_res_time[-1]/1.005):
                end_points = 0  # Few points for the baseline after peak
            elif high_res_time[peak_end] > (high_res_time[-1]/1.05):
                end_points = 2  # Few points for the baseline after peak
            elif high_res_time[peak_end] > (high_res_time[-1]/1.5):
                end_points = 6  # Few points for the baseline after peak
            elif high_res_time[peak_end] > (high_res_time[-1]/2):
                end_points = 8  # Few points for the baseline after peak
            elif high_res_time[peak_end] > (high_res_time[-1]/3):
                end_points = 16
            elif high_res_time[peak_end] > (high_res_time[-1]/4):
                end_points = 24
            else:
                end_points = 32

            # Generate time points for each region
            time_start = np.linspace(high_res_time[0], high_res_time[peak_start], start_points, endpoint=False)
            time_rising = np.linspace(high_res_time[peak_start], high_res_time[peak_top_start], rising_points)
            time_top = np.linspace(high_res_time[peak_top_start], high_res_time[peak_top_end], top_points)
            time_falling = np.linspace(high_res_time[peak_top_end], high_res_time[peak_end], falling_points)
            time_end = np.linspace(high_res_time[peak_end], high_res_time[-1], end_points, endpoint=False)

            # Combine and sort the time arrays
            time_red = np.sort(np.concatenate((time_start, time_rising, time_top, time_falling, time_end)))

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


            # Identify the largest jump in the result DataFrame
            jumps = np.abs(np.diff(result[comp.name]))
            max_jump_index = np.argmax(jumps)
            if jumps[max_jump_index] > (max_conc/4/multiplier):
                # Add interpolated points at this jump
                result = add_interpolated_points(comp, result, max_jump_index, num_points=3)

            # The DataFrame 'result' now has time points with the specified three levels of density
            comp.concentrationTime = result
            print(comp.concentrationTime.shape[0])

            # PEAK WIDTH__________________________________________________________________________________________
            '''high_res_concentarionTime = pd.DataFrame({
                'Time': high_res_time,
                comp.name: (high_res_gauss_data / multiplier)
                })'''
            #comp.inflectionWidth = calculate_peak_width_at_inflections(comp.name, high_res_concentarionTime) * 60
            comp.inflectionWidth = peak_half_width * 60
            # __________________________________________________________________________________________________________

            # UNCERTAINTY_______________________________________________________________________________________________
            squared_residuals = solution[2]['fvec'] ** 2  # squared residuals
            mse = np.mean(squared_residuals) # mean of the squared errors
            rmse = np.sqrt(mse) # root mean square error

            # Calculate the standard deviation of the scaled residuals and add to preprocessingScore
            uncertainty = rmse / comp.feedConcentration # normalizing the value by concentration in feed
            #uncertainty = np.mean(np.abs(res)) / max_abs_value
            #uncertainty = np.var(res) / (max_abs_value ** 2)
            #uncertainty = np.std(res) / np.var(res)

            #print(str(uncertainty))
            #print('Exp, comp:' + str(exp.metadata.description) + ', ' + str(comp.name) + '; FIT GAUSS UNCERTAINTY VALUE: ' + str(uncertainty))
            comp.preprocessingScore = uncertainty
            #print('-----preprocessing score after: ' + str(comp.preprocessingScore))
            #___________________________________________________________________________________________________________

            # ZERO RESIDUALS____________________________________________________________________________________________

            #___________________________________________________________________________________________________________

    return experimentSetGauss


