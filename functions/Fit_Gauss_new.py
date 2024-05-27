import pandas as pd
import numpy as np
from scipy.optimize import leastsq
from scipy.special import erf

def enforce_minimum_time_step(comp, result, min_time_step=0.2, conc_threshold=1e-5):
    """
    Reduces density of points when the change in concentration is very low.

    Args:
    result (pd.DataFrame): DataFrame containing 'Time' and 'Suc' columns.
    min_time_step (float): Minimum allowed time step between points.
    conc_threshold (float): Threshold below which the concentration change is considered negligible.

    Returns:
    pd.DataFrame: Updated DataFrame with reduced point density where applicable.
    """
    # Calculate changes in concentration
    conc_changes = np.abs(np.diff(result[comp.name]))
    time_steps = np.diff(result['Time'])

    # Identify points to keep
    keep = [True]  # Always keep the first point
    last_time = result['Time'][0]

    for i in range(1, len(result)):
        if conc_changes[i - 1] > conc_threshold or (result['Time'][i] - last_time >= min_time_step):
            keep.append(True)
            last_time = result['Time'][i]
        else:
            keep.append(False)

    # Filter the DataFrame
    filtered_result = result[keep].reset_index(drop=True)
    return filtered_result

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
            # Existing setup
            data_set = comp.concentrationTime.to_numpy()
            data_set[:, 0] = data_set[:, 0] / 60  # Convert time to minutes if necessary
            max_time = data_set[-1, 0]
            max_conc = max(data_set[:, 1])
            max_conc_index = data_set[:, 1].tolist().index(max_conc)

            # Prepend the zero concentration at time zero
            zero_point = np.array([[0, 0]])  # Create a new array with time=0 and conc=0
            data_set = np.vstack((zero_point, data_set))  # Stack it on top of the existing data

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
            print('INIT!!!!  ' + str(initials))
            n_value = len(initials)

            solution = leastsq(residuals, initials, args=(data_set[:, 1], data_set[:, 0], n_value), maxfev=10000, full_output=1)
            const = solution[0]

            #___________________________________________________________________________________________________________
            # UNCERNTAINTY
            res = solution[2]['fvec']  # raw residuals
            # Scale residuals by the maximum absolute value of observed data
            max_abs_value = np.max(np.abs(data_set[:, 1])) / multiplier
            #scaled_residuals = res / max_abs_value

            # Calculate the standard deviation of the scaled residuals and add to preprocessingScore
            uncertainty = np.std(res) / max_abs_value
            #uncertainty = np.mean(np.abs(res)) / max_abs_value
            #uncertainty = np.var(res) / (max_abs_value ** 2)
            #uncertainty = np.std(res) / np.var(res)

            print(str(uncertainty))
            #print('Exp, comp:' + str(exp.metadata.description) + ', ' + str(comp.name) + '; FIT GAUSS UNCERTAINTY VALUE: ' + str(uncertainty))
            #print('-----preprocessing score before: ' + str(comp.preprocessingScore))
            comp.preprocessingScore += uncertainty
            #print('-----preprocessing score after: ' + str(comp.preprocessingScore))
            #___________________________________________________________________________________________________________

            # Generate initial high-resolution Gaussian data
            high_res_time = np.linspace(0, max_time, 1000 * len(data_set[:, 0]))
            high_res_gauss_data = GaussSum(high_res_time, const, n_value) / multiplier

            # DEFINE NUMBER OF POINTS (can be changed when enforce_minimum_time_step is called)
            total_points = 60

            # Dynamically allocate time points based on the normalized slope
            slopes = np.abs(np.gradient(high_res_gauss_data, high_res_time)) + 8e-3
            normalized_slopes = slopes / np.sum(slopes)
            cumulative_slopes = np.cumsum(normalized_slopes)
            all_time_points = np.interp(np.linspace(0, cumulative_slopes[-1], total_points), cumulative_slopes, high_res_time)

            # Calculate the Gaussian data at these time points
            all_gauss_data = GaussSum(all_time_points, const, n_value) / multiplier

            # Create the DataFrame with the final high-resolution time and Gaussian data
            result = pd.DataFrame({
                'Time': all_time_points * 60,  # Convert time from minutes to seconds
                comp.name: all_gauss_data
            })

            # Identify the largest jump in the result DataFrame
            jumps = np.abs(np.diff(result[comp.name]))
            max_jump_index = np.argmax(jumps)
            if jumps[max_jump_index] > (max_conc/2/multiplier):
                # Add interpolated points at this jump
                result = add_interpolated_points(comp, result, max_jump_index, num_points=6)

            result = enforce_minimum_time_step(comp, result, min_time_step=20, conc_threshold=1e-5)
            # Update the component's concentrationTime with the new DataFrame
            comp.concentrationTime = result.sort_values(by='Time').reset_index(drop=True)

    return experimentSetGauss
