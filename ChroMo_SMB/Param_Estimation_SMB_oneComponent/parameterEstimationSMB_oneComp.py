from scipy.optimize import minimize
from objectiveSMB_oneComp import objectiveSMB
import gc

gc.collect()

# Global variable to track objective function calls
number_of_obj_calls = 0

def parameterEstimationSMB(experimental_data_path, end_time, initial_guess, bounds, component='A'):
    """
    Optimizes Henry constants and Delta parameters for a single component (A or B) using Nelder-Mead optimization.

    Args:
        experimental_data_path (str): Path to experimental data Excel file.
        end_time (float): Total simulation time.
        initial_guess (list): Initial guess for [henry_constant, delta].
        bounds (list): Bounds for each parameter [(min_henry, max_henry), (min_delta, max_delta)].
        component (str): 'A' to optimize Component A, 'B' for Component B.

    Returns:
        dict: Optimized parameters and MSE.
    """
    def objective_function(params):
        henry_constant, delta, porosity = params

        global number_of_obj_calls
        number_of_obj_calls += 1
        print (f"  Total number of calls: {number_of_obj_calls}")

        if component == 'Man':
            mse = objectiveSMB(end_time=end_time,
                               experimental_data_path=experimental_data_path,
                               porosity=porosity,
                               henry_constantA=henry_constant, deltaA=delta,
                               optimize_component='Man')
        else:
            mse = objectiveSMB(end_time=end_time,
                               experimental_data_path=experimental_data_path,
                               porosity=porosity,
                               henry_constantB=henry_constant, deltaB=delta,
                               optimize_component='Gal')
        return mse

    # Run optimization using Nelder-Mead
    result = minimize(objective_function, initial_guess, method='Nelder-Mead',
                      bounds=bounds, options={'maxiter': 5000, 'fatol': 1e-6, 'disp': True})

    # Extract optimized parameters
    optimized_params = {
        "henry_constant": result.x[0],
        "delta": result.x[1],
        "porosity": result.x[2],
        "MSE": result.fun
    }

    print(f"Optimization Results for Component {component}:", optimized_params)
    return optimized_params

# Example usage
# porosity=0.05, henry_constantA=0.58, deltaA=300
experimental_data_path = "SMB_onePeriond_experiment5_RI.xlsx"
end_time = 11705
initial_guess_porosity = 0.05
initial_guess_A = [0.58, 300, initial_guess_porosity]  # Initial Henry & Delta values for Component A
initial_guess_B = [0.4855, 30.3856, initial_guess_porosity]  # Initial Henry & Delta values for Component B
bounds = [(0.1, 10), (5, 10000), (0.005, 0.9)]  # Bounds for each parameter

optimized_params_A = parameterEstimationSMB(experimental_data_path, end_time, initial_guess_A, bounds, component='Man')
#optimized_params_B = parameterEstimationSMB(experimental_data_path, end_time, initial_guess_B, bounds, component='Gal')
