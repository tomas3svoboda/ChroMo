from collections import OrderedDict
from functions.solvers.Lin_Solver import Lin_Solver
from functions.solvers.Nonlin_Solver import Nonlin_Solver
import math
from scipy.interpolate import interp1d
import time
import pandas as pd
from datetime import timedelta
import numpy as np


class LossFunctionWrapper:
    def __init__(self, component_names, is_linear=True,):
        self.component_names = component_names  # Names of all components, order important
        self.is_linear = is_linear  # Linear/nonlinear flag
        self.start_time = time.time()  # Start time when instance is created
        self.min_objective_value = float('inf')  # Initialize with infinity

        # Initialize the DataFrame columns based on linearity and component names
        columns = ['Elapsed Time', 'Objective Value', 'D_corr']
        if is_linear:
            columns += [f'K_{name}' for name in component_names]
        else:
            columns += [f'L_{name}' for name in component_names] + [f'S_{name}' for name in component_names]
        self.details_df = pd.DataFrame(columns=columns)

    def __call__(self, params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms, uncertainties):
        if self.is_linear:
            value = Lin_loss_function_flat(params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms, uncertainties)
        else:
            value = Nonlin_loss_function_flat(params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms, uncertainties)

        # Update the running minimum objective value if the current value is lower
        self.min_objective_value = min(self.min_objective_value, value)

        # Compute and format elapsed time
        elapsed_time = time.time() - self.start_time
        elapsed_time_str = str(timedelta(seconds=round(elapsed_time)))

        # Format parameters for printing
        params_str = " ".join([f"{param:.4f}" for param in params])

        # Update the DataFrame with the current optimization step details
        self.details_df = update_details_df(params, elapsed_time, value, self.details_df, self.component_names, self.is_linear)

        # Print current and minimum objective value so far
        print(f"Time: {elapsed_time_str}, Parameters: [{params_str}], Objective value: {value:.6f} (Min: {self.min_objective_value:.6f})")

        return value

def Lin_loss_function_flat(params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms, uncertainties):
    dispCorrelParam = params[-1]  # Universal dispersion parameter at the end
    porosity = 0.3752
    results = []
    errSum = 0

    # Use OrderedDict.fromkeys() to remove duplicates while preserving order
    component_names = list(OrderedDict.fromkeys(component_names_all))
    # Assuming one optimized parameter per component
    component_param_index = {name: i for i, name in enumerate(component_names)}  # Map component to its parameter index

    for i, component_name in enumerate(component_names_all):
        param_index = component_param_index[component_name]  # Get the index for the component's parameter
        henry_constant = params[param_index]  # Fetch the Henry's constant for the component
        flowSpeed = (flow_rates[i] * 1000/3600) / ((math.pi * (diameters[i]**2) / 4) * porosity)
        disperCoef = (1/dispCorrelParam) * lengths[i] * flowSpeed + 0.00001
        model = Lin_Solver(flow_rates[i],
                        lengths[i],
                        diameters[i],
                        feed_volumes[i],
                        concentrations[i],
                        porosity,  # porosity
                        henry_constant,  # Henry's constant
                        disperCoef,  # Dispersion
                        100,
                        3000,
                        10800,
                        debugPrint=False,
                        full=False)

        # Error calculation as a sum of squared errors finally divided by max output concentration
        df = chromatograms[i]
        modelCurve = model[0][:, -1]
        time = model[1]
        f = interp1d(time, modelCurve, fill_value="extrapolate")
        modelCurveInterpolated = f(df.iloc[:, 0].to_numpy())
        tmpErrSum = 0
        max = 0
        for a, b in zip(df.iloc[:, 1].to_numpy(), modelCurveInterpolated):
            err = ((a-b)**2)
            tmpErrSum += err
            if a > max:
                max = a
        tmpErrSum = tmpErrSum/(max**2)/(uncertainties[i] * np.sqrt(df.shape[0]))
        errSum += tmpErrSum
    return errSum

def Nonlin_loss_function_flat(params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms, uncertainties):
    dispCorrelParam = params[-1]  # Universal dispersion parameter at the end
    porosity = 0.3752
    errSum = 0

    # Use OrderedDict.fromkeys() to remove duplicates while preserving order
    component_names = list(OrderedDict.fromkeys(component_names_all))
    # Assuming two parameters per component for the nonlinear case
    component_param_index = {name: i*2 for i, name in enumerate(component_names)}  # Map component to its parameter indices

    for i, component_name in enumerate(component_names_all):
        param_start_index = component_param_index[component_name]  # Get the start index for the component's parameters
        # Fetch the component-specific parameters for the Nonlin_Solver
        parameter1 = params[param_start_index]
        parameter2 = params[param_start_index + 1]  # The second parameter for the nonlinear model

        flowSpeed = (flow_rates[i] * 1000/3600) / ((math.pi * (diameters[i]**2) / 4) * porosity)
        disperCoef = (1/dispCorrelParam) * lengths[i] * flowSpeed + 0.00001

        # Assuming Nonlin_Solver takes an additional parameter (parameter2) for the nonlinear case
        model = Nonlin_Solver(flow_rates[i],
                              lengths[i],
                              diameters[i],
                              feed_volumes[i],
                              concentrations[i],
                              porosity,  # porosity
                              parameter1,  # First component-specific parameter
                              parameter2,  # Second component-specific parameter for nonlinear case
                              disperCoef,  # Dispersion
                              100,
                              3000,
                              10800,
                              debugPrint=False,
                              full=False)

        # Error calculation follows a similar process as in Lin_loss_function_flat
        df = chromatograms[i]
        modelCurve = model[0][:, -1]
        time = model[1]
        f = interp1d(time, modelCurve, fill_value="extrapolate")
        modelCurveInterpolated = f(df.iloc[:, 0].to_numpy())
        tmpErrSum = 0
        max = 0
        for a, b in zip(df.iloc[:, 1].to_numpy(), modelCurveInterpolated):
            err = ((a-b)**2)
            tmpErrSum += err
            if a > max:
                max = a
        tmpErrSum = tmpErrSum/(max**2)/(uncertainties[i] * np.sqrt(df.shape[0]))
        errSum += tmpErrSum
    return errSum

def update_details_df(params, elapsed_time, objective_value, details_df, component_names, is_linear):
    # Prepare data for the new row with common information
    data = {'Elapsed Time': [elapsed_time], 'Objective Value': [objective_value], 'D_corr': [params[-1]]}
    num_components = len(component_names)

    # Add parameter-specific data based on the case (linear or nonlinear)
    if is_linear:
        for i, name in enumerate(component_names):
            data[f'K_{name}'] = [params[i]]
    else:
        for i, name in enumerate(component_names):
            # For nonlinear cases, assume params are ordered as [L_A, L_B, ..., S_A, S_B, ...]
            data[f'L_{name}'] = [params[i]]
            data[f'S_{name}'] = [params[i + num_components]]

    # Create a new DataFrame for the row to add
    new_row_df = pd.DataFrame(data)

    # Use pandas.concat to add the new row to the existing DataFrame
    details_df = pd.concat([details_df, new_row_df], ignore_index=True)
    return details_df
