from functions.solvers.Lin_Solver import Lin_Solver
from functions.solvers.Nonlin_Solver import Nonlin_Solver
import matplotlib.pyplot as plt
import math



def plot_component_results_lin(component_names, optimized_params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms):
    for component_name in component_names:
        param_index = component_names.index(component_name)
        henry_constant = optimized_params[param_index]
        dispCorrelParam = optimized_params[-1]

        plt.figure(figsize=(10, 6))  # Corrected position

        for i, comp_name in enumerate(component_names_all):
            if comp_name == component_name:
                flowSpeed = (flow_rates[i] * 1000/3600) / ((math.pi * (diameters[i]**2) / 4) * 0.3752)
                disperCoef = (1/dispCorrelParam) * lengths[i] * flowSpeed + 0.00001
                model = Lin_Solver(flow_rates[i], lengths[i], diameters[i], feed_volumes[i], concentrations[i], 0.3752, henry_constant, disperCoef, 100, 3000, 10800, debugPrint=False, full=True)

                time, modelCurve = model[0], model[1]
                df = chromatograms[i]
                exp_time = df.iloc[:, 0].to_numpy()
                exp_concentration = df.iloc[:, 1].to_numpy()

                plt.plot(time, modelCurve, label=f'Model Prediction for Exp {i+1}', linewidth=2)
                plt.scatter(exp_time, exp_concentration, label=f'Experimental Data for Exp {i+1}', marker='o', alpha=0.7)

        plt.title(f"Component {component_name} - Model vs. Experimental Data")
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.show()  # Moved outside the inner loop

def plot_component_results_nonlin(component_names, optimized_params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms):
    """
    Plots the results for each component using the nonlinear solver, comparing model predictions against experimental data.

    Parameters:
    - component_names: List of unique component names.
    - optimized_params: List of optimized parameters from the minimization process.
    - component_names_all: List of all component names corresponding to each experiment.
    - flow_rates, diameters, lengths, feed_volumes, concentrations: Experimental conditions.
    - chromatograms: Experimental data for comparison.
    """
    for component_name in component_names:
        # For the nonlinear case, each component has two specific parameters.
        param_index = component_names.index(component_name) * 2  # Adjust index for two parameters per component
        parameter1 = optimized_params[param_index]
        parameter2 = optimized_params[param_index + 1]
        dispCorrelParam = optimized_params[-1]  # Universal dispersion parameter at the end

        plt.figure(figsize=(10, 6))

        for i, comp_name in enumerate(component_names_all):
            if comp_name == component_name:
                # Setup model with Nonlin_Solver using two component-specific parameters and a dispersion coefficient
                flowSpeed = (flow_rates[i] * 1000 / 3600) / ((math.pi * (diameters[i] ** 2) / 4) * 0.3752)
                disperCoef = (1 / dispCorrelParam) * lengths[i] * flowSpeed + 0.00001
                model = Nonlin_Solver(flow_rates[i], lengths[i], diameters[i], feed_volumes[i], concentrations[i], 0.3752, parameter1, parameter2, disperCoef, 100, 3000, 10800, debugPrint=False, full=True)

                time, modelCurve = model[0], model[1]

                df = chromatograms[i]
                exp_time = df.iloc[:, 0].to_numpy()
                exp_concentration = df.iloc[:, 1].to_numpy()

                plt.plot(time, modelCurve, label='Model Prediction', color='blue', linewidth=2)
                plt.scatter(exp_time, exp_concentration, label='Experimental Data', color='red', marker='o')
                plt.title(f"Component {component_name} - Model vs. Experimental Data")
                plt.xlabel('Time')
                plt.ylabel('Concentration')
                plt.legend()
                plt.show()
