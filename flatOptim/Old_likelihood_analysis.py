from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flatOptim.Loss_function_flat import LossFunctionWrapper
from flatOptim.plot_component_results import plot_component_results_lin
from flatOptim.plot_component_results import plot_component_results_nonlin
from objects.Operator import Operator
from datetime import timedelta
import time

operator_instance = Operator()

#this parts loads experiment set from the folder, creates all the data handling objects and also does preprocessing

path = 'C:/UserData/z004d8nt/Documents/VSCHT/001_Paper_JChA/Paper1_Data/Gal_Man_Omnilab/Bi_LEVEL_SPEED_testing'
experimentSet = operator_instance.Load_Experiment_Set(path)  #create object of the set of experiments
print('Starting preprocessing')
experimentSet = operator_instance.Preprocess(experimentSet, True, True, True, 0.005)  # does preprocessing
print('Preprocessing done')
experimentCluster = operator_instance.Cluster_By_Component(experimentSet).clusters  # creates new object where components are clustered together

'''object 'experimentCluster' is a dictionary of all the component names as a keys and experimentComp objects which hold
#chromatogram itself and information about the experiment setup, namely feed volume and concentration, column lenght,
#column diameter, flow rate, and commentary'''

i = 0
component_names_all = []
chromatograms = []
concentrations = []
feed_volumes = []
flow_rates = []
diameters = []
lengths = []

for comp_name, comp_objects in experimentCluster.items():
    # comp_objects is a list of ExperimentComponent objects
    for comp_object in comp_objects:  # Iterate over each ExperimentComponent object in the list
        chromatograms.append(comp_object.concentrationTime)  # Now you can access the concentrationTime attribute
        component_names_all.append(comp_object.name)
        feed_volumes. append(comp_object.experiment.experimentCondition.feedVolume)
        concentrations.append(comp_object.feedConcentration)
        flow_rates.append(comp_object.experiment.experimentCondition.flowRate)
        diameters.append(comp_object.experiment.experimentCondition.columnDiameter)
        lengths.append(comp_object.experiment.experimentCondition.columnLength)
        i += 1

component_names = list(dict.fromkeys(component_names_all))
num_of_components = len(component_names)

lin_compParams_dict = {} #  dictionary for the results of component specific params
nonlin_compParams_dict = {} #  dictionary for the results of component specific params

for component in component_names:
    lin_compParams_dict[component] = [1]
    nonlin_compParams_dict[component] = [1, 1]

flat_lin_params = []
flat_nonlin_params = []

for params in lin_compParams_dict.values():
    flat_lin_params.extend(params)

flat_lin_params.append(1)  # appending one more slot for dispersion correlation coefficient
# component params are ordered in alphabetical order

for params in nonlin_compParams_dict.values():
    flat_nonlin_params.extend(params)

flat_nonlin_params.append(1)  # appending one more slot for dispersion correlation coefficient

'''I will use least squares loss function which are relativized by diving the error value for experiment by feed concetration squared giving no-unit final value scaled by feed conc'''
# define initial guess and boundaries
# LINEAR
K_A, K_A_bounds = 4.359, (0.01, 100.0)
K_B, K_B_bounds = 7.943, (0.01, 100.0)
K_C, K_C_bounds = 3.038, (0.01, 100.0)
K_D, K_D_bounds = 4.432, (0.01, 100.0)
K_E, K_E_bounds = 5.993, (0.01, 100.0)
K_F, K_F_bounds = 8.753, (0.01, 100.0)
K_G, K_G_bounds = 3.858, (0.01, 100.0)
K_H, K_H_bounds = 6.623, (0.01, 100.0)
K_I, K_I_bounds = 2.679, (0.01, 100.0)
K_J, K_J_bounds = 3.845, (0.01, 100.0)
optimal_dispersion, disperCoef_bounds = 18.526, (0.01, 100.0)
optimal_params = [K_A,K_B,K_C,K_D,K_E,K_F,K_G,K_H,K_I,K_J]
bounds_lin = [K_A_bounds,K_B_bounds,K_C_bounds,K_D_bounds,K_E_bounds,K_F_bounds,K_G_bounds,K_H_bounds,K_I_bounds,K_J_bounds]
optimal_objective_value = 0.11318375

wrapper = LossFunctionWrapper(component_names, is_linear=True)

# Function to optimize other parameters with fixed dispersion
def optimize_with_fixed_dispersion(initial_guess, fixed_dispersion, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms):
    def objective(params_reduced, fixed_dispersion, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms):
        # Combine params_reduced with fixed_dispersion. If params_reduced is a list, this is straightforward:
        params = list(params_reduced) + [fixed_dispersion]
        # If params_reduced is a numpy array, you could use np.append instead:
        # params = np.append(params_reduced, fixed_dispersion)

        # Now pass this combined list to the wrapper
        return wrapper(params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms)

    # Note that we need to include fixed_dispersion as part of args since it's not a variable being optimized,
    # but still a parameter required by the objective function
    return minimize(objective,
                    initial_guess,
                    args=(fixed_dispersion, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms),
                    method='Nelder-Mead',
                    options={'disp': True, 'return_all': True, 'xatol': 1e-2, 'fatol': 1e-4})


# Define the range for the dispersion coefficient
dispersion_values = np.linspace(optimal_dispersion * 0.5, optimal_dispersion * 1.5, num=20)

# Find the index of the value in dispersion_values closest to optimal_dispersion
index_optimal = np.abs(dispersion_values - optimal_dispersion).argmin()

# Ensure that the optimal value is within the array bounds
index_optimal = max(min(index_optimal, len(dispersion_values) - 2), 1)

left_range = dispersion_values[:index_optimal][::-1]  # Reverse the left part for descending order
right_range = dispersion_values[index_optimal + 1:]

profile_likelihood_results = [(optimal_dispersion, optimal_objective_value)]

# First, move to the left from the optimal value
current_guess = optimal_params

for disp_value in left_range:
    # Optimize with fixed dispersion
    result = optimize_with_fixed_dispersion(current_guess, disp_value, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms)
    profile_likelihood_results.append((disp_value, result.fun))
    current_guess = result.x  # Update the guess for the next iteration

# Reset the guess before moving to the right
current_guess = optimal_params
# Now, move to the right from the optimal value

for disp_value in right_range:
    # Optimize with fixed dispersion
    result = optimize_with_fixed_dispersion(current_guess, disp_value, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms)
    profile_likelihood_results.append((disp_value, result.fun))
    current_guess = result.x  # Update the guess for the next iteration


# Convert results to DataFrame and plot
df_profile_likelihood = pd.DataFrame(profile_likelihood_results, columns=['Dispersion', 'Objective'])

# Plot the profile likelihood
plt.figure(figsize=(10, 6))
plt.plot(df_profile_likelihood['Dispersion'], df_profile_likelihood['Objective'], marker='o', linestyle='-')
plt.xlabel('Dispersion Coefficient')
plt.ylabel('Objective Function Value')
plt.title('Profile Likelihood of Dispersion Coefficient')
plt.show()
