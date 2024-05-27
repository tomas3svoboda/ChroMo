from scipy.optimize import minimize
from flatOptim.Loss_function_flat import LossFunctionWrapper
from flatOptim.plot_component_results import plot_component_results_lin
from flatOptim.plot_component_results import plot_component_results_nonlin
from objects.Operator import Operator
from datetime import timedelta
import time
import pandas as pd

operator_instance = Operator()

#this parts loads experiment set from the folder, creates all the data handling objects and also does preprocessing

path = 'C:/UserData/z004d8nt/Documents/VSCHT/001_Paper_JChA/Paper1_Data/Suc_Glu_GE_Copy/'
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

print('There are ' + str(num_of_components) + ' separated components:')

lin_compParams_dict = {} #  dictionary for the results of component specific params
nonlin_compParams_dict = {} #  dictionary for the results of component specific params

for component in component_names:
    print(component)
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

'''I will use least squares loss function which are relativized by diving the error value for experiment by 
max outlet concetration squared giving no-unit final value scaled by feed conc'''
# define initial guess and boundaries
# LINEAR
'''K_A, K_A_bounds = 5.0, (0.01, 100.0)
K_B, K_B_bounds = 5.0, (0.01, 100.0)
K_C, K_C_bounds = 5.0, (0.01, 100.0)
K_D, K_D_bounds = 5.0, (0.01, 100.0)
K_E, K_E_bounds = 5.0, (0.01, 100.0)
K_F, K_F_bounds = 5.0, (0.01, 100.0)
K_G, K_G_bounds = 5.0, (0.01, 100.0)
K_H, K_H_bounds = 5.0, (0.01, 100.0)
K_I, K_I_bounds = 5.0, (0.01, 100.0)
K_J, K_J_bounds = 5.0, (0.01, 100.0)
disperCoef, disperCoef_bounds = 10.0, (0.01, 100.0)
initial_guess_lin = [K_A,K_B,K_C,K_D,K_E,K_F,K_G,K_H,K_I,K_J,disperCoef]
bounds_lin = [K_A_bounds,K_B_bounds,K_C_bounds,K_D_bounds,K_E_bounds,K_F_bounds,K_G_bounds,K_H_bounds,K_I_bounds,K_J_bounds,disperCoef_bounds]'''

# Suc/Glu separation
K_Suc, K_Suc_bounds = 50.0, (0.01, 100.0)
K_Glu, K_Glu_bounds = 50.0, (0.01, 100.0)
disperCoef, disperCoef_bounds = 80.0, (0.01, 100.0)
initial_guess_lin = [K_Suc,K_Glu,disperCoef]
bounds_lin = [K_Suc_bounds,K_Glu_bounds,disperCoef_bounds]

# NONLINEAR
L_A = 2.0
S_A = 10.0
L_B = 2.0
S_B = 10.0
L_C = 2.0
S_C = 10.0
L_D = 2.0
S_D = 10.0
L_E = 2.0
S_E = 10.0
L_F = 2.0
S_F = 10.0
L_G = 2.0
S_G = 10.0
L_H = 2.0
S_H = 10.0
L_I = 2.0
S_I = 10.0
L_J = 2.0
S_J = 10.0
disperCoeff = 10.0
initial_guess_nonlin = [L_A,S_A,L_B,S_B,L_C,S_C,L_D,S_D,L_E,S_E,L_F,S_F,L_G,S_G,L_H,S_H,L_I,S_I,L_J,S_J,disperCoef]

# Start the timer for overall time measurement
start_time = time.time()
print('Starting optimization')

# BEGINNING OF LINEAR CASE ------------------------------------------
wrapper = LossFunctionWrapper(component_names, is_linear=True)
result = minimize(wrapper,
                  initial_guess_lin,
                  args=(component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms),
                  method='Nelder-Mead',
                  options={'disp': True,
                          'return_all': True,
                          'xatol': 1e-2,
                          'fatol': 1e-4},
                  bounds=bounds_lin
                  )

# END OF LINEAR CASE ------------------------------------------

# BEGINNING OF NONLINEAR CASE (uncomment)------------------------------------------
'''
wrapper = LossFunctionWrapper(component_names, is_linear=False)
result = minimize(wrapper,
                  initial_guess_nonlin,
                  args=(component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms),
                  method='Nelder-Mead', 
                  options={'disp': True,
                          'return_all': True,
                          'xatol': 1e-2,
                          'fatol': 1e-4},
                  bounds=bounds_nonlin
                  )
'''
# END OF NONLINEAR CASE (uncomment)------------------------------------------

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
# Format elapsed time
formatted_elapsed_time = str(timedelta(seconds=elapsed_time))
print(f"Optimization took {formatted_elapsed_time}.")

optimized_params = result.x
# Save the updated DataFrame to CSV
wrapper.details_df.to_csv('optimization_run_details.csv', index=False)

#print('Plotting results')
#plot_component_results_lin(component_names, optimized_params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms)

'''
plot_component_results_nonlin(component_names, optimized_params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms)
'''
