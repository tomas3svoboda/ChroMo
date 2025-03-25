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

path = 'C:/UserData/z004d8nt/Documents/VSCHT/001_Paper_JChA/Paper1_Data/Suc_Glu_GE/'
experimentSet = operator_instance.Load_Experiment_Set(path)  #create object of the set of experiments
print('Starting preprocessing')
experimentSet = operator_instance.Preprocess(experimentSet, True, True, True, 0.01)  # does preprocessing
print('Preprocessing done')
experimentCluster = operator_instance.Cluster_By_Component(experimentSet).clusters  # creates new object where components are clustered together

'''object 'experimentCluster' is a dictionary of all the component names as a keys and experimentComp objects which hold
#chromatogram itself and information about the experiment setup, namely feed volume and concentration, column lenght,
#column diameter, flow rate, and commentary'''

i = 0
component_names_all = []
chromatograms = []
uncertainties = []
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
        uncertainties.append(comp_object.preprocessingScore)
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

# Suc/Glu separation
K_Suc, K_Suc_bounds = 1.0458, (0.00001, 45.06)
K_Glu, K_Glu_bounds = 2.9256, (0.00001, 45.06)

disperCoef1, disperCoef_bounds1 = 1, (1, 266500)
initial_guess_lin = [K_Suc,K_Glu,disperCoef1]
bounds_lin = [K_Suc_bounds,K_Glu_bounds,disperCoef_bounds1]

# Suc/Glu separation
KL_Suc, Q_Suc, KL_Suc_bounds, Q_Suc_bounds = 0.06, 6.68, (0.008, 100000.0), (0.00001, 342.3)
KL_Glu, Q_Glu, KL_Glu_bounds, Q_Glu_bounds = 0.14, 24.15, (0.029, 100000.0), (0.00001, 180.2)
disperCoef2, disperCoef_bounds2 = 5.84, (1.0, 266500)
initial_guess_nonlin = [KL_Suc, Q_Suc, KL_Glu, Q_Glu, disperCoef2]
bounds_nonlin = [KL_Suc_bounds, Q_Suc_bounds, KL_Glu_bounds, Q_Glu_bounds, disperCoef_bounds2]

# Start the timer for overall time measurement
start_time = time.time()
print('Starting optimization')

# BEGINNING OF LINEAR CASE ------------------------------------------
'''wrapper = LossFunctionWrapper(component_names, is_linear=True)
result = minimize(wrapper,
                  initial_guess_lin,
                  args=(component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms, uncertainties),
                  method='Nelder-Mead',
                  options={'disp': True,
                          'return_all': True,
                          #'xatol': 1e-2,
                          #'fatol': 1e-1,
                          'adaptive': False},
                  bounds=bounds_lin
                  )
'''
# END OF LINEAR CASE ------------------------------------------

# BEGINNING OF NONLINEAR CASE (uncomment)------------------------------------------
wrapper = LossFunctionWrapper(component_names, is_linear=False)
result = minimize(wrapper,
                  initial_guess_nonlin,
                  args=(component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms, uncertainties),
                  method='Nelder-Mead', 
                  options={'disp': True,
                          'return_all': True,
                          'xatol': 1e-2,
                          'fatol': 1e-1},
                  bounds=bounds_nonlin
                  )
# END OF NONLINEAR CASE (uncomment)------------------------------------------

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
# Format elapsed time
formatted_elapsed_time = str(timedelta(seconds=elapsed_time))
print(f"Optimization took {formatted_elapsed_time}.")

optimized_params = result.x
# Save the updated DataFrame to CSV
wrapper.details_df.to_csv('optimization_run_nonlin_LAST11_CFLCorr.csv', index=False)

#print('Plotting results')
#plot_component_results_lin(component_names, optimized_params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms)

'''
plot_component_results_nonlin(component_names, optimized_params, component_names_all, flow_rates, diameters, lengths, feed_volumes, concentrations, chromatograms)
'''
