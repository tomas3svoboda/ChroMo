from objects.Operator import Operator
from functions.singleLossFunctions.Single_Loss_Function_Choice import Single_Loss_Function_Choice
import numpy as np
import pandas as pd
import math
import time
import datetime
import uuid
import os
from scipy.optimize import differential_evolution

def calculate_flow_speed(flow_rate, diameter, porosity):
    return (flow_rate * 1000 / 3600) / ((math.pi * (diameter ** 2) / 4) * porosity)

def calculate_dispersion_coefficient(flow_speed, length, B_value, A):
    return (1 / B_value) * length * flow_speed + A

def loss_function_wrapper(porosity, langmuirConst, disperCoef, experimentCluster, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name, saturCoef):
    total_loss = 0
    for idx, (cluster_key, components) in enumerate(experimentCluster.items()):
        components = [comp for comp in components if comp.name == component_name]
        if not components:
            continue
        for experimentComp in components:
            params = [porosity, langmuirConst, disperCoef, saturCoef]
            loss = Single_Loss_Function_Choice(choice, params, experimentComp, solver, factor, spacialDiff, timeDiff, total_time)
            total_loss += loss
    return total_loss

def find_optimal_saturCoef(porosity, langmuirConst, B, experimentCluster, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name, saturCoef_range, A):
    best_saturCoef = None
    best_loss = float('inf')

    for idx, (cluster_key, components) in enumerate(experimentCluster.items()):
        components = [comp for comp in components if comp.name == component_name]
        if not components:
            continue
        for experimentComp in components:
            flowRate = experimentComp.experiment.experimentCondition.flowRate
            diameter = experimentComp.experiment.experimentCondition.columnDiameter
            length = experimentComp.experiment.experimentCondition.columnLength
            flowSpeed = calculate_flow_speed(flowRate, diameter, porosity)
            disperCoef = calculate_dispersion_coefficient(flowSpeed, length, B[idx], A)

            result = differential_evolution(lambda saturCoef: loss_function_wrapper(porosity, langmuirConst, disperCoef, experimentCluster, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name, saturCoef[0]),
                                            bounds=[saturCoef_range])
            if result.fun < best_loss:
                best_loss = result.fun
                best_saturCoef = result.x[0]

    return best_saturCoef, best_loss

def sensitivity_analysis(porosity, B, langmuirConst_values, saturCoef_range, experimentCluster, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name, A=0.0001):
    results = []

    for langmuirConst in langmuirConst_values:
        best_saturCoef, best_loss = find_optimal_saturCoef(porosity, langmuirConst, B, experimentCluster, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name, saturCoef_range, A)
        results.append({
            'porosity': porosity,
            'langmuirConst': langmuirConst,
            'best_saturCoef': best_saturCoef,
            'best_loss': best_loss
        })
        print(f"LangmuirConst: {langmuirConst}, Best SaturCoef: {best_saturCoef}, Best Loss: {best_loss}")

    return results

def log_results(results, log_file):
    if os.path.exists(log_file):
        mode = 'a'
    else:
        mode = 'w'

    with open(log_file, mode) as f:
        f.write(f"Log Date: {datetime.datetime.now()}\n")
        for result in results:
            f.write(f"LangmuirConst: {result['langmuirConst']}, Best SaturCoef: {result['best_saturCoef']}, Best Loss: {result['best_loss']}\n")
        f.write("________________________________________________________\n")

# Initialize operator and preprocess data
operator_instance = Operator()
path = 'C:/UserData/z004d8nt/Documents/VSCHT/001_Paper_JChA/Paper1_Data/Suc_Glu_GE'
experimentSet = operator_instance.Load_Experiment_Set(path)
experimentCluster = operator_instance.Cluster_By_Component(experimentSet).clusters
print('Starting preprocessing')
experimentSet_preprocessed = operator_instance.Preprocess(experimentSet, False, False, False, 0.01)
print('Preprocessing done')
experimentCluster_preprocessed = operator_instance.Cluster_By_Component(experimentSet_preprocessed).clusters

# Define parameters and perform sensitivity analysis
porosity = 0.3752
B = [19.7, 19.7]
langmuirConst_values = [0.20, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]
saturCoef_range = (10, 44)
choice = 'Squares'
solver = 'Nonlin'
factor = 3
spacialDiff = 30
timeDiff = 3000
total_time = 10800
component_name = 'Glu'
log_file = 'sensitivity_likelihood_noPP_Glu.txt'

results = sensitivity_analysis(porosity, B, langmuirConst_values, saturCoef_range, experimentCluster_preprocessed, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name)
log_results(results, log_file)

'''from objects.Operator import Operator
from functions.singleLossFunctions.Single_Loss_Function_Choice import Single_Loss_Function_Choice
import numpy as np
import pandas as pd
import math
import time
import datetime
import uuid
import os
from scipy.optimize import minimize

def calculate_flow_speed(flow_rate, diameter, porosity):
    return (flow_rate * 1000 / 3600) / ((math.pi * (diameter ** 2) / 4) * porosity)

def calculate_dispersion_coefficient(flow_speed, length, B_value, A):
    return (1 / B_value) * length * flow_speed + A

def loss_function_wrapper(porosity, langmuirConst, disperCoef, experimentCluster, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name, saturCoef):
    total_loss = 0
    for idx, (cluster_key, components) in enumerate(experimentCluster.items()):
        components = [comp for comp in components if comp.name == component_name]
        if not components:
            continue
        for experimentComp in components:
            params = [porosity, langmuirConst, disperCoef, saturCoef]
            loss = Single_Loss_Function_Choice(choice, params, experimentComp, solver, factor, spacialDiff, timeDiff, total_time)
            total_loss += loss
    return total_loss

def find_optimal_saturCoef(porosity, langmuirConst, B, experimentCluster, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name, saturCoef_range, A, initial_guess):
    best_saturCoef = None
    best_loss = float('inf')

    for idx, (cluster_key, components) in enumerate(experimentCluster.items()):
        components = [comp for comp in components if comp.name == component_name]
        if not components:
            continue
        for experimentComp in components:
            flowRate = experimentComp.experiment.experimentCondition.flowRate
            diameter = experimentComp.experiment.experimentCondition.columnDiameter
            length = experimentComp.experiment.experimentCondition.columnLength
            flowSpeed = calculate_flow_speed(flowRate, diameter, porosity)
            disperCoef = calculate_dispersion_coefficient(flowSpeed, length, B[idx], A)

            result = minimize(lambda saturCoef: loss_function_wrapper(porosity, langmuirConst, disperCoef, experimentCluster, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name, saturCoef[0]),
                              x0=[initial_guess],
                              bounds=[saturCoef_range],
                              method='Powell')
            if result.fun < best_loss:
                best_loss = result.fun
                best_saturCoef = result.x[0]

    return best_saturCoef, best_loss

def sensitivity_analysis(porosity, B, langmuirConst_values, saturCoef_range, experimentCluster, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name, initial_guess, A=0.0001):
    results = []

    for langmuirConst in langmuirConst_values:
        best_saturCoef, best_loss = find_optimal_saturCoef(porosity, langmuirConst, B, experimentCluster, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name, saturCoef_range, A, initial_guess)
        results.append({
            'porosity': porosity,
            'langmuirConst': langmuirConst,
            'best_saturCoef': best_saturCoef,
            'best_loss': best_loss
        })
        print(f"LangmuirConst: {langmuirConst}, Best SaturCoef: {best_saturCoef}, Best Loss: {best_loss}")

    return results

def log_results(results, log_file):
    if os.path.exists(log_file):
        mode = 'a'
    else:
        mode = 'w'

    with open(log_file, mode) as f:
        f.write(f"Log Date: {datetime.datetime.now()}\n")
        for result in results:
            f.write(f"LangmuirConst: {result['langmuirConst']}, Best SaturCoef: {result['best_saturCoef']}, Best Loss: {result['best_loss']}\n")
        f.write("________________________________________________________\n")

# Initialize operator and preprocess data
operator_instance = Operator()
path = 'C:/UserData/z004d8nt/Documents/VSCHT/001_Paper_JChA/Paper1_Data/Single'
experimentSet = operator_instance.Load_Experiment_Set(path)
experimentCluster = operator_instance.Cluster_By_Component(experimentSet).clusters
print('Starting preprocessing')
experimentSet_preprocessed = operator_instance.Preprocess(experimentSet, True, True, True, 0.01)
print('Preprocessing done')
experimentCluster_preprocessed = operator_instance.Cluster_By_Component(experimentSet_preprocessed).clusters

# Define parameters and perform sensitivity analysis
porosity = 0.3752
B = [27.37, 27.37]
langmuirConst_values = [0.05, 0.075, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
saturCoef_range = (1, 10000)
initial_guess = 7
choice = 'Squares'
solver = 'Nonlin'
factor = 3
spacialDiff = 30
timeDiff = 3000
total_time = 10800
component_name = 'Suc'
log_file = 'sensitivity_analysis_log.txt'

results = sensitivity_analysis(porosity, B, langmuirConst_values, saturCoef_range, experimentCluster_preprocessed, choice, solver, factor, spacialDiff, timeDiff, total_time, component_name, initial_guess)
log_results(results, log_file)'''
