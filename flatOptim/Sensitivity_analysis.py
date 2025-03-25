from objects.Operator import Operator
from functions.singleLossFunctions.Single_Loss_Function_Choice import Single_Loss_Function_Choice
import numpy as np
import pandas as pd
from itertools import product
import math
import time
import datetime
import uuid

def sensitivity_analysis_to_excel(porosity, B, langmuirConst_range, saturCoef_range, steps_langmuirConst, steps_saturCoef, experimentCluster, choice, solver, factor, spacialDiff, timeDiff, total_time, output_file, component_name, A=0.0001):
    """
    Performs sensitivity analysis by varying parameters within given ranges and logging the resulting loss values to an Excel file.

    Args:
    porosity (float): Constant porosity value.
    B (list): Bodenstein number for each experiment.
    langmuirConst_range (tuple): (min, max) range for langmuirConst.
    saturCoef_range (tuple): (min, max) range for saturCoef.
    steps_langmuirConst (int): Number of steps for langmuirConst.
    steps_saturCoef (int): Number of steps for saturCoef.
    experimentCluster (dict): Dictionary of clusters of components.
    choice (str): Choice of loss function.
    solver (str): Solver type.
    factor (int): Normalization factor.
    spacialDiff (numeric): Spatial resolution.
    timeDiff (numeric): Temporal resolution.
    time (numeric): Total simulation time.
    output_file (str): Path to the output Excel file.
    component_name (str): Name of the component to perform the sensitivity analysis on.
    A (float): Fixed value for the diffusion coefficient, used to calculate the dispersion coefficient.

    Returns:
    pd.DataFrame: DataFrame containing parameter sets and corresponding loss values.
    """
    # Generate parameter values
    langmuirConst_values = np.linspace(langmuirConst_range[0], langmuirConst_range[1], steps_langmuirConst)
    saturCoef_values = np.linspace(saturCoef_range[0], saturCoef_range[1], steps_saturCoef)

    results = []
    total_combinations = len(langmuirConst_values) * len(saturCoef_values)

    # Start timing the first iteration
    start_time = time.time()

    # Generate all combinations of langmuirConst and saturCoef values
    for i, (langmuirConst, saturCoef) in enumerate(product(langmuirConst_values, saturCoef_values)):
        total_loss = 0
        individual_losses = {}
        for idx, (cluster_key, components) in enumerate(experimentCluster.items()):
            # Filter to get the specified component
            components = [comp for comp in components if comp.name == component_name]
            if not components:
                continue  # Skip if the component is not found
            for experimentComp in components:
                flowRate = experimentComp.experiment.experimentCondition.flowRate
                diameter = experimentComp.experiment.experimentCondition.columnDiameter
                length = experimentComp.experiment.experimentCondition.columnLength
                flowSpeed = (flowRate * 1000 / 3600) / ((math.pi * (diameter ** 2) / 4) * porosity)
                disperCoef = (1 / B) * length * flowSpeed + A

                # Parameters for the component under analysis
                params = [porosity, langmuirConst, disperCoef, saturCoef]

                loss = Single_Loss_Function_Choice(choice, params, experimentComp, solver, factor, spacialDiff, timeDiff, total_time)
                total_loss += loss

                # Log the individual loss for this experiment component
                experiment_id = experimentComp.experiment.metadata.description
                if experiment_id not in individual_losses:
                    individual_losses[experiment_id] = 0
                individual_losses[experiment_id] += loss

        result = {
            'porosity': porosity,
            'langmuirConst': langmuirConst,
            'saturCoef': saturCoef,
            'total_loss': total_loss,
            **individual_losses
        }
        results.append(result)
        print(f"Parameters: {params} => Total Loss: {total_loss}")

        # Estimate remaining time
        if i == 0:
            iteration_time = time.time() - start_time
            print(f"First iteration took {iteration_time:.2f} seconds.")
        else:
            elapsed_time = time.time() - start_time
            average_time_per_iteration = elapsed_time / (i + 1)
            remaining_iterations = total_combinations - (i + 1)
            remaining_time = remaining_iterations * average_time_per_iteration
            print(f"Iteration {i + 1}/{total_combinations} took {average_time_per_iteration:.2f} seconds on average. Estimated remaining time: {remaining_time:.2f} seconds.")
        print('________________________________________________________________________')
    # Create DataFrame from results
    df_results = pd.DataFrame(results)

    # Remove the file extension from the output file name if it exists
    if output_file.endswith('.xlsx'):
        output_file = output_file[:-5]

    # Generate a unique filename
    unique_filename = f"{output_file}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{component_name}.xlsx"

    # Save DataFrame to Excel file
    df_results.to_excel(unique_filename, index=False)
    print(f"Results saved to {unique_filename}")

    # Pivot the DataFrame to create a matrix format
    df_pivot = df_results.pivot(index='saturCoef', columns='langmuirConst', values='total_loss')

    # Remove the file extension from the output file name if it exists
    if output_file.endswith('.xlsx'):
        output_file = output_file[:-5]

    # Generate a unique filename
    unique_filename_pivot = f"PIVOT{output_file}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.xlsx"

    # Save DataFrame to Excel file
    df_pivot.to_excel(unique_filename_pivot)
    print(f"Pivot results saved to {unique_filename_pivot}")
    print('________________________________________________________________________')
    print('________________________________________________________________________')

    return df_pivot

    return df_results

operator_instance = Operator()

#this parts loads experiment set from the folder, creates all the data handling objects and also does preprocessing

path = 'C:/UserData/z004d8nt/Documents/VSCHT/001_Paper_JChA/Paper1_Data/Suc_Glu_GE'
experimentSet = operator_instance.Load_Experiment_Set(path)  #create object of the set of experiments
experimentCluster = operator_instance.Cluster_By_Component(experimentSet).clusters  # creates new object where components are clustered together
print('Starting preprocessing')
experimentSet_preprocessed = operator_instance.Preprocess(experimentSet, True, True, True, 0.01)  # does preprocessing
print('Preprocessing done')
experimentCluster_preprocessed = operator_instance.Cluster_By_Component(experimentSet_preprocessed).clusters  # creates new object where components are clustered together

'''object 'experimentCluster' is a dictionary of all the component names as a keys and experimentComp objects which hold
#chromatogram itself and information about the experiment setup, namely feed volume and concentration, column lenght,
#column diameter, flow rate, and commentary'''


# Define params and grid:
porosity = 0.3752
#B = [7.4, 7.4]  # Example Bodenstein numbers for two experiments
B = 9.69
langmuirConst_range = (0.0075, 0.010)
saturCoef_range = (342.3, 342.31)
#steps_langmuirConst = 40
#steps_saturCoef = 40
steps_langmuirConst = 2
steps_saturCoef = 2
choice = 'Squares'  # Example loss function choice
solver = 'Nonlin'
factor = 3 #selects relativization method
spacialDiff = 30
timeDiff = 3000
total_time = 10800
output_file = 'Glu_detail3_Bo7_35.xlsx'
component_names = ['Glu']  # Name of the component to analyze

# Perform sensitivity analysis and save results to Excel
for comp_name in component_names:
        sensitivity_results = sensitivity_analysis_to_excel(porosity,
                                                    B,
                                                    langmuirConst_range,
                                                    saturCoef_range,
                                                    steps_langmuirConst,
                                                    steps_saturCoef,
                                                    experimentCluster_preprocessed,
                                                    choice,
                                                    solver,
                                                    factor,
                                                    spacialDiff,
                                                    timeDiff,
                                                    total_time,
                                                    output_file,
                                                    comp_name)


