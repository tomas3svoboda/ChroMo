# Import necessary modules
from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet
from functions.Handle_File_Creation import Handle_File_Creation
from scipy.integrate import simps
import scipy.optimize
import numpy as np
import os

def calculate_component_mass(comp, feed_time, flow_rate):
    """Calculate mass fed and mass eluted for a component."""
    concentrations = comp.concentrationTime.iloc[:, 1].to_numpy()
    times = comp.concentrationTime.iloc[:, 0].to_numpy()
    # Calculate mass eluted using numerical integration in mg
    output_mass = simps(concentrations, times) * flow_rate / 3600 # Convert flow rate from per hour to per second
    # Calculate mass fed in mg
    feed_mass = feed_time * comp.feedConcentration * flow_rate / 3600
    return output_mass, feed_mass

def calculate_relative_uncertainty(tau_optim, tau_new, width, n_comp):
    if width > 0 and n_comp > 0:
        print(abs(tau_optim - tau_new) / (np.sqrt(n_comp)))
        return abs(tau_optim - tau_new) / (width * np.sqrt(n_comp))
    return 0

def Mass_Balance_Cor(experimentSetCor2, writeToFile=False):
    experimentSetCor3 = Deep_Copy_ExperimentSet(experimentSetCor2)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Mass balance correction started...')
    '''for exp in experimentSetCor3.experiments:
        for comp in exp.experimentComponents:
            print(str(exp.metadata.description) + ', ' + str(comp.name))'''

    if writeToFile:
        filePath = experimentSetCor2.metadata.path + "\\Mass_Correction.txt"
        file = Handle_File_Creation(filePath)

    for exp2, exp3 in zip(experimentSetCor2.experiments, experimentSetCor3.experiments):
        initial_feed_time = exp2.experimentCondition.feedTime * 3600
        flow_rate = exp2.experimentCondition.flowRate
        num_components = len(exp2.experimentComponents)

        def Loss_Func(feed_time):
            output_sum = 0
            for comp in exp2.experimentComponents:
                output_mass, feed_mass = calculate_component_mass(comp, feed_time, flow_rate)
                output_sum += abs(output_mass - feed_mass)
            return output_sum

        # result = scipy.optimize.minimize_scalar(Loss_Func, bounds=(initial_feed_time - (initial_feed_time / 2), initial_feed_time + (initial_feed_time / 2)), method='bounded')
        result = scipy.optimize.minimize_scalar(Loss_Func, bounds=(initial_feed_time - (initial_feed_time / 1000), initial_feed_time + (initial_feed_time * 10)), method='bounded')
        new_feed_time = result.x

        for comp2, comp3 in zip(exp2.experimentComponents, exp3.experimentComponents):
            concentration_time_data = {'Time': comp2.concentrationTime.iloc[:, 0], 'Concentration': comp2.concentrationTime.iloc[:, 1]}
            width = comp2.inflectionWidth

            def objective_function(tau, feed_concentration, flow_rate, concentration_data, times):
                """Objective function to minimize: absolute difference between mass fed and mass eluted."""
                mass_fed = tau * feed_concentration * flow_rate / 3600 # convert h to s
                mass_eluted = simps(concentration_data, times) * flow_rate / 3600  # Convert flow rate from mL/h to mL/sec if time is in seconds
                return abs(mass_fed - mass_eluted)

            times = concentration_time_data['Time']
            concentrations = concentration_time_data['Concentration']
            '''res_tau_optim = scipy.optimize.minimize_scalar(
                            objective_function,
                            args=(new_feed_time, flow_rate, concentrations, times),
                            method='Brent'  # Brent's method does not require bounds
                            )
            
            tau_optim = res_tau_optim.x'''
            tau_optim = (simps(concentrations, times)) / (comp2.feedConcentration)
            uncertainty_score = calculate_relative_uncertainty(tau_optim, new_feed_time, width, num_components)

            #print(str(uncertainty_score))
            #print('-----preprocessing score before: ' + str(comp3.preprocessingScore))
            comp3.preprocessingScore += uncertainty_score  # Update preprocessingScore

            # ZERO RESIDUALS____________________________________________________________________________________________
            # Assuming experimentComp is defined
            df = comp3.concentrationTime
            # Model predictions are zero
            zeros_interpolated = np.zeros_like(df.iloc[:, 1].to_numpy())
            # Compute squared error sum using experimental data normalized by max outlet concentration and weighted by uncertainty score
            zero_err_sum = np.sum((df.iloc[:, 1].to_numpy() - zeros_interpolated)**2) / (df[comp3.name].max() ** 2) / comp3.preprocessingScore / np.sqrt(len(df[comp3.name]))
            comp3.zeroSumOfResiduals = zero_err_sum
            #print('Zero peak objective: ' + str(zero_err_sum))
            #___________________________________________________________________________________________________________

            if writeToFile:
                experimentName = os.path.splitext(os.path.basename(exp3.metadata.path))[0]
                file.write(f"Experiment: {experimentName}, Component: {comp3.name}, Original Feed Time: {initial_feed_time:.2f}s, New Feed Time: {new_feed_time:.2f}s, Uncertainty Score: {uncertainty_score:.4f}\n")

        exp3.experimentCondition.originalFeedTime = initial_feed_time / 3600
        exp3.experimentCondition.feedTime = new_feed_time / 3600

    if writeToFile:
        file.close()

    return experimentSetCor3
