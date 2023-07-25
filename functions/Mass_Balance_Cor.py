# Import necessary modules
from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet
from functions.Handle_File_Creation import Handle_File_Creation
import numpy as np
import pandas as pd
import scipy
import os


def Mass_Balance_Cor(experimentSetCor2, writeToFile=False):
    """Function implementing mass balance correction on experiment set data."""
    # Create a deep copy of the input experiment set to avoid modifying the original data
    experimentSetCor3 = Deep_Copy_ExperimentSet(experimentSetCor2)

    # If writeToFile is True, create a file to write the output to
    if writeToFile:
        filePath = experimentSetCor2.metadata.path + "\\Mass_Correction.txt"
        file = Handle_File_Creation(filePath)

    # Loop over all experiments in the input experiment set and the deep copied version
    for exp2, exp3 in zip(experimentSetCor2.experiments, experimentSetCor3.experiments):

        # Record the initial feed time for the experiment
        initialFeedTime = exp2.experimentCondition.feedTime

        # Define a function to minimize the mass balance error by adjusting the feed time
        def Loss_Func(feedTime):

            # Initialize a variable to track the total difference between the mass of input and output components
            outputSum = 0.0

            # Loop over all components in the experiment
            for comp2, comp3 in zip(exp2.experimentComponents, exp3.experimentComponents):

                # Retrieve the concentration-time data for the current component in both experiments
                df2 = comp2.concentrationTime

                # Find the maximum concentration of the component in the experiment
                peakVal = float(df2[comp2.name].loc[df2[comp2.name].idxmax()])

                # Skip the component if the final concentration is greater than 1/10 of the maximum concentration, indicating incomplete reaction or adsorption
                if float(df2[comp2.name].iat[-1]) > peakVal / 10 or float(
                        df2[comp2.name].iat[-2]) > peakVal / 10 or float(df2[comp2.name].iat[-3]) > peakVal / 10:
                    continue

                # Calculate the mass of the component in the output stream using trapezoidal integration of the concentration-time curve
                trapzRes = np.trapz(x=df2.iloc[:, 0].to_numpy(), y=df2.iloc[:, 1].to_numpy())
                comp_output_mass = trapzRes * exp2.experimentCondition.flowRate / 3600  # [mg]

                # Calculate the mass of the component in the input stream using the current feed time and feed concentration
                comp_feed_mass = feedTime * comp2.feedConcentration * exp2.experimentCondition.flowRate  # feedTime in [h], result in [mg]

                # Add the difference between the input and output masses to the total difference
                outputSum += abs(comp_output_mass - comp_feed_mass)

            # Return the total difference between the input and output masses
            return outputSum

        # Minimize the mass balance error using the Loss_Func function and the initial feed time as the starting point
        newFeedTime = scipy.optimize.minimize_scalar(Loss_Func, bounds=(
        initialFeedTime - (initialFeedTime / 2), initialFeedTime + (initialFeedTime / 2)), method='bounded')

        # If writeToFile is True, write the results of the mass balance correction to the output file
        if writeToFile:
            head, tail = os.path.split(exp3.metadata.path)
            experimentName, extesion = os.path.splitext(tail)
            file.write("Experiment: " + experimentName + ", Original Feed Time: " + str(exp3.experimentCondition.feedTime*3600) + "s, New Feed Time: " + str(newFeedTime.x*3600) + "s\n")

        # Save original feed time
        exp3.experimentCondition.originalFeedTime = exp3.experimentCondition.feedTime

        # Save calculated new feed time
        exp3.experimentCondition.feedTime = newFeedTime.x

    # If writeToFile is True, close output file
    if writeToFile:
        file.close()

    # Return experiment set with adjusted feed times
    return experimentSetCor3
