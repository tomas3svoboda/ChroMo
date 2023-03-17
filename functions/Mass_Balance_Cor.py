from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet
from functions.Handle_File_Creation import Handle_File_Creation
import numpy as np
import pandas as pd
import scipy
import os

def Mass_Balance_Cor(experimentSetCor2, writeToFile = False):
    experimentSetCor3 = Deep_Copy_ExperimentSet(experimentSetCor2)
    if writeToFile:
        filePath = experimentSetCor2.metadata.path + "\\Mass_Correction.txt"
        file = Handle_File_Creation(filePath)
    for exp2, exp3 in zip(experimentSetCor2.experiments, experimentSetCor3.experiments):
        initialFeedTime = exp2.experimentCondition.feedTime
        #print(initialFeedTime - (initialFeedTime/2), initialFeedTime + (initialFeedTime/2))
        def Loss_Func(feedTime):
            outputSum = 0.0
            for comp2, comp3 in zip(exp2.experimentComponents, exp3.experimentComponents):
                df2 = comp2.concentrationTime
                peakVal = float(df2[comp2.name].loc[df2[comp2.name].idxmax()])
                if float(df2[comp2.name].iat[-1]) > peakVal/10 or float(df2[comp2.name].iat[-2]) > peakVal/10 or float(df2[comp2.name].iat[-3]) > peakVal/10:
                    #print("Skipping " + comp2.name)
                    continue
                trapzRes = np.trapz(x=df2.iloc[:, 0].to_numpy(), y=df2.iloc[:, 1].to_numpy())
                comp_output_mass = trapzRes * exp2.experimentCondition.flowRate / 3600 # [mg]
                comp_feed_mass = feedTime * comp2.feedConcentration * exp2.experimentCondition.flowRate # feedTime in [h], result in [mg]
                outputSum += abs(comp_output_mass - comp_feed_mass)
            return outputSum
        newFeedTime = scipy.optimize.minimize_scalar(Loss_Func, bounds=(initialFeedTime - (initialFeedTime/2), initialFeedTime + (initialFeedTime/2)), method='bounded')
        if writeToFile:
            head, tail = os.path.split(exp3.metadata.path)
            experimentName, extesion = os.path.splitext(tail)
            file.write("Experiment: " + experimentName + ", Original Feed Time: " + str(exp3.experimentCondition.feedTime*3600) + "s, New Feed Time: " + str(newFeedTime.x*3600) + "s\n")
        exp3.experimentCondition.originalFeedTime = exp3.experimentCondition.feedTime
        exp3.experimentCondition.feedTime = newFeedTime.x
    """
    for exp in experimentSetCor3.experiments:
        print("Experiment: " + exp.metadata.path)
        print("Loss Function absolute value: " + str(exp.experimentCondition.feedTime))
        print("Loss Function relative value: " + str(exp.experimentCondition.feedTime / exp.feedMassSum))
        exp.feedMassSum = None
    """
    if writeToFile:
        file.close()
    return experimentSetCor3
