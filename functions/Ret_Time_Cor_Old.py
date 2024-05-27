from functions.Handle_File_Creation import Handle_File_Creation
import pandas as pd
from scipy.optimize import minimize
import os
import math


def Ret_Time_Cor(experimentSet, experimentClustersExp, threshold=0, writeToFile=False):
    """Function implementing retention time correction on experiment set data."""
    # If writeToFile flag is set, create a file to write output to
    if writeToFile:
        filePath = experimentSet.metadata.path + "\\Time_Shifts.txt"
        print(filePath)
        file = Handle_File_Creation(filePath)

    # Calculate maximum negative shifts for each experiment to not lose non-zero values
    # This is stored as a temporary property on the experiment object
    for key1, cluster in experimentClustersExp.clusters.items():
        for exp in cluster[0]:
            maxShift = math.inf
            for comp in exp.experimentComponents:
                column = pd.to_numeric(comp.concentrationTime[comp.name])
                gtRes = column.gt(threshold)
                firstNonZeroIndex = gtRes.idxmax()

                if firstNonZeroIndex == 0 and gtRes[0] == False:
                    if comp.concentrationTime.iloc[-1, 0] < maxShift:
                        maxShift = comp.concentrationTime.iloc[-1, 0]
                    continue
                firstNonZeroTime = comp.concentrationTime.iloc[firstNonZeroIndex, 0]
                if firstNonZeroTime < maxShift:
                    maxShift = firstNonZeroTime
            exp.maxShift = -maxShift

    # Define loss function for minimization task
    def Shift_Loss_Function(shifts, avgPeakTimes, cluster):
        sum = 0
        for idx, exp in enumerate(cluster):
            for comp in exp.experimentComponents:
                column = pd.to_numeric(comp.concentrationTime[comp.name])
                peakIndex = column.idxmax()
                peakTime = comp.concentrationTime.iloc[peakIndex, 0]
                sum += abs(peakTime + shifts[idx] - avgPeakTimes[comp.name])
        return sum

    # Calculate average peak time and shift for each cluster
    for key, value in experimentClustersExp.clusters.items():
        avgPeakTimes = dict()
        for key2, value2 in value[1].items():
            peakTimeSum = 0
            for comp in value2:
                pd.set_option('display.max_colwidth', None)
                column = pd.to_numeric(comp.concentrationTime[comp.name])
                peakIndex = column.idxmax()
                peakTime = comp.concentrationTime.iloc[peakIndex, 0]
                peakTimeSum += peakTime
            peakTimeAvg = peakTimeSum / len(value2)
            avgPeakTimes[key2] = peakTimeAvg
        initalGuess = list()
        bounds = list()
        for exp in value[0]:
            initalGuess.append(0)
            bounds.append((exp.maxShift, None))
        res = minimize(Shift_Loss_Function, initalGuess, args=(avgPeakTimes, value[0]),
                        bounds=bounds,  method='Nelder-Mead')
        for idx, exp in enumerate(value[0]):
            for comp in exp.experimentComponents:
                df = comp.concentrationTime
                df.iloc[:, 0] += res.x[idx]
                df.drop(df[df['Time'] < 0].index, inplace=True)
            exp.shift = res.x[idx]
            if writeToFile:
                head2, tail2 = os.path.split(exp.metadata.path)
                experimentName, extesion = os.path.splitext(tail2)
                file.write("Experiment: " + experimentName + ", Shift: " + str(res.x[idx]) + "\n")

    # If writeToFile flag is set, close the file
    if writeToFile:
        file.close()

    # Return the updated experiment set
    return experimentSet
