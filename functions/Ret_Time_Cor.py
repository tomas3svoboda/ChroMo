from functions.Handle_File_Creation import Handle_File_Creation
import pandas as pd
from scipy.optimize import minimize
import os

def Ret_Time_Cor(experimentSet, experimentClustersExp, writeToFile = False):
    if writeToFile:
        filePath = experimentSet.metadata.path + "\\Time_Shifts.txt"
        print(filePath)
        file = Handle_File_Creation(filePath)

    # calculate maximum negative shifts for each experiment to not lose non-zero values, again as temporary property
    for key1, cluster in experimentClustersExp.clusters.items():
        for exp in cluster[0]:
            maxShift = -1
            for comp in exp.experimentComponents:
                column = pd.to_numeric(comp.concentrationTime[comp.name])
                firstNonZeroIndex = column.ne(0).idxmax()
                firstNonZeroTime = comp.concentrationTime.iloc[firstNonZeroIndex, 0]
                if maxShift < 0 or firstNonZeroTime < maxShift:
                    maxShift = firstNonZeroTime
            exp.maxShift = -maxShift

    # defines loss function for minimization task
    def Shift_Loss_Function(shifts, avgPeakTimes, cluster):
        sum = 0
        for idx, exp in enumerate(cluster):
            for comp in exp.experimentComponents:
                column = pd.to_numeric(comp.concentrationTime[comp.name])
                peakIndex = column.idxmax()
                peakTime = comp.concentrationTime.iloc[peakIndex, 0]
                sum += abs(peakTime+shifts[idx]-avgPeakTimes[comp.name])
        return sum

    # calculates avg peak time and shift for each cluster
    for key, value in experimentClustersExp.clusters.items():
        avgPeakTimes = dict()
        for key2, value2 in value[1].items():
            peakTimeAvg = 0
            for comp in value2:
                column = pd.to_numeric(comp.concentrationTime[comp.name])
                peakIndex = column.idxmax()
                peakTime = comp.concentrationTime.iloc[peakIndex, 0]
                peakTimeAvg += peakTime
            peakTimeAvg = peakTimeAvg/len(value2)
            avgPeakTimes[key2] = peakTimeAvg
        initalGuess = list()
        bounds = list()
        for exp in value[0]:
            initalGuess.append(0)
            bounds.append((exp.maxShift, None))
        res = minimize(Shift_Loss_Function, initalGuess, args=(avgPeakTimes, value[0]),
                       bounds=bounds, method='Nelder-Mead')
        for idx, exp in enumerate(value[0]):
            for comp in exp.experimentComponents:
                df = comp.concentrationTime
                df.iloc[:, 0] += res.x[idx]
                df.drop(df[df['Time'] < 0].index, inplace=True)
            if writeToFile:
                head2, tail2 = os.path.split(exp.metadata.path)
                experimentName, extesion = os.path.splitext(tail2)
                file.write("Experiment: " + experimentName + ", Shift: " + str(res.x[idx]) + "\n")

    if writeToFile:
        file.close()
    return experimentSet
