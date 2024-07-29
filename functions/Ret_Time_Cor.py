from functions.Handle_File_Creation import Handle_File_Creation
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import os
import math

def calculate_theoretical_shifts(component, avg_peak_times, num_experiments):
    original_peak_time = component.concentrationTime.iloc[component.concentrationTime[component.name].idxmax(), 0]
    avg_peak_time = avg_peak_times[component.name]
    # Calculate the peak width using the inflection points method
    peak_width = component.inflectionWidth
    if num_experiments > 1 and peak_width > 0:
        abs_difference = abs(original_peak_time - avg_peak_time)
        relative_shift = abs_difference / (peak_width * np.sqrt(num_experiments))
    else:
        relative_shift = 0
        print('No retention time shift score can be assigned!')
    return relative_shift


def Ret_Time_Cor(experimentSet, experimentClustersExp, threshold=0, writeToFile=False):
    """Function implementing retention time correction on experiment set data."""
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Retention time correction started...')
    # If writeToFile flag is set, create a file to write output to
    if writeToFile:
        filePath = experimentSet.metadata.path + "\\Time_Shifts.txt"
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
            for comp in value2:
                theoretical_shift = calculate_theoretical_shifts(comp, avgPeakTimes, len(value2))
                #print(str(theoretical_shift))
                #print('comp:' + str(comp.name) + ';  RELATIVE THEORETICAL SHIFT: ' + str(theoretical_shift) + '; numb of exp in CLUSTER: ' + str(len(value2)))
                #print('-----preprocessing score before: ' + str(comp.preprocessingScore))
                comp.preprocessingScore += theoretical_shift
                #print('-----preprocessing score after: ' + str(comp.preprocessingScore))
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

                #df.to_excel(comp.name + exp.metadata.description + '.xlsx', index=False)
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
