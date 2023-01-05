from functions.Handle_File_Creation import Handle_File_Creation
import pandas as pd
import os

def Ret_Time_Cor(experimentSet, experimentClustersCompCond, writeToFile = False):
    if writeToFile:
        head, tail = os.path.split(list(experimentClustersCompCond.clusters.values())[0][0].experiment.metadata.path)
        filePath = head + "\\Time_Shifts.txt"
        file = Handle_File_Creation(filePath)
    for key, value in experimentClustersCompCond.clusters.items():
        peakTimeAvg = 0
        for comp in value:
            column = pd.to_numeric(comp.concentrationTime[comp.name])
            peakIndex = column.idxmax()
            peakTime = comp.concentrationTime.iloc[peakIndex, 0]
            peakTimeAvg += peakTime
        peakTimeAvg = peakTimeAvg/len(value)
        for comp in value:
            column = pd.to_numeric(comp.concentrationTime[comp.name])
            peakIndex = column.idxmax()
            peakTime = comp.concentrationTime.iloc[peakIndex, 0]
            timeDifference = peakTimeAvg - peakTime
            df = comp.concentrationTime
            df.iloc[:, 0] += timeDifference
            df.drop(df[df['time'] < 0].index, inplace=True)
            if writeToFile:
                head2, tail2 = os.path.split(comp.experiment.metadata.path)
                experimentName, extesion = os.path.splitext(tail2)
                file.write("Experiment: " + experimentName + ", Component: " + comp.name + ", Shift: " + str(timeDifference) + "\n")
    if writeToFile:
        file.close()
    return experimentSet
