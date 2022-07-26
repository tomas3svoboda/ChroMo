import pandas as pd

def Ret_Time_Cor(experimentSet, experimentClustersCompCond):
    for key, value in experimentClustersCompCond.clusters.items():
        peakTimeSum = 0
        for comp in value:
            column = pd.to_numeric(comp.concentrationTime[comp.name])
            peakIndex = column.idxmax()
            peakTime = comp.concentrationTime.iloc[peakIndex, 0]
            peakTimeSum += peakTime
        peakTimeAvg = peakTimeSum/len(value)
        for comp in value:
            column = pd.to_numeric(comp.concentrationTime[comp.name])
            peakIndex = column.idxmax()
            peakTime = comp.concentrationTime.iloc[peakIndex, 0]
            timeDifference = peakTimeAvg - peakTime
            comp.concentrationTime.iloc[:, 0] += timeDifference
    return experimentSet
