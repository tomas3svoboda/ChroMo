import pandas as pd

def Ret_Time_Cor(experimentSet, experimentClustersCompCond):
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
    return experimentSet
