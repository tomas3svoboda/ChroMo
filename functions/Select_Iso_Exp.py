import pandas as pd
from objects.ExperimentClusters import ExperimentClusters

def Select_Iso_Exp(experimentSetCor3, experimentClustersComp):
    componentDict = {}
    for key, value in experimentClustersComp.clusters.items():
        maxConc = [0, 0]
        for comp in value:
            column = pd.to_numeric(comp.concentrationTime.iloc[:, 1])
            peakVal = column.max()
            if peakVal > maxConc[0]:
                maxConc[0] = peakVal
                maxConc[1] = comp
        componentDict[key] = list()
        componentDict[key].append(maxConc[1])
    expIso = ExperimentClusters()
    expIso.clusters = componentDict
    expIso.metadata.description = "Selected Components with highest concentration"
    return expIso