import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import walk
from objects.ExperimentSet import ExperimentSet
from objects.ExperimentComponent import ExperimentComponent
from objects.Experiment import Experiment
from objects.ExperimentClusters import ExperimentClusters
import datetime
from functions.Fit_Gauss import Fit_Gauss
from functions.Ret_Time_Cor import Ret_Time_Cor
from functions.Remote_DP_Elim import Remote_DP_Elim
from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet
from functions.Compare_ExperimentSets import Compare_ExperimentSets
from functions.Mass_Balance_Cor import Mass_Balance_Cor
from functions.Select_Iso_Exp import Select_Iso_Exp
"""
Main class orchestrating program functions and user interface
"""
class Operator:
    def Start(self):
        """
        par1, par2, par3, par4 = self.Setting_parameters()
        print(par1, par2, par3, par4)
        path = input('Enter path to Experiment set: ')
        """
        path = "C:\\Users\\Adam\\ChroMo\\docu\\TestExperimentSet1"
        #path = "C:\\Users\\z004d8nt\\PycharmProjects\\ChoMo\\docu\\TestExperimentSet1"
        experimentSet = self.Load_Experiment_Set(path)
        experimentSetCopy = Deep_Copy_ExperimentSet(experimentSet)
        experimentClusterComp = self.Cluster_By_Component(experimentSetCopy)
        expIso = Select_Iso_Exp(experimentSetCopy, experimentClusterComp)
        print(expIso.clusters)
        #experimentSetCor1 = Mass_Balance_Cor(experimentSet, experimentSet)
        #experimentClusterCompCond = self.Cluster_By_Condition2(experimentSetCopy)
        #Ret_Time_Cor(experimentSetCopy, experimentClusterCompCond)
        #experimentSetCopy = Fit_Gauss(experimentSetCopy)
        #Compare_ExperimentSets(experimentSet, experimentSetCopy)



    def Setting_Parameters(self):
        par1 = float(input('Enter parameter 1: '))
        par2 = float(input('Enter parameter 2: '))
        par3 = float(input('Enter parameter 3: '))
        par4 = float(input('Enter parameter 4: '))
        return par1, par2, par3, par4

    def Load_Experiment_Set(self, path):
        experimentSet = ExperimentSet()
        filenames = next(walk(path), (None, None, []))[2]
        for file in filenames:
            df = pd.read_excel(path + "\\" + file)
            description = df.iat[0, 3]
            date = df.iat[2, 3]
            columnLength = float(df.iat[0, 1])
            columnDiameter = float(df.iat[1, 1])
            flowRate = float(df.iat[2, 1])
            feedVolume = float(df.iat[3, 1])
            columnNames = df.iloc[[7]].to_numpy()[0]
            feedConcentrations = df.iloc[[6]].replace(',','.', regex=True).to_numpy()[0][1:]
            df.drop([0, 1, 2, 3, 4, 5, 6, 7], axis=0, inplace=True)
            df.columns = columnNames
            df = df.replace(',','.', regex=True).astype(float)
            experiment = Experiment()
            experiment.metadata.date = date
            experiment.metadata.description = description
            experiment.experimentCondition.flowRate = float(flowRate)
            experiment.experimentCondition.feedVolume = float(feedVolume)
            experiment.experimentCondition.columnLength = float(columnLength)
            experiment.experimentCondition.columnDiameter = float(columnDiameter)
            for index in range(columnNames[1:].size):
                experimentComponent = ExperimentComponent()
                experimentComponent.concentrationTime = df.iloc[:, [0, 1 + index]].astype(float)
                experimentComponent.concentrationTime.reset_index(inplace=True, drop=True)
                experimentComponent.name = columnNames[1+index]
                experimentComponent.feedConcentration = float(feedConcentrations[index])
                experimentComponent.experiment = experiment
                experiment.experimentComponents.append(experimentComponent)
            experimentSet.experiments.append(experiment)
        return experimentSet

    def Cluster_By_Component(self, experimentSet):
        componentDict = {}
        for experiment in experimentSet.experiments:
            for component in experiment.experimentComponents:
                if component.name in componentDict:
                    componentDict[component.name].append(component)
                else:
                    componentDict[component.name] = list()
                    componentDict[component.name].append(component)
        clusters = ExperimentClusters()
        clusters.clusters = componentDict
        clusters.metadata.description = "Clusters by component"
        return clusters

    def Cluster_By_Condition(self, experimentSet):
        clusterByCondition = ExperimentClusters()
        for experiment in experimentSet.experiments:
            for component in experiment.experimentComponents:
                foundFlag = False
                for key, value in clusterByCondition.clusters.items():
                    if self.Cluster_Match(value[0], component):
                        value.append(component)
                        foundFlag = True
                if not foundFlag:
                    clusterByCondition.clusters[self.Create_Key(component)] = list()
                    clusterByCondition.clusters[self.Create_Key(component)].append(component)
        return clusterByCondition

    def Cluster_By_Condition2(self, experimentSet):
        clusterByCondition = ExperimentClusters()
        tmpCompList = list()
        for experiment in experimentSet.experiments:
            tmpCompList = tmpCompList + experiment.experimentComponents
        while len(tmpCompList) > 0:
            maxCount = 0
            maxCluster = list()
            for experimentComp in tmpCompList:
                tmpCluster = list()
                tmpCluster.append(experimentComp)
                for experimentComp2 in tmpCompList:
                    if experimentComp != experimentComp2 and self.Cluster_Match(experimentComp, experimentComp2):
                        tmpCluster.append(experimentComp2)
                if len(tmpCluster) > maxCount:
                    maxCount = len(tmpCluster)
                    maxCluster = tmpCluster
            clusterByCondition.clusters[self.Create_Key(maxCluster[0])] = maxCluster
            tmpCompList = [x for x in tmpCompList if x not in maxCluster]
        return clusterByCondition


    """
    Calculates if comp2 is close to comp1 with tolerance(default 0.05)
    """
    def Cluster_Match(self, comp1, comp2, tolerance = 0.05):
        cond1 = comp1.experiment.experimentCondition
        cond2 = comp2.experiment.experimentCondition
        if(comp1.name == comp2.name and
                abs(comp1.feedConcentration - comp2.feedConcentration) < tolerance * comp1.feedConcentration and
                abs(cond1.flowRate - cond2.flowRate) < tolerance * cond1.flowRate and
                abs(cond1.columnDiameter - cond2.columnDiameter) < tolerance * cond1.columnDiameter and
                abs(cond1.columnLength - cond2.columnLength) < tolerance * cond1.columnLength and
                abs(cond1.feedVolume - cond2.feedVolume) < tolerance * cond1.feedVolume):
            return True
        return False

    """
    Creates key for cluster dictionary from component
    """
    def Create_Key(self, comp):
        name = comp.name
        feedConc = str(comp.feedConcentration)
        colDia = str(comp.experiment.experimentCondition.columnDiameter)
        colLen = str(comp.experiment.experimentCondition.columnLength)
        feedVol = str(comp.experiment.experimentCondition.feedVolume)
        flowRate = str(comp.experiment.experimentCondition.flowRate)
        return ":".join([name, feedConc, colDia, colLen, feedVol, flowRate])

    def Print_Component_Graph(self, experimentComponent):
        experimentComponent.concentrationTime.plot.line(x='Time')
        plt.show()
