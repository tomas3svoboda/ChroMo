import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import walk
import datetime
import time
import random #just for testing, delete on release
from objects.ExperimentSet import ExperimentSet
from objects.ExperimentComponent import ExperimentComponent
from objects.Experiment import Experiment
from objects.ExperimentClusters import ExperimentClusters
from objects.Solution import Solution
from functions.Fit_Gauss import Fit_Gauss
from functions.Ret_Time_Cor import Ret_Time_Cor
from functions.Remote_DP_Elim import Remote_DP_Elim
from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet
from functions.Compare_ExperimentSets import Compare_ExperimentSets
from functions.Mass_Balance_Cor import Mass_Balance_Cor
from functions.Select_Iso_Exp import Select_Iso_Exp
from functions.Lin_Solver import Lin_Solver
from functions.Nonlin_Solver import Nonlin_Solver
from functions.Single_Loss_Function_Simple import Single_Loss_Function_Simple
from functions.Bilevel_Optim import Bilevel_Optim
from functions.Lev2_Optim import Lev2_Optim
from functions.Lev2_Loss_Function import Lev2_Loss_Function
from functions.Loss_Function_Analysis import Loss_Function_Analysis

"""
Time measuring decorator
"""
def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.22f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


"""
Main class orchestrating program functions and user interface
"""
class Operator:
    def Start(self):
        par1, par2, par3, par4 = self.Setting_parameters()
        path = input('Enter path to Experiment set: ')
        experimentSet = self.Load_Experiment_Set(path)
        experimentSetCopy = Deep_Copy_ExperimentSet(experimentSet)
        experimentClusterCompCond = self.Cluster_By_Condition2(experimentSetCopy)
        experimentSetCor1 = Ret_Time_Cor(experimentSetCopy, experimentClusterCompCond)
        experimentSetGauss = Fit_Gauss(experimentSetCor1)
        while True:
            dpElimInput = input("Use fitted Gauss curve directly as a corrected datapoints?[Y, N]")
            if dpElimInput == "Y":
                experimentSetCor2 = Remote_DP_Elim(experimentSetCor1, experimentSetGauss)
                break
            if dpElimInput == "N":
                experimentSetCor2 = experimentSetGauss
                break
            else:
                print("Wrong input.")
        experimentSetCor3 = Mass_Balance_Cor(experimentSetCor2, experimentSetGauss)
        experimentClusterComp = self.Cluster_By_Component(experimentSetCor3)
        expIso = Select_Iso_Exp(experimentSetCor3, experimentClusterComp)
        # TODO Continue

    #Start for testing purposes

    @timeit
    def Start_For_Testing(self):
        #Nonlin_Solver()
        #path = "C:\\Users\\z004d8nt\\PycharmProjects\\ChoMo\\docu\\TestExperimentSet1"
        #Single_Loss_Function(experimentSet.experiments[0])
        #experimentSetCopy = Fit_Gauss(experimentSetCopy)
        #Compare_ExperimentSets(experimentSet, experimentSetCopy)

        #comp = experimentSetCopy.experiments[0].experimentComponents[0]
        #cond = experimentSetCopy.experiments[0].experimentCondition
        #print(cond.flowRate, cond.columnLength, cond.columnDiameter, cond.feedVolume, comp.feedConcentration)
        #res = Lin_Solver(cond.flowRate, cond.columnLength, cond.columnDiameter, cond.feedVolume, comp.feedConcentration, 0.52 ,12000,  8000, debugPrint=True, debugGraph=True)
        path = "C:\\Users\\Adam\\ChroMo\\docu\\TestExperimentSet1"
        solution = Solution()
        experimentSet = self.Load_Experiment_Set(path)
        for exp in experimentSet.experiments:
            for comp in exp.experimentComponents:
                solution.Add_Result(comp.name, comp.experiment.metadata.path, random.random(), random.random(), random.random())
        solution.Export_To_CSV("C:\\Users\\Adam\\ChroMo\\testSolution.csv")
        #experimentSetCopy = Deep_Copy_ExperimentSet(experimentSet)
        #experimentClusterComp = self.Cluster_By_Component(experimentSetCopy)
        #Loss_Function_Analysis(experimentClusterComp, component='Glc', xstep=200, ystep=200)
        #experimentClusterCompCond = self.Cluster_By_Condition2(experimentSetCopy)
        #experimentSetCopy = Ret_Time_Cor(experimentSetCopy, experimentClusterCompCond)
        #experimentSetCopy = Mass_Balance_Cor(experimentSetCopy, experimentSetCopy)

        #comp = experimentSetCopy.experiments[0].experimentComponents[2].concentrationTime
        #cond = experimentSetCopy.experiments[0].experimentCondition
        #plt.scatter(comp['time'],comp['Fru'])
        #plt.show()
        #comp.plot(x ='Time', y='Sac')
        #plt.show()
        #t = np.linspace(0, 10800, 5000)
        #plt.plot(t, res[:, -1])
        #plt.show()
        #result = Bilevel_Optim(experimentSetCopy, experimentClusterComp)
        #result = Lev2_Optim([150, 235, 16, 3, 150e-3], experimentClusterComp.clusters['Sac'])
        #print(result)
        #expIso = Select_Iso_Exp(experimentSetCopy, experimentClusterComp)
        #experimentSetCor1 = Mass_Balance_Cor(experimentSet, experimentSet)

    def Setting_Parameters(self):
        par1 = float(input('Enter parameter 1: '))
        par2 = float(input('Enter parameter 2: '))
        par3 = float(input('Enter parameter 3: '))
        par4 = float(input('Enter parameter 4: '))
        return par1, par2, par3, par4

    def Load_Experiment_Set(self, path):
        experimentSet = ExperimentSet()
        experimentSet.metadata.path = path
        experimentSet.metadata.date = datetime.date.today().strftime("%m/%d/%Y")
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
            experiment.metadata.path = path + "\\" + file
            experiment.experimentCondition.flowRate = float(flowRate)
            experiment.experimentCondition.feedVolume = float(feedVolume)
            experiment.experimentCondition.columnLength = float(columnLength)
            experiment.experimentCondition.columnDiameter = float(columnDiameter)
            for index in range(columnNames[1:].size):
                experimentComponent = ExperimentComponent()
                experimentComponent.concentrationTime = df.iloc[:, [0, 1 + index]].astype(float)
                experimentComponent.concentrationTime['Time'] = experimentComponent.concentrationTime['Time'].apply(lambda x: x*60)
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
        clusters.metadata = experimentSet.metadata
        clusters.clusters = componentDict
        clusters.metadata.description += "\nClusters by component"
        return clusters

    def Cluster_By_Condition(self, experimentSet):
        clusterByCondition = ExperimentClusters()
        clusterByCondition.metadata = experimentSet.metadata
        clusterByCondition.metadata.description += "\nClusters by condition"
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
        clusterByCondition.metadata = experimentSet.metadata
        clusterByCondition.metadata.description += "\nClusters by condition"
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
