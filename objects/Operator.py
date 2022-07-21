import pandas as pd
from os import walk
from objects.ExperimentSet import ExperimentSet
from objects.ExperimentComponent import ExperimentComponent
from objects.Experiment import Experiment
from objects.ExperimentClusters import ExperimentClusters
import datetime

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
        experimentSet = self.Load_Experiment_Set(path)
        """
        n = 3
        print(len(self.expSet.experiments))
        print(self.expSet.experiments[0].metadata.date)
        print(self.expSet.experiments[0].metadata.description)
        print(self.expSet.experiments[0].experiment_condition.column_length)
        print(self.expSet.experiments[0].experiment_condition.column_diameter)
        print(self.expSet.experiments[0].experiment_condition.feed_volume)
        print(self.expSet.experiments[0].experiment_components[n].name)
        print(self.expSet.experiments[0].experiment_components[n].feed_concentration)
        print(self.expSet.experiments[0].experiment_components[n].concentration_time)
        print(self.expSet.experiments[0].experiment_components[n].experiment)
        print(self.expSet.experiments[0])
        """
        """
        component_clusters = self.Cluster_by_component(experimentSet)
        print(component_clusters[0].clusters)
        print(component_clusters[0].metadata.description)
        """
        condition_cluster = self.Cluster_By_Conditions(experimentSet)
        for key, values in condition_cluster.clusters.items():
            print(key, ":")
            for value in values:
                print("   ", value.name,value.feedConcentration, value.experiment.experimentCondition.feedVolume,
                      value.experiment.experimentCondition.columnDiameter, value.experiment.experimentCondition.columnLength,
                      value.experiment.experimentCondition.flowRate)

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
            columnLength = df.iat[0, 1]
            columnDiameter = df.iat[1, 1]
            flowRate = df.iat[2, 1]
            feedVolume = df.iat[3, 1]
            columnNames = df.iloc[[7]].to_numpy()[0]
            feedConcentrations = df.iloc[[6]].to_numpy()[0][1:]
            df.drop([0, 1, 2, 3, 4, 5, 6, 7], axis=0, inplace=True)
            df.columns = columnNames
            experiment = Experiment()
            experiment.metadata.date = date
            experiment.metadata.description = description
            experiment.experimentCondition.feedVolume = float(feedVolume)
            experiment.experimentCondition.columnLength = float(columnLength)
            experiment.experimentCondition.columnDiameter = float(columnDiameter)
            experiment.experimentCondition.flowRate = float(flowRate)
            for index in range(columnNames[1:].size):
                experimentComponent = ExperimentComponent()
                experimentComponent.concentrationTime = df.iloc[:, [0, 1 + index]]
                experimentComponent.name = columnNames[1+index]
                experimentComponent.feedConcentration = float(feedConcentrations[index].replace(',', '.'))
                experimentComponent.experiment = experiment
                experiment.experimentComponents.append(experimentComponent)
                print(len(experiment.experimentComponents))
            experimentSet.experiments.append(experiment)
        return experimentSet

    def Cluster_By_Component(self, experiment_set):
        component_dict = {}
        for experiment in experiment_set.experiments:
            for component in experiment.experimentComponents:
                if component.name in component_dict:
                    component_dict[component.name].append(component)
                else:
                    component_dict[component.name] = list()
                    component_dict[component.name].append(component)
        clusters = ExperimentClusters()
        clusters.clusters = component_dict
        clusters.metadata.description = "Clusters by component"
        return clusters

    def Cluster_By_Conditions(self, experiment_set):
        clusterByCondition = ExperimentClusters()
        for experiment in experiment_set.experiments:
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