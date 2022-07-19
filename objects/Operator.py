import pandas as pd
from os import walk
from Experiment_set import Experiment_set
from Experiment_component import Experiment_component
from Experiment import Experiment


class Operator:
    expSet = Experiment_set()
    def Start(self):
        """
        par1, par2, par3, par4 = self.Setting_parameters()
        print(par1, par2, par3, par4)
        path = input('Enter path to Experiment set: ')
        """
        path = "C:\\Users\\Adam\\ChoMo\\docu\\TestExperimentSet1"
        self.Load_experiment_set(path)
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
        component_clusters = self.Cluster_by_component()
        print(component_clusters)

    def Setting_parameters(self):
        par1 = input('Enter parameter 1: ')
        par2 = input('Enter parameter 2: ')
        par3 = input('Enter parameter 3: ')
        par4 = input('Enter parameter 4: ')
        return par1, par2, par3, par4

    def Load_experiment_set(self, path):
        filenames = next(walk(path), (None, None, []))[2]
        for file in filenames:
            df = pd.read_excel(path + "\\" + file)
            description = df.iat[0, 3]
            date = df.iat[2, 3]
            column_length = df.iat[0, 1]
            column_diameter = df.iat[1, 1]
            flow_rate = df.iat[2, 1]
            feed_volume = df.iat[3, 1]
            columnNames = df.iloc[[7]].to_numpy()[0]
            feed_concentrations = df.iloc[[6]].to_numpy()[0][1:]
            df.drop([0, 1, 2, 3, 4, 5, 6, 7], axis=0, inplace=True)
            df.columns = columnNames
            experiment = Experiment()
            experiment.metadata.date = date
            experiment.metadata.description = description
            experiment.experiment_condition.feed_volume = feed_volume
            experiment.experiment_condition.column_length = column_length
            experiment.experiment_condition.column_diameter = column_diameter
            experiment.experiment_condition.flow_rate = flow_rate
            for index in range(columnNames[1:].size):
                experiment_component = Experiment_component()
                experiment_component.concentration_time = df.iloc[:, [0, 1+index]]
                experiment_component.name = columnNames[1+index]
                experiment_component.feed_concentration = feed_concentrations[index]
                experiment_component.experiment = experiment
                experiment.experiment_components.append(experiment_component)
            self.expSet.experiments.append(experiment)

    def Cluster_by_component(self):
        component_dict = {}
        for experiment in self.expSet.experiments:
            for component in experiment.experiment_components:
                if component.name in component_dict:
                    component_dict[component.name].append(component)
                else:
                    component_dict[component.name] = list()
                    component_dict[component.name].append(component)
        return component_dict

