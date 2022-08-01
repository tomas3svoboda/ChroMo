from objects.ExperimentSet import ExperimentSet
from objects.ExperimentComponent import ExperimentComponent
from objects.Experiment import Experiment

def Deep_Copy_ExperimentSet(experimentSet):
    newExperimentSet = ExperimentSet()
    for experiment in experimentSet.experiments:
        newExperiment = Experiment()
        newExperiment.metadata.date = experiment.metadata.date
        newExperiment.metadata.description = experiment.metadata.description
        newExperiment.experimentCondition.flowRate = experiment.experimentCondition.flowRate
        newExperiment.experimentCondition.feedVolume = experiment.experimentCondition.feedVolume
        newExperiment.experimentCondition.columnLength = experiment.experimentCondition.columnLength
        newExperiment.experimentCondition.columnDiameter = experiment.experimentCondition.columnDiameter
        for experimentComponent in experiment.experimentComponents:
            newExperimentComponent = ExperimentComponent()
            newExperimentComponent.concentrationTime = experimentComponent.concentrationTime.copy(deep=True)
            newExperimentComponent.name = experimentComponent.name
            newExperimentComponent.feedConcentration = experimentComponent.feedConcentration
            newExperimentComponent.experiment = newExperiment
            newExperiment.experimentComponents.append(newExperimentComponent)
        newExperimentSet.experiments.append(newExperiment)
    return newExperimentSet