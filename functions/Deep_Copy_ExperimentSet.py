from objects.ExperimentSet import ExperimentSet
from objects.ExperimentComponent import ExperimentComponent
from objects.Experiment import Experiment

# Function that creates a deep copy of experiment set
def Deep_Copy_ExperimentSet(experimentSet):
    newExperimentSet = ExperimentSet()
    newExperimentSet.metadata.path = experimentSet.metadata.path
    newExperimentSet.metadata.date = experimentSet.metadata.date
    newExperimentSet.metadata.description = experimentSet.metadata.description
    for experiment in experimentSet.experiments:
        newExperiment = Experiment()
        newExperiment.shift = experiment.shift
        newExperiment.metadata.date = experiment.metadata.date
        newExperiment.metadata.description = experiment.metadata.description
        newExperiment.metadata.path = experiment.metadata.path
        newExperiment.experimentCondition.flowRate = experiment.experimentCondition.flowRate
        newExperiment.experimentCondition.feedVolume = experiment.experimentCondition.feedVolume
        newExperiment.experimentCondition.deadVolume = experiment.experimentCondition.deadVolume
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
