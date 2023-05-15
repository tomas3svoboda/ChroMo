from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet


def Dead_Volume_Preprocess(experimentSet):
    currExperimentSet = Deep_Copy_ExperimentSet(experimentSet)
    for exp in currExperimentSet.experiments:
        # calculates time of dead volume as t
        t = (exp.experimentCondition.deadVolume/exp.experimentCondition.flowRate)*3600
        for comp in exp.experimentComponents:
            # gets name of first column (should be time)
            timeColumnName = comp.concentrationTime.columns[0]
            # removes all rows with time lower than t
            comp.concentrationTime = comp.concentrationTime[comp.concentrationTime[timeColumnName] >= t]
            # substracts t from all remaining times
            comp.concentrationTime[timeColumnName] = comp.concentrationTime[timeColumnName].apply(lambda x: x-t)
    return currExperimentSet
