from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet


def Remote_DP_Elim(experimentSetCor1, experimentSetGauss, absTolerance = 0.1):
    experimentSetCor2 = Deep_Copy_ExperimentSet(experimentSetCor1)
    for exp1, expG, exp2 in zip(experimentSetCor1.experiments, experimentSetGauss.experiments, experimentSetCor2.experiments):
        for comp1, compG, comp2 in zip(exp1.experimentComponents, expG.experimentComponents, exp2.experimentComponents):
            removeList = list()
            for i in comp1.concentrationTime.index:
                if(abs(comp1.concentrationTime.iat[i, 1] - compG.concentrationTime.iat[i, 1]) > absTolerance):
                    removeList.append(i)
            comp2.concentrationTime.drop(removeList, axis=0, inplace=True)
        return experimentSetCor2