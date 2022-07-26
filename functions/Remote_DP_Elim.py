
def Remote_DP_Elim(experimentSetCor1, experimentSetGauss, absTolerance = 0.1):
    removeList = list()
    for exp1, expG in zip(experimentSetCor1.experiments, experimentSetGauss.experiments):
        for comp1, compG in zip(exp1.experimentComponents, expG.experimentComponents):
            for i in range(len(comp1.index)):
                if(abs(comp1.iat[i, 1] - compG.iat[i, 1]) > absTolerance):
                    removeList.append(i)
    experimentSetCor2 = experimentSetCor1.copy()
    experimentSetCor2.drop(removeList, axis=0, inplace=True)
    return experimentSetCor2