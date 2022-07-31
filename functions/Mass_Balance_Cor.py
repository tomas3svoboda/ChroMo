from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet
import numpy as np

def Mass_Balance_Cor(experimentSetCor2, experimentSetGauss):
    experimentSetCor3 = Deep_Copy_ExperimentSet(experimentSetCor2)
    for exp2, expG, exp3 in zip(experimentSetCor2.experiments, experimentSetGauss.experiments, experimentSetCor3.experiments):
        for comp2, compG, comp3 in zip(exp2.experimentComponents, expG.experimentComponents, exp3.experimentComponents):
            df2 = comp2.concentrationTime
            dfG = compG.concentrationTime
            df3 = comp3.concentrationTime
            # rename to match naming convention to compOutputMass
            comp_output_mass = np.trapz(y=dfG.iloc[:, 0].to_numpy(), x=dfG.iloc[:, 1].to_numpy())/expG.experimentCondition.flowRate
            # WORK IN PROGRESS
    return experimentSetCor3