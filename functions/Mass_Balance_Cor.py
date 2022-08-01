from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet
import numpy as np
import scipy

def Mass_Balance_Cor(experimentSetCor2, experimentSetGauss):
    experimentSetCor3 = Deep_Copy_ExperimentSet(experimentSetCor2)
    for exp2, expG, exp3 in zip(experimentSetCor2.experiments, experimentSetGauss.experiments, experimentSetCor3.experiments):
        initialFeedTime = exp2.experimentCondition.feedTime
        #print(initialFeedTime - (initialFeedTime/2), initialFeedTime + (initialFeedTime/2))
        def Loss_Func(feedTime):
            outputMassSum = 0.0
            feedMassSum = 0.0
            for comp2, compG, comp3 in zip(exp2.experimentComponents, expG.experimentComponents, exp3.experimentComponents):
                dfG = compG.concentrationTime
                # rename to match naming convention to compOutputMass and compFeedMass
                comp_output_mass = np.trapz(y=dfG.iloc[:, 0].to_numpy(), x=dfG.iloc[:, 1].to_numpy())/expG.experimentCondition.flowRate
                comp_feed_mass = feedTime * comp2.feedConcentration * exp2.experimentCondition.flowRate
                #print("feedTime * feedConc * flowRate = " + str(feedTime) + " * " + str(comp2.feedConcentration) + " * " + str(exp2.experimentCondition.flowRate) + " = " + str(comp_feed_mass))
                outputMassSum += comp_output_mass
                feedMassSum += comp_feed_mass
                #print("outputMassSum = " + str(outputMassSum))
                #print("feedMassSum = " + str(feedMassSum))
            return outputMassSum - feedMassSum
        newFeedTime = scipy.optimize.minimize_scalar(Loss_Func, bounds=(initialFeedTime - (initialFeedTime/2), initialFeedTime + (initialFeedTime/2)), method='bounded')
        #print(newFeedTime)
        exp3.experimentCondition.feedTime = newFeedTime.x
    return experimentSetCor3