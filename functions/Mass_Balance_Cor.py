from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet
import numpy as np
import scipy

def Mass_Balance_Cor(experimentSetCor2):
    experimentSetCor3 = Deep_Copy_ExperimentSet(experimentSetCor2)
    for exp2, exp3 in zip(experimentSetCor2.experiments, experimentSetCor3.experiments):
        initialFeedTime = exp2.experimentCondition.feedTime
        #print(initialFeedTime - (initialFeedTime/2), initialFeedTime + (initialFeedTime/2))
        def Loss_Func(feedTime):
            outputMassSum = 0.0
            feedMassSum = 0.0
            for comp2, comp3 in zip(exp2.experimentComponents, exp3.experimentComponents):
                df2 = comp2.concentrationTime
                # rename to match naming convention to compOutputMass and compFeedMass
                #comp_feed
                comp_output_mass = np.trapz(x=df2.iloc[:, 0].to_numpy(), y=df2.iloc[:, 1].to_numpy())/exp2.experimentCondition.flowRate/3600
                comp_feed_mass = feedTime * comp2.feedConcentration * exp2.experimentCondition.flowRate
                outputMassSum += comp_output_mass
                feedMassSum += comp_feed_mass
                result = abs(outputMassSum - feedMassSum)
                exp3.feedMassSum = feedMassSum
            return result
        newFeedTime = scipy.optimize.minimize_scalar(Loss_Func, bounds=(initialFeedTime - (initialFeedTime/2), initialFeedTime + (initialFeedTime/2)), method='bounded')
        #print(newFeedTime)
        exp3.experimentCondition.feedTime = newFeedTime.x
    for exp in experimentSetCor3.experiments:
        print("Experiment: " + exp.metadata.path)
        print("Loss Function absolute value: " + str(exp.experimentCondition.feedTime))
        print("Loss Function relative value: " + str(exp.experimentCondition.feedTime / exp.feedMassSum))
        exp.feedMassSum = None
    return experimentSetCor3