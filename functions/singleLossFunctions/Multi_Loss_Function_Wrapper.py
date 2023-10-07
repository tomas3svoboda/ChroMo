from functions.singleLossFunctions.Single_Loss_Function_Choice import Single_Loss_Function_Choice
import math


def Multi_Loss_Function_Wrapper(params, choice, experimentCluster, solver = 'Lin', factor = 1, spacialDiff = 30, timeDiff = 3000, time = 10800):
    """Function allowing to choose between loss function based on choice parameter
    Choices:
    'Simple' - Single_Loss_Function_Simple
    'Squares' - Single_Loss_Function_Squares
    'LogSimple' - Single_Loss_Function_LogSimple
    'LogSquares' - Single_Loss_Function_LogSquares
    Other choices will raise an exception
    """
    if solver == "Lin":
        sum = 0
        for idx, pair in enumerate(experimentCluster.clusters.items()):
            tmp = [params[0], params[idx*2+1], params[idx*2+2]]
            for experimentComp in pair[1]:
                res = Single_Loss_Function_Choice(choice, tmp, experimentComp, solver, factor, spacialDiff, timeDiff, time)
                sum += res
        return sum

    elif solver == "Nonlin":
        sum = 0
        for idx, pair in enumerate(experimentCluster.clusters.items()):
            tmp = [params[0], params[idx*3+1], params[idx*3+2], params[idx*3+3]]
            for experimentComp in pair[1]:
                res = Single_Loss_Function_Choice(choice, tmp, experimentComp, solver, factor, spacialDiff, timeDiff, time)
                sum += res
        return sum