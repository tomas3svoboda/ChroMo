from functions.singleLossFunctions.Single_Loss_Function_Choice import Single_Loss_Function_Choice
import math


def Multi_Loss_Function_Wrapper(params, B, choice, experimentCluster, solver = 'Lin', factor = 1, spacialDiff = 30, timeDiff = 3000, time = 10800, fixedPorosity = 0, fixedA = []):
    """Function allowing to choose between loss function based on choice parameter
    Choices:
    'Simple' - Single_Loss_Function_Simple
    'Squares' - Single_Loss_Function_Squares
    'LogSimple' - Single_Loss_Function_LogSimple
    'LogSquares' - Single_Loss_Function_LogSquares
    Other choices will raise an exception
    """
    porosity = params[0]
    addidx = 1
    if fixedPorosity:
        porosity = fixedPorosity
        addidx = 0
    if solver == "Lin":
        sum = 0
        for idx, pair in enumerate(experimentCluster.clusters.items()):
            flowRate = pair[1][0].experiment.experimentCondition.flowRate
            diameter = pair[1][0].experiment.experimentCondition.columnDiameter
            length = pair[1][0].experiment.experimentCondition.columnLength
            flowSpeed = (flowRate * 1000 / 3600) / ((math.pi * (diameter ** 2) / 4) * porosity)
            if fixedA:
                disperCoef = (1/fixedA[idx]) * length * flowSpeed + B[idx]
                tmp = [porosity, params[idx+addidx], disperCoef]
            else:
                disperCoef = (1/params[idx*2+addidx+1]) * length * flowSpeed + B[idx]
                tmp = [porosity, params[idx*2+addidx], disperCoef]
            for experimentComp in pair[1]:
                res = Single_Loss_Function_Choice(choice, tmp, experimentComp, solver, factor, spacialDiff, timeDiff, time)
                sum += res
        return sum

    elif solver == "Nonlin":
        sum = 0
        for idx, pair in enumerate(experimentCluster.clusters.items()):
            flowRate = pair[1][0].experiment.experimentCondition.flowRate
            diameter = pair[1][0].experiment.experimentCondition.columnDiameter
            length = pair[1][0].experiment.experimentCondition.columnLength
            flowSpeed = (flowRate * 1000 / 3600) / ((math.pi * (diameter ** 2) / 4) * params[0])
            #disperCoef = ((fixedParams[1] * diameter * flowSpeed) / (1 + flowSpeed)) + fixedParams[2]
            if fixedA:
                disperCoef = (1/fixedA[idx]) * length * flowSpeed + B[idx]
                tmp = [porosity, params[idx*2+addidx], disperCoef, params[idx*2+addidx+1]]
            else:
                disperCoef = (1/params[idx*3+addidx+1]) * length * flowSpeed + B[idx]
                tmp = [porosity, params[idx*3+addidx], disperCoef, params[idx*3+addidx+2]]
            for experimentComp in pair[1]:
                res = Single_Loss_Function_Choice(choice, tmp, experimentComp, solver, factor, spacialDiff, timeDiff, time)
                sum += res
        return sum