# Function calculates loss value for level 2 sub-optimization of the main bi-level optimization
import os
import math
import functions.global_ as gl
from functions.singleLossFunctions.Single_Loss_Function_Choice import Single_Loss_Function_Choice

"""
Loss Function options:
    Default or 'Simple' - Single_Loss_Function_Simple
    'Squares' - Single_Loss_Function_Squares
    'LogSimple' - Single_Loss_Function_LogSimple
    'LogSquares' - Single_Loss_Function_LogSquares
"""
def Lev2_Loss_Function(params, experimentCluster, fixedParams, lossFunction ='Simple', factor = 1, solver ="Lin", spacialDiff = 30, timeDiff = 3000, time = 10800, optimId=1, optimType=None):
    """Loss function for level 2 optimization.
    Part of the parameter estimation workflow.
    """
    #print("Calling Lev2_Loss_Function!")
    if optimType == "bilevel":
        if solver == "Lin":
            params2 = [fixedParams[0], params[0], params[1]]
        elif solver == "Nonlin":
            params2 = [fixedParams[0], params[0], params[1], params[2]]
        else:
            raise Exception("unknown solver " + solver)
    elif optimType == "singlelevel":
        if solver == "Lin":
            params2 = [params[0], params[1], params[2]]
        elif solver == "Nonlin":
            params2 = [params[0], params[1], params[2], params[3]]
        else:
            raise Exception("unknown solver " + solver)
    elif optimType == "calcDisper":
        # CALCULATE DISPERSION
        flowRate = experimentCluster[0].experiment.experimentCondition.flowRate
        diameter = experimentCluster[0].experiment.experimentCondition.columnDiameter
        flowSpeed = (flowRate * 1000 / 3600) / ((math.pi * (diameter ** 2) / 4) * fixedParams[0])
        disperCoef = ((fixedParams[1] * diameter * flowSpeed) / (1 + flowSpeed)) + fixedParams[2]
        if solver == "Lin":
            params2 = [fixedParams[0], params[0], disperCoef]
        elif solver == "Nonlin":
            params2 = [fixedParams[0], params[0], disperCoef, params[1]]
        else:
            raise Exception("unknown solver " + solver)
    elif optimType == "calcDisper2":
        # CALCULATE DISPERSION
        flowRate = experimentCluster[0].experiment.experimentCondition.flowRate
        diameter = experimentCluster[0].experiment.experimentCondition.columnDiameter
        flowSpeed = (flowRate * 1000 / 3600) / ((math.pi * (diameter ** 2) / 4) * fixedParams[0])
        disperCoef = ((params[1] * diameter * flowSpeed) / (1 + flowSpeed)) + fixedParams[1]
        if solver == "Lin":
            params2 = [fixedParams[0], params[0], disperCoef]
        elif solver == "Nonlin":
            params2 = [fixedParams[0], params[0], disperCoef, params[2]]
        else:
            raise Exception("unknown solver " + solver)
    else:
        raise Exception("unknown optimization type " + optimType)
    sum = 0
    for comp in experimentCluster:
        head, tail = os.path.split(comp.experiment.metadata.path)
        if not optimId in gl.lossFunctionProgress:
            gl.lossFunctionProgress[optimId] = {}
        if not comp.name in gl.lossFunctionProgress[optimId]:
            gl.lossFunctionProgress[optimId][comp.name] = {}
        if not tail in gl.lossFunctionProgress[optimId][comp.name]:
            gl.lossFunctionProgress[optimId][comp.name][tail] = []
        res = Single_Loss_Function_Choice(lossFunction, params2, comp, solver, factor, spacialDiff, timeDiff, time)
        if not optimId in gl.index:
            gl.index[optimId] = 0
        if len(gl.lossFunctionProgress[optimId][comp.name][tail]) == gl.index[optimId]:
            gl.lossFunctionProgress[optimId][comp.name][tail].append(res)
        elif len(gl.lossFunctionProgress[optimId][comp.name][tail]) == gl.index[optimId]+1:
            gl.lossFunctionProgress[optimId][comp.name][tail][gl.index[optimId]] = res
        sum += res
    #print("lEVEL 2 Loss function value: " + str(sum))
    return sum
