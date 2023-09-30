# Function calculates loss value for the level 1 main bi-level optimization.
# The value is calculated as sum of loss values from level 2 sub-optimizations.
from functions.Lev2_Optim import Lev2_Optim
import numpy as np
import functions.global_ as gl
import copy
import time as t
import datetime


def Lev1_Loss_Function(lvl1Params, experimentClustersComp, lossFunction, factor, solver, spacialDiff = 30, timeDiff = 3000, time = 10800, optimId=1, lvl2optim=None, optimType=None, fixporosity=False):
    """Loss function for level 1 optimization.
    Part of the parameter estimation workflow.
    """
    #print("Calling Lev1_Loss_Function!")
    if fixporosity:
        lvl1Params = np.insert(lvl1Params, 0, gl.tmpporosity)
    sum = 0
    timeStart = t.time()
    for key in experimentClustersComp.clusters:
        res = Lev2_Optim(lvl1Params, experimentClustersComp.clusters[key], key, lossFunction, factor, solver, spacialDiff, timeDiff, time, optimId, lvl2optim, optimType, fixporosity)
        sum += res
    print("used lvl1 params", lvl1Params)
    if sum < gl.bestLvl1LossFunctionVal[optimId]:
        gl.bestLvl1LossFunctionVal[optimId] = sum
        print("BETTER SAVING IT!")
        gl.bestLvl1ParamDict[optimId] = copy.deepcopy(lvl1Params)
        gl.bestLvl2ParamDict[optimId] = copy.deepcopy(gl.lvl2ParamDict[optimId])
        gl.bestLvl2LossFunctionVals[optimId] = copy.deepcopy(gl.lv2LossFunctionVals[optimId])
    print("saved dict now", gl.bestLvl1ParamDict[optimId])

    gl.index[optimId] += 1
    print('__________________________________________')
    print("LEVEL 1 Loss function finished with value:")
    print(sum)
    if optimType == "bilevel":
        print('porosity:')
        print(lvl1Params[0])
        for key, val in gl.lvl2ParamDict[optimId].items():
            print('K, D, (Q) for', key, ':')
            for par in val:
                print(par)
    elif optimType == "singlelevel":
        for key, val in gl.lvl2ParamDict[optimId].items():
            print('Porosity, K, D, (Q) for', key, ':')
            for par in val:
                print(par)
    elif optimType == "calcDisper":
        print('porosity, A :')
        for par in lvl1Params:
            print(par)
        for key, val in gl.lvl2ParamDict[optimId].items():
            print('K, (Q) for', key, ':')
            for par in val:
                print(par)
    elif optimType == "calcDisper2":
        print('porosity:')
        print(lvl1Params[0])
        for key, val in gl.lvl2ParamDict[optimId].items():
            print('K, A, (Q) for', key, ':')
            for par in val:
                print(par)
    print('time:')
    print(str(datetime.timedelta(seconds=t.time() - timeStart)))
    print('__________________________________________')
    return sum
