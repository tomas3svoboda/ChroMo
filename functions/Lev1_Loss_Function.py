# Function calculates loss value for the level 1 main bi-level optimization.
# The value is calculated as sum of loss values from level 2 sub-optimizations.
from functions.Lev2_Optim import Lev2_Optim
import functions.global_ as gl
import copy
import time as t
import datetime

def Lev1_Loss_Function(porosity, experimentClustersComp, lossFunction, factor, solver, spacialDiff = 30, timeDiff = 3000, time = 10800, optimId=1, lvl2optim=None):
    sum = 0
    timeStart = t.time()
    for key in experimentClustersComp.clusters:
        res = Lev2_Optim(porosity[0], experimentClustersComp.clusters[key], key, lossFunction, factor, solver, spacialDiff, timeDiff, time, optimId, lvl2optim)
        sum += res
    if sum < gl.bestLvl1LossFunctionVal[optimId]:
        gl.bestLvl1LossFunctionVal[optimId] = sum
        gl.bestPorosity[optimId] = porosity[0]
        gl.bestCompParamDict[optimId] = copy.deepcopy(gl.compParamDict[optimId])
        gl.bestLvl2LossFunctionVals[optimId] = copy.deepcopy(gl.lv2LossFunctionVals[optimId])
    gl.index[optimId] += 1
    print('__________________________________________')
    print("LEVEL 1 Loss function finished with value:")
    print(sum)
    print('porosity:')
    print(porosity[0])
    for key, val in gl.compParamDict[optimId].items():
        print('K, D, (Q) for', key, ':')
        for par in val:
            print(par)
    print('time:')
    print(str(datetime.timedelta(seconds=t.time() - timeStart)))
    print('__________________________________________')
    return sum
