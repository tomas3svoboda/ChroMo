# Function calculates loss value for the level 1 main bi-level optimization.
# The value is calculated as sum of loss values from level 2 sub-optimizations.
from functions.Lev2_Optim import Lev2_Optim
import functions.global_ as gl
import copy

def Lev1_Loss_Function(porosity, experimentClustersComp, lossFunction, factor, solver, spacialDiff = 30, timeDiff = 3000, time = 10800, optimId=1, lvl2optim=None):
    sum = 0
    for key in experimentClustersComp.clusters:
        res = Lev2_Optim(porosity[0], experimentClustersComp.clusters[key], key, lossFunction, factor, solver, spacialDiff, timeDiff, time, optimId, lvl2optim)
        sum += res
    print('_________________________________________________________________________________________________')
    print('porosity: ' + str(porosity.round(2)))
    print("LEVEL 1 Loss function value: " + str(round(sum, 2)))
    print('__________________________________________')
    if sum < gl.bestLvl1LossFunctionVal[optimId]:
        gl.bestLvl1LossFunctionVal[optimId] = sum
        gl.bestPorosity[optimId] = porosity[0]
        gl.bestCompParamDict[optimId] = copy.deepcopy(gl.compParamDict[optimId])
        gl.bestLvl2LossFunctionVals[optimId] = copy.deepcopy(gl.lv2LossFunctionVals[optimId])
    gl.index[optimId] += 1
    return sum
