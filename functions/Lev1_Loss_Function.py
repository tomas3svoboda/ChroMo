# Function calculates loss value for the level 1 main bi-level optimization.
# The value is calculated as sum of loss values from level 2 sub-optimizations.
from functions.Lev2_Optim import Lev2_Optim
import functions.global_ as gl

def Lev1_Loss_Function(porosity, experimentClustersComp, lossFunction, factor, solver, spacialDiff = 30, timeDiff = 3000, time = 10800, optimId=1):
    sum = 0
    for key in experimentClustersComp.clusters:
        res = Lev2_Optim(porosity[0], experimentClustersComp.clusters[key], key, lossFunction, factor, solver, spacialDiff, timeDiff, time, optimId)
        sum += res
    print('_________________________________________________________________________________________________')
    print('porosity: ' + str(porosity.round(2)))
    print("LEVEL 1 Loss function value: " + str(round(sum, 2)))
    print('__________________________________________')
    gl.index[optimId] += 1
    return sum
