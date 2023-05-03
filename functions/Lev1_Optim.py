# Level 1 main optimization searching minima only for sorbent porosity
# based on loss value from Lev1_Loss_Function.py
from functions.Lev1_Loss_Function import Lev1_Loss_Function
from functions.handle_Optim_Settings import handle_Optim_Settings
import functions.global_ as gl

# This function performs the first level of optimization
# Inputs:
#     experimentClustersComp: a list of ExperimentClusterComp objects
#     lossFunction: the loss function to use
#     factor: a factor to multiply the loss function by
#     solver: the solver to use
#     spacialDiff: the spacial difference
#     timeDiff: the time difference
#     time: the time to use
#     optimId: the optimization ID to use
#     lvl1optim: the level 1 optimization settings
#     lvl2optim: the level 2 optimization settings
# Returns:
#     The value of the level 1 loss function
def Lev1_Optim(experimentClustersComp, lossFunction, factor, solver, spacialDiff = 30, timeDiff = 3000, time = 10800, optimId=1, lvl1optim = None, lvl2optim = None):
    gl.index[optimId] = 0
    res = handle_Optim_Settings(Lev1_Loss_Function, gl.porosity[optimId],
                              (experimentClustersComp, lossFunction, factor, solver, spacialDiff, timeDiff, time, optimId, lvl2optim),
                               [(gl.porosityRange[optimId][0], gl.porosityRange[optimId][1])],
                               lvl1optim, 1)
    if lvl1optim["algorithm"] == "1":
        gl.porosity[optimId] = res[0]
        gl.lv1LossFunctionVal[optimId] = res[1]
    else:
        gl.porosity[optimId] = res.x[0]
        gl.lv1LossFunctionVal[optimId] = res.fun
    return gl.lv1LossFunctionVal[optimId]


