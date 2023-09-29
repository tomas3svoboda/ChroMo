# Nelder-Mead optimization algorithm. Level 2 sub-optimization of main bi-level optimization.
# Minimazes loss value from Lev2_Loss_Function.py and uses Lin_Solver.py or Nonlin_Solver.py
# based on IsoSelect from Iso_Decision.py
from functions.Lev2_Loss_Function import Lev2_Loss_Function
import functions.global_ as gl
from functions.handle_Optim_Settings import handle_Optim_Settings
import numpy as np


def Lev2_Optim(lvl1Params, experimentCluster, key, lossFunction, factor, solver, spacialDiff = 30, timeDiff = 3000, time = 10800, optimId=1, lvl2optim=None, optimType=None):
    """Optimizazion function for level 2.
    Part of the parameter estimation workflow.
    """
    #print("Calling Lev2_Optim!")
    if not optimId in gl.lossFunctionProgress:
        gl.lossFunctionProgress[optimId] = {}
    if not key in gl.lossFunctionProgress[optimId]:
        gl.lossFunctionProgress[optimId][key] = {}
    if solver == "Lin":
        if optimType == "bilevel":
            bnds = [(gl.lvl2RangeDict[optimId][key][0][0], gl.lvl2RangeDict[optimId][key][0][1]), (gl.lvl2RangeDict[optimId][key][1][0], gl.lvl2RangeDict[optimId][key][1][1])]
        elif optimType == "singlelevel":
            bnds = [(gl.lvl2RangeDict[optimId][key][0][0], gl.lvl2RangeDict[optimId][key][0][1]), (gl.lvl2RangeDict[optimId][key][1][0], gl.lvl2RangeDict[optimId][key][1][1]), (gl.lvl2RangeDict[optimId][key][2][0], gl.lvl2RangeDict[optimId][key][2][1])]
        elif optimType == "calcDisper":
            bnds = [(gl.lvl2RangeDict[optimId][key][0][0], gl.lvl2RangeDict[optimId][key][0][1])]
        elif optimType == "calcDisper2":
            bnds = [(gl.lvl2RangeDict[optimId][key][0][0], gl.lvl2RangeDict[optimId][key][0][1]), (gl.lvl2RangeDict[optimId][key][1][0], gl.lvl2RangeDict[optimId][key][1][1])]
    elif solver == "Nonlin":
        if optimType == "bilevel":
            bnds = [(gl.lvl2RangeDict[optimId][key][0][0], gl.lvl2RangeDict[optimId][key][0][1]), (gl.lvl2RangeDict[optimId][key][1][0], gl.lvl2RangeDict[optimId][key][1][1]), (gl.lvl2RangeDict[optimId][key][2][0], gl.lvl2RangeDict[optimId][key][2][1])]
        elif optimType == "singlelevel":
            bnds = [(gl.lvl2RangeDict[optimId][key][0][0], gl.lvl2RangeDict[optimId][key][0][1]), (gl.lvl2RangeDict[optimId][key][1][0], gl.lvl2RangeDict[optimId][key][1][1]), (gl.lvl2RangeDict[optimId][key][2][0], gl.lvl2RangeDict[optimId][key][2][1]), (gl.lvl2RangeDict[optimId][key][3][0], gl.lvl2RangeDict[optimId][key][3][1])]
        elif optimType == "calcDisper":
            bnds = [(gl.lvl2RangeDict[optimId][key][0][0], gl.lvl2RangeDict[optimId][key][0][1]), (gl.lvl2RangeDict[optimId][key][1][0], gl.lvl2RangeDict[optimId][key][1][1])]
        elif optimType == "calcDisper2":
            bnds = [(gl.lvl2RangeDict[optimId][key][0][0], gl.lvl2RangeDict[optimId][key][0][1]), (gl.lvl2RangeDict[optimId][key][1][0], gl.lvl2RangeDict[optimId][key][1][1]), (gl.lvl2RangeDict[optimId][key][2][0], gl.lvl2RangeDict[optimId][key][2][1])]
    else:
        raise "Unknown solver choice in Lev2_Optim"
    if optimType == "calcDisper" or optimType == "calcDisper2":
        lvl1Params = np.append(lvl1Params, [gl.bVars[optimId][key]])
    res = handle_Optim_Settings(Lev2_Loss_Function,
                                gl.lvl2ParamDict[optimId][key],
                                (experimentCluster, lvl1Params, lossFunction, factor, solver, spacialDiff, timeDiff, time, optimId, optimType),
                                bnds,
                                lvl2optim)
    if lvl2optim["algorithm"] == "1":
        gl.lvl2ParamDict[optimId][key] = np.array(res[0], ndmin=1)
        gl.lv2LossFunctionVals[optimId][key] = res[1]
    else:
        gl.lvl2ParamDict[optimId][key] = res.x
        gl.lv2LossFunctionVals[optimId][key] = res.fun
    return gl.lv2LossFunctionVals[optimId][key]
