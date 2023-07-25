# Nelder-Mead optimization algorithm. Level 2 sub-optimization of main bi-level optimization.
# Minimazes loss value from Lev2_Loss_Function.py and uses Lin_Solver.py or Nonlin_Solver.py
# based on IsoSelect from Iso_Decision.py
from functions.Lev2_Loss_Function import Lev2_Loss_Function
import functions.global_ as gl
from functions.handle_Optim_Settings import handle_Optim_Settings


def Lev2_Optim(porosity, experimentCluster, key, lossFunction, factor, solver, spacialDiff = 30, timeDiff = 3000, time = 10800, optimId=1, lvl2optim=None):
    """Optimizazion function for level 2.
    Part of the parameter estimation workflow.
    """
    if not optimId in gl.lossFunctionProgress:
        gl.lossFunctionProgress[optimId] = {}
    if not key in gl.lossFunctionProgress[optimId]:
        gl.lossFunctionProgress[optimId][key] = {}
    if solver == "Lin":
        bnds = [(gl.compRangeDict[optimId][key][0][0], gl.compRangeDict[optimId][key][0][1]), (gl.compRangeDict[optimId][key][1][0], gl.compRangeDict[optimId][key][1][1])]
    elif solver == "Nonlin":
        bnds = [(gl.compRangeDict[optimId][key][0][0], gl.compRangeDict[optimId][key][0][1]), (gl.compRangeDict[optimId][key][1][0], gl.compRangeDict[optimId][key][1][1]), (gl.compRangeDict[optimId][key][2][0], gl.compRangeDict[optimId][key][2][1])]
    else:
        raise "Unknown solver choice in Lev2_Optim"
    res = handle_Optim_Settings(Lev2_Loss_Function,
                                gl.compParamDict[optimId][key],
                                (experimentCluster, porosity, lossFunction, factor, solver, spacialDiff, timeDiff, time, optimId),
                                bnds,
                                lvl2optim)
    if lvl2optim["algorithm"] == "1":
        gl.compParamDict[optimId][key] = res[0]
        gl.lv2LossFunctionVals[optimId][key] = res[1]
    else:
        gl.compParamDict[optimId][key] = res.x
        gl.lv2LossFunctionVals[optimId][key] = res.fun
    return gl.lv2LossFunctionVals[optimId][key]
