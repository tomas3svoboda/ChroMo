# Level 1 main optimization searching minima only for sorbent porosity
# based on loss value from Lev1_Loss_Function.py
from functions.Lev1_Loss_Function import Lev1_Loss_Function
from functions.handle_Optim_Settings import handle_Optim_Settings
import functions.global_ as gl

def Lev1_Optim(experimentClustersComp, lossFunction, factor, solver, spacialDiff = 30, timeDiff = 3000, time = 10800, optimId=1, lvl1optim = None, lvl2optim = None):
    print("Calling Lev1_Optim with porosity " + str(round(gl.porosity[optimId], 2)) + " and range [" + str(round(gl.porosityRange[optimId][0], 2)) + ", " + str(round(gl.porosityRange[optimId][1], 2)) + "]!")
    #res = minimize(Lev1_Loss_Function, gl.porosity, args=(experimentClustersComp), bounds=Bounds(lb=0, ub=1), method='Nelder-Mead',options={'fatol': 2,'maxfev': 25})
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


