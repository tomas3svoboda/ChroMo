# Nelder-Mead level 1 main optimization searching minima only for sorbent porosity
# based on loss value from Lev1_Loss_Function.py
from scipy.optimize import minimize
from scipy.optimize import Bounds
from functions.Lev1_Loss_Function import Lev1_Loss_Function
import functions.global_ as gl

def Lev1_Optim(experimentClustersComp, lossFunction, factor):
    print("Calling Lev1_Optim with porosity " + str(round(gl.porosity, 2)) + " and range [" + str(round(gl.porosityRange[0], 2)) + ", " + str(round(gl.porosityRange[1], 2)) + "]!")
    #res = minimize(Lev1_Loss_Function, gl.porosity, args=(experimentClustersComp), bounds=Bounds(lb=0, ub=1), method='Nelder-Mead',options={'fatol': 2,'maxfev': 25})
    res = minimize(Lev1_Loss_Function,
                   gl.porosity,
                   args=(experimentClustersComp, lossFunction, factor),
                   bounds=Bounds(lb=gl.porosityRange[0], ub=gl.porosityRange[1]),
                   method='Nelder-Mead',
                   options={'fatol': 0.5})
    gl.porosity = res.x[0]
    gl.lv1LossFunctionVal = res.fun
    return res.fun
