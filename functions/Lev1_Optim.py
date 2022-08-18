# Nelder-Mead level 1 main optimization searching minima only for sorbent porosity
# based on loss value from Lev1_Loss_Function.py
from scipy.optimize import minimize
from scipy.optimize import Bounds
from functions.Lev1_Loss_Function import Lev1_Loss_Function
import functions.global_ as gl

def Lev1_Optim(experimentClustersComp):
    print("Calling Lev1_Optim with porosity " + str(gl.porosity) + "!")
    res = minimize(Lev1_Loss_Function, gl.porosity, args=(experimentClustersComp), bounds=Bounds(lb=0, ub=1), method='Nelder-Mead',tol=0.1)
    gl.porosity = res.x[0]
    return res.fun
