# Nelder-Mead optimization algorithm. Level 2 sub-optimization of main bi-level optimization.
# Minimazes loss value from Lev2_Loss_Function.py and uses Lin_Solver.py or Nonlin_Solver.py
# based on IsoSelect from Iso_Decision.py
from scipy.optimize import minimize
from functions.Lev2_Loss_Function import Lev2_Loss_Function
import functions.global_ as gl

def Lev2_Optim(porosity, experimentCluster, key):
    #print("Calling Lev2_Optim with params " + str(gl.compParamDict[key]) + "!")
    #res = minimize(Lev2_Loss_Function, gl.compParamDict[key], args=(experimentCluster, porosity), bounds=((0, None), (0, None)), method='Nelder-Mead', options={'fatol': 0.5,'maxfev': 25})
    res = minimize(Lev2_Loss_Function, gl.compParamDict[key], args=(experimentCluster, porosity), bounds=((0, None), (0, None)), method='Nelder-Mead', options={'fatol': 0.1})
    print('__________________________________________')
    for i in [0,1]:
        #if res.x[i] == 0 or res.x[i] == 1000:
        if res.x[i] >= 15000:
            print('Bound hit! ' + str(res.x))
            res.x[i] = 1000
        elif res.x[i] == 0:
            print('Bound hit! ' + str(res.x))
            res.x[i] = 50
    gl.compParamDict[key] = res.x
    print(str(key) + ' has params: ')
    print('henry, dispers: '+ str(res.x))
    print("lEVEL 2 Loss function value: " + str(res.fun))
    return res.fun
