# Nelder-Mead optimization algorithm. Level 2 sub-optimization of main bi-level optimization.
# Minimazes loss value from Lev2_Loss_Function.py and uses Lin_Solver.py or Nonlin_Solver.py
# based on IsoSelect from Iso_Decision.py
from scipy.optimize import minimize
from functions.Lev2_Loss_Function import Lev2_Loss_Function
import functions.global_ as gl
from scipy.optimize import shgo


def Lev2_Optim(porosity, experimentCluster, key, lossFunction, factor, solver):
    #print("Calling Lev2_Optim with params " + str(gl.compParamDict[key]) + "!")
    #res = minimize(Lev2_Loss_Function, gl.compParamDict[key], args=(experimentCluster, porosity), bounds=((0, None), (0, None)), method='Nelder-Mead', options={'fatol': 0.5,'maxfev': 25})
    print("Calling Lev2_Optim with:\nK " +
          str(round(gl.compParamDict[key][0], 2)) +
          " and range [" +
          str(round(gl.compRangeDict[key][0][0], 2)) +
          ", " +
          str(round(gl.compRangeDict[key][0][1], 2)) +
          "]!\nD " +
          str(round(gl.compParamDict[key][1], 2)) +
          " and range [" +
          str(round(gl.compRangeDict[key][1][0], 2)) +
          ", " +
          str(round(gl.compRangeDict[key][1][1], 2)) +
          "]!")
    if solver == "Lin":
        res = shgo(func = lambda x : Lev2_Loss_Function(gl.compParamDict[key], experimentCluster, porosity, lossFunction, factor, solver),
                       bounds=((gl.compRangeDict[key][0][0], gl.compRangeDict[key][0][1]), (gl.compRangeDict[key][1][0], gl.compRangeDict[key][1][1])),
                       args=(),
                       options={'f_tol': 0.1})
    elif solver == "Nonlin":
        res = shgo(func = lambda x : Lev2_Loss_Function(gl.compParamDict[key], experimentCluster, porosity, lossFunction, factor, solver),
                       bounds=((gl.compRangeDict[key][0][0], gl.compRangeDict[key][0][1]), (gl.compRangeDict[key][1][0], gl.compRangeDict[key][1][1]), (gl.compRangeDict[key][2][0], gl.compRangeDict[key][2][1])),
                       args=(),
                       options={'f_tol': 0.1})
    print('__________________________________________')

    '''for i in [0,1]:
        #if res.x[i] == 0 or res.x[i] == 1000:
        if res.x[i] >= 15000:
            print('Bound hit! ' + str(res.x.round(2)))
            res.x[i] = 1000
        elif res.x[i] == 0:
            print('Bound hit! ' + str(res.x.round(2)))
            res.x[i] = 50'''
    gl.compParamDict[key] = res.x
    gl.lv2LossFunctionVals[key] = res.fun
    print(str(key) + ' has params: ')
    print('henry, dispers: '+ str(res.x.round(2)))
    print("lEVEL 2 Loss function value: " + str(round(res.fun, 2)))
    return res.fun
