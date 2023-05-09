# Nelder-Mead optimization algorithm. Level 2 sub-optimization of main bi-level optimization.
# Minimazes loss value from Lev2_Loss_Function.py and uses Lin_Solver.py or Nonlin_Solver.py
# based on IsoSelect from Iso_Decision.py
from functions.Lev2_Loss_Function import Lev2_Loss_Function
import functions.global_ as gl
from functions.handle_Optim_Settings import handle_Optim_Settings


def Lev2_Optim(porosity, experimentCluster, key, lossFunction, factor, solver, spacialDiff = 30, timeDiff = 3000, time = 10800, optimId=1, lvl2optim=None):
    #print("Calling Lev2_Optim with params " + str(gl.compParamDict[key]) + "!")
    #res = minimize(Lev2_Loss_Function, gl.compParamDict[key], args=(experimentCluster, porosity), bounds=((0, None), (0, None)), method='Nelder-Mead', options={'fatol': 0.5,'maxfev': 25})
    '''print("Calling Lev2_Optim with:\nK " +
          str(round(gl.compParamDict[optimId][key][0], 2)) +
          " and range [" +
          str(round(gl.compRangeDict[optimId][key][0][0], 2)) +
          ", " +
          str(round(gl.compRangeDict[optimId][key][0][1], 2)) +
          "]!\nD " +
          str(round(gl.compParamDict[optimId][key][1], 2)) +
          " and range [" +
          str(round(gl.compRangeDict[optimId][key][1][0], 2)) +
          ", " +
          str(round(gl.compRangeDict[optimId][key][1][1], 2)) +
          "]!")'''
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
    # print("optimId: " + str(optimId) + " lvl2alg: " + str(lvl2optim["algorithm"]))
    res = handle_Optim_Settings(Lev2_Loss_Function,
                                gl.compParamDict[optimId][key],
                                (experimentCluster, porosity, lossFunction, factor, solver, spacialDiff, timeDiff, time, optimId),
                                bnds,
                                lvl2optim)
    # print('__________________________________________')

    '''for i in [0,1]:
        #if res.x[i] == 0 or res.x[i] == 1000:
        if res.x[i] >= 15000:
            print('Bound hit! ' + str(res.x.round(2)))
            res.x[i] = 1000
        elif res.x[i] == 0:
            print('Bound hit! ' + str(res.x.round(2)))
            res.x[i] = 50'''
    if lvl2optim["algorithm"] == "1":
        gl.compParamDict[optimId][key] = res[0]
        gl.lv2LossFunctionVals[optimId][key] = res[1]
    else:
        gl.compParamDict[optimId][key] = res.x
        gl.lv2LossFunctionVals[optimId][key] = res.fun
    '''print(str(key) + ' has params: ')
    print('henry, dispers: '+ str(res.x.round(2)))
    print("lEVEL 2 Loss function value: " + str(round(res.fun, 2)))'''
    return gl.lv2LossFunctionVals[optimId][key]
