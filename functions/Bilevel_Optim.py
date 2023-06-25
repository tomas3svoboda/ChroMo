from functions.Lev1_Optim import Lev1_Optim
import functions.global_ as gl

# Starts bilevel optimization
def Bilevel_Optim(experimentSetCor3, experimentClustersComp, porosityIntervals, KDIntervals, lossFunction, factor,
                  solver, spacialDiff=30, timeDiff=3000, time=10800, optimId=1, lvl1optim=None, lvl2optim=None):
    print("Calling Bilevel_Optim!")

    # Initialize dictionaries to store computation parameters, ranges, and loss function values
    gl.compParamDict[optimId] = {}
    gl.compRangeDict[optimId] = {}
    gl.lossFunctionProgress[optimId] = {}
    gl.lv2LossFunctionVals[optimId] = {}

    # Iterate over experiment clusters
    for key in experimentClustersComp.clusters:
        if solver == "Lin":
            # Store computation parameters for linear solver
            gl.compParamDict[optimId][key] = [KDIntervals[key]["kinit"], KDIntervals[key]["dinit"]]
            gl.compRangeDict[optimId][key] = [KDIntervals[key]["krange"], KDIntervals[key]["drange"]]
        elif solver == "Nonlin":
            # Store computation parameters for nonlinear solver
            gl.compParamDict[optimId][key] = [KDIntervals[key]["kinit"], KDIntervals[key]["dinit"],
                                              KDIntervals[key]["qinit"]]
            gl.compRangeDict[optimId][key] = [KDIntervals[key]["krange"], KDIntervals[key]["drange"],
                                              KDIntervals[key]["qrange"]]

    # Store porosity parameters
    gl.porosity[optimId] = porosityIntervals["init"]
    gl.porosityRange[optimId] = porosityIntervals["range"]

    # Call Lev1_Optim function
    Lev1_Optim(experimentClustersComp, lossFunction, factor, solver, spacialDiff, timeDiff, time, optimId, lvl1optim,
               lvl2optim)

    # Build the result dictionary
    result = {}
    result["optimparams"] = {}
    result["optimparams"]["porosityIntervals"] = porosityIntervals
    result["optimparams"]["KDIntervals"] = KDIntervals
    result["optimparams"]["lossFunction"] = lossFunction
    result["optimparams"]["factor"] = factor
    result["optimparams"]["solver"] = solver
    result["optimparams"]["lvl1optim"] = lvl1optim
    result["optimparams"]["lvl2optim"] = lvl2optim
    result["porosity"] = gl.bestPorosity[optimId]
    result["lv1lossfunctionval"] = gl.bestLvl1LossFunctionVal[optimId]
    result["compparams"] = gl.bestCompParamDict[optimId]
    result["lv2lossfunctionvals"] = gl.bestLvl2LossFunctionVals[optimId]
    result["lossfunctionprogress"] = gl.lossFunctionProgress[optimId]

    return result