from functions.Lev1_Optim import Lev1_Optim
import functions.global_ as gl

def Bilevel_Optim(experimentSetCor3, experimentClustersComp, Lvl1ParamDict, Lvl2ParamDict, lossFunction, factor,
                  solver, spacialDiff=30, timeDiff=3000, time=10800, optimId=1, lvl1optim=None, lvl2optim=None, optimType=None):
    """Starts bilevel optimization"""
    print("Calling Bilevel_Optim!")

    # Initialize dictionaries to store computation parameters, ranges, and loss function values
    gl.lvl2ParamDict[optimId] = {}
    gl.lvl2RangeDict[optimId] = {}
    gl.lossFunctionProgress[optimId] = {}
    gl.lv2LossFunctionVals[optimId] = {}

    # Iterate over experiment clusters
    for key in experimentClustersComp.clusters:
        if solver == "Lin":
            # Store computation parameters for linear solver
            if optimType == "singlelevel":
                gl.lvl2ParamDict[optimId][key] = [Lvl2ParamDict[key]["pinit"], Lvl2ParamDict[key]["kinit"], Lvl2ParamDict[key]["dinit"]]
                gl.lvl2RangeDict[optimId][key] = [Lvl2ParamDict[key]["prange"], Lvl2ParamDict[key]["krange"], Lvl2ParamDict[key]["drange"]]
            elif optimType == "bilevel":
                gl.lvl2ParamDict[optimId][key] = [Lvl2ParamDict[key]["kinit"], Lvl2ParamDict[key]["dinit"]]
                gl.lvl2RangeDict[optimId][key] = [Lvl2ParamDict[key]["krange"], Lvl2ParamDict[key]["drange"]]
            elif optimType == "calcDisper":
                gl.lvl2ParamDict[optimId][key] = [Lvl2ParamDict[key]["kinit"]]
                gl.lvl2RangeDict[optimId][key] = [Lvl2ParamDict[key]["krange"]]
            else:
                raise Exception("Unknown optimization type in Bilevel_Optim.")
        elif solver == "Nonlin":
            # Store computation parameters for nonlinear solver
            if optimType == "singlelevel":
                gl.lvl2ParamDict[optimId][key] = [Lvl2ParamDict[key]["pinit"], Lvl2ParamDict[key]["kinit"],
                                                  Lvl2ParamDict[key]["dinit"], Lvl2ParamDict[key]["qinit"]]
                gl.lvl2RangeDict[optimId][key] = [Lvl2ParamDict[key]["prange"], Lvl2ParamDict[key]["krange"],
                                                  Lvl2ParamDict[key]["drange"], Lvl2ParamDict[key]["qrange"]]
            elif optimType == "bilevel":
                gl.lvl2ParamDict[optimId][key] = [Lvl2ParamDict[key]["kinit"], Lvl2ParamDict[key]["dinit"],
                                                  Lvl2ParamDict[key]["qinit"]]
                gl.lvl2RangeDict[optimId][key] = [Lvl2ParamDict[key]["range"], Lvl2ParamDict[key]["drange"],
                                                  Lvl2ParamDict[key]["qrange"]]
            elif optimType == "calcDisper":
                gl.lvl2ParamDict[optimId][key] = [Lvl2ParamDict[key]["kinit"], Lvl2ParamDict[key]["qinit"]]
                gl.lvl2RangeDict[optimId][key] = [Lvl2ParamDict[key]["krange"], Lvl2ParamDict[key]["qrange"]]
            else:
                raise Exception("Unknown optimization type in Bilevel_Optim.")
    # Store porosity parameters
    if optimType == "singlelevel":
        gl.lvl1ParamDict[optimId] = []
        gl.lvl1RangeDict[optimId] = []
    elif optimType != "calcDisper":
        gl.lvl1ParamDict[optimId] = [Lvl1ParamDict["pinit"]]
        gl.lvl1RangeDict[optimId] = [Lvl1ParamDict["prange"]]
    else:
        gl.lvl1ParamDict[optimId] = [Lvl1ParamDict["pinit"], Lvl1ParamDict["ainit"]]
        gl.lvl1RangeDict[optimId] = [Lvl1ParamDict["prange"], Lvl1ParamDict["arange"]]


    # Call Lev1_Optim function
    Lev1_Optim(experimentClustersComp, lossFunction, factor, solver, spacialDiff, timeDiff, time, optimId, lvl1optim,
               lvl2optim, optimType)

    # Build the result dictionary
    result = {}
    result["optimparams"] = {}
    result["optimparams"]["Lvl1ParamDict"] = Lvl1ParamDict
    result["optimparams"]["Lvl2ParamDict"] = Lvl2ParamDict
    result["optimparams"]["lossFunction"] = lossFunction
    result["optimparams"]["factor"] = factor
    result["optimparams"]["solver"] = solver
    result["optimparams"]["lvl1optim"] = lvl1optim
    result["optimparams"]["lvl2optim"] = lvl2optim
    result["optimparams"]["optimType"] = optimType
    result["bestLvl1Params"] = gl.bestLvl1ParamDict[optimId]
    result["bestLvl1LossFunctionVal"] = gl.bestLvl1LossFunctionVal[optimId]
    result["bestLvl2Params"] = gl.bestLvl2ParamDict[optimId]
    result["lv2lossfunctionvals"] = gl.bestLvl2LossFunctionVals[optimId]
    result["lossfunctionprogress"] = gl.lossFunctionProgress[optimId]

    return result