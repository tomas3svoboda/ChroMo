# Function which calls Lev1_Optim.py.
# Initializes bi-level optimization proces based on IsoSelect and InitParams from Iso_Decision.py.
# Generates FinalResults data object.
from functions.Lev1_Optim import Lev1_Optim
import functions.global_ as gl
from functions.solvers.Lin_Solver import Lin_Solver


def Bilevel_Optim(experimentSetCor3, experimentClustersComp, porosityIntervals, KDIntervals, lossFunction, factor, solver, spacialDiff = 30, timeDiff = 3000, time = 10800, optimId=1):
    print("Calling Bilevel_Optim!")
    gl.compParamDict[optimId] = {}
    gl.compRangeDict[optimId] = {}
    gl.lossFunctionProgress[optimId] = {}
    gl.lv2LossFunctionVals[optimId] = {}
    for key in experimentClustersComp.clusters:
        gl.compParamDict[optimId][key] = [KDIntervals[key]["kinit"], KDIntervals[key]["dinit"], KDIntervals[key]["qinit"]]
        gl.compRangeDict[optimId][key] = [KDIntervals[key]["krange"], KDIntervals[key]["drange"], KDIntervals[key]["qrange"]]
    gl.porosity[optimId] = porosityIntervals["init"]
    gl.porosityRange[optimId] = porosityIntervals["range"]
    Lev1_Optim(experimentClustersComp, lossFunction, factor, solver, spacialDiff, timeDiff, time, optimId)
    result = dict()
    result["solver"] = solver
    result["porosity"] = gl.porosity[optimId]
    result["lv1lossfunctionval"] = gl.lv1LossFunctionVal[optimId]
    result["compparams"] = gl.compParamDict[optimId]
    result["lv2lossfunctionvals"] = gl.lv2LossFunctionVals[optimId]
    result["lossfunctionprogress"] = gl.lossFunctionProgress[optimId]

    '''cond = experimentSetCor3.experiments[0].experimentCondition
    comp = experimentSetCor3.experiments[0].experimentComponents[0]
    res = Lin_Solver(cond.flowRate, cond.columnLength, cond.columnDiameter, cond.feedVolume, comp.feedConcentration, gl.porosity, gl.compParamDict['Sac'][0], gl.compParamDict['Sac'][1], debugPrint=True)
    t = np.linspace(0, 10800, 200)
    plt.plot(t, res[:, -1])
    comp.concentrationTime.plot.line(x=0)
    plt.show()'''
    return result # TODO return Solution object
