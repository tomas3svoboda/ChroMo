# Function which calls Lev1_Optim.py.
# Initializes bi-level optimization proces based on IsoSelect and InitParams from Iso_Decision.py.
# Generates FinalResults data object.
from functions.Lev1_Optim import Lev1_Optim
import functions.global_ as gl
from functions.solvers.Lin_Solver import Lin_Solver


def Bilevel_Optim(experimentSetCor3, experimentClustersComp):
    print("Calling Bilevel_Optim!")
    for key in experimentClustersComp.clusters:
        gl.compParamDict[key] = [40, 30]
    gl.porosity = 0.4
    result = Lev1_Optim(experimentClustersComp)
    cond = experimentSetCor3.experiments[0].experimentCondition
    comp = experimentSetCor3.experiments[0].experimentComponents[0]
    res = Lin_Solver(cond.flowRate, cond.columnLength, cond.columnDiameter, cond.feedVolume, comp.feedConcentration, gl.porosity, gl.compParamDict['Sac'][0], gl.compParamDict['Sac'][1], debugPrint=True)
    '''t = np.linspace(0, 10800, 200)
    plt.plot(t, res[:, -1])
    comp.concentrationTime.plot.line(x=0)
    plt.show()'''
    return result #???
