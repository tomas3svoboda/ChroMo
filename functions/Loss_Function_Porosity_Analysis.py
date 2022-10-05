import numpy as np
import matplotlib.pyplot as plt
from functions.Lev2_Loss_Function import Lev2_Loss_Function
from objects.ExperimentClusters import ExperimentClusters

def Loss_Function_Porosity_Analysis(experimentClusterComp,
                            component = 'Sac',
                            henryConst = 20,
                            disperCoef = 20,
                            porosityStart = 0.2,
                            porosotyEnd = 1,
                            porosityStep = 0.1,
                            lossFunctionChoice = "Simple"):
    experimentCluster = experimentClusterComp.clusters[component]
    resultArr = []
    porosityArr = []
    for porosity in np.arange(porosityStart, porosotyEnd, porosityStep):
        res = Lev2_Loss_Function([henryConst, disperCoef], experimentCluster, porosity, lossFunctionChoice)
        resultArr.append(res)
        porosityArr.append(porosity)
        print(str(round((porosity-porosityStart) / ((porosotyEnd-((porosotyEnd-porosityStart)%porosityStep))-porosityStart) * 100)) + "%")
    plt.plot(porosityArr, resultArr)
    plt.show()