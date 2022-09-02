import numpy as np
import matplotlib.pyplot as plt
from functions.Lev2_Loss_Function import Lev2_Loss_Function
from objects.ExperimentClusters import ExperimentClusters

"""
x axis - Henry constant
y axis - Dispersion coeficient
"""
def Loss_Function_Analysis(experimentClusterComp,
                            component = 'Sac',
                            xstart = 0,
                            ystart = 0,
                            xend = 5002,
                            yend = 5002,
                            xstep = 50,
                            ystep = 50,
                            porosityStart = 0.2,
                            porosotyEnd = 1,
                            porosityStep = 0.1):
    experimentCluster = experimentClusterComp.clusters[component]
    for porosity in np.arange(porosityStart, porosotyEnd, porosityStep):
        x = 0
        resultArr = np.zeros((len(np.arange(xstart, xend, xstep)), len(np.arange(ystart, yend, ystep))))
        for henryConst in np.arange(xstart, xend, xstep):
            y = 0
            for disperCoef in np.arange(ystart, yend, ystep):
                res = Lev2_Loss_Function([henryConst, disperCoef], experimentCluster, porosity)
                resultArr[x, y] = res
                y += 1
            x += 1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(ystart, yend, ystep), np.arange(xstart, xend, xstep))
        Z = resultArr
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('Dispersion Coeficient')
        ax.set_ylabel('Henry Constant')
        ax.set_zlabel('Loss Function Value')
        ax.set_title('porosity = ' + str(porosity))
        xindex = np.argwhere(resultArr == np.min(resultArr))[0][0]
        yindex = np.argwhere(resultArr == np.min(resultArr))[0][1]
        print("Minimum:")
        print("Hentry Constant = " + str(xstart + (xindex*xstep)))
        print("Dispersion Coeficient = " + str(ystart + (yindex*ystep)))
        plt.show()
        while True:
            i = input("Print closeup?[Y - yes, N - no, E - exit]")
            if i == "Y":
                xstart2 = int(input("Henry Constant start?"))
                xend2 = int(input("Henry Constant end?"))
                xstep2 = int(input("Henry Constant step?"))
                ystart2 = int(input("Dispersion Coeficient start?"))
                yend2 = int(input("Dispersion Coeficient end?"))
                ystep2 = int(input("Dispersion Coeficient step?"))
                Loss_Function_Analysis(experimentClusterComp, component, xstart2, ystart2, xend2, yend2, xstep2, ystep2, porosity, porosity+(porosityStep/2), porosityStep)
                break
            if i == "N":
                break
            if i == "E":
                return