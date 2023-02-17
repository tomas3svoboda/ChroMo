import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions.Lev2_Loss_Function import Lev2_Loss_Function
from objects.ExperimentClusters import ExperimentClusters

"""
x axis - Henry constant
y axis - Dispersion coeficient
"""
def Loss_Function_Analysis_Simple(experimentClusterComp,
                            component,
                            path,
                            xstart,
                            ystart,
                            xend,
                            yend,
                            xstep,
                            ystep,
                            porosity,
                            satur,
                            lossFunctionChoice = "Squares",
                            factor = 1,
                            solver = "Lin",
                            webMode = False,
                            fileName = ""):
    experimentCluster = experimentClusterComp.clusters[component]
    x = 0
    resultArr = np.zeros((len(np.arange(xstart, xend, xstep)), len(np.arange(ystart, yend, ystep))))
    for henryLangConst in np.arange(xstart, xend, xstep):
        y = 0
        for disperCoef in np.arange(ystart, yend, ystep):
            res = Lev2_Loss_Function([henryLangConst, disperCoef, satur], experimentCluster, porosity, lossFunctionChoice, factor, solver)
            resultArr[x, y] = res
            y += 1
        x += 1
        endpoint = xend-((xend-xstart)%xstep)
        if webMode:
            yield str(round((henryLangConst-xstart) / (endpoint-xstart) * 100)) + "%"
        else:
            print(str(round((henryLangConst-xstart) / (endpoint-xstart) * 100)) + "%")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(ystart, yend, ystep), np.arange(xstart, xend, xstep))
    Z = resultArr
    ax.plot_surface(X, Y, Z, alpha=0.5)
    ax.set_xlabel('Dispersion Coeficient')
    ax.set_ylabel('Henry Constant')
    ax.set_zlabel('Loss Function Value')
    ax.set_title('porosity = ' + str(porosity))
    xindex = np.argwhere(resultArr == np.min(resultArr))[0][0]
    yindex = np.argwhere(resultArr == np.min(resultArr))[0][1]
    ax.scatter(ystart + (yindex*ystep), xstart + (xindex*xstep), resultArr[xindex, yindex], color='r', marker=',', s=10)
    tmparr = np.insert(resultArr, 0 , np.arange(xstart, xend, xstep), axis=1)
    tmparr2 = ["henry const\\disper coef"]
    tmparr2.extend([i for i in np.arange(ystart, yend, ystep)])
    resultMatrix = pd.DataFrame(tmparr, columns=tmparr2)
    resultMatrix.set_index('henry const\\disper coef')
    if webMode:
        angle = 20
        while True:
            ax.view_init(20, angle)
            angle += 60
            yield [str(xstart + (xindex*xstep)), str(ystart + (yindex*ystep)), resultMatrix]
    else:
        print("Minimum:")
        print("Hentry Constant = " + str(xstart + (xindex*xstep)))
        print("Dispersion Coeficient = " + str(ystart + (yindex*ystep)))
        plt.show()
        save = input("Save the plot?[Y - yes]:")
        if save == "Y":
            fileName = input("Enter filename:")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z)
            ax.set_xlabel('Dispersion Coeficient')
            ax.set_ylabel('Henry Constant')
            ax.set_zlabel('Loss Function Value')
            ax.set_title('porosity = ' + str(porosity))
            plt.savefig(path + "\\" + fileName)
            plt.cla()
        yield "DONE"