from functions.solvers.Solver_Choice import Solver_Choice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Solver_Analysis(experimentSet, componentList, paramList, solver="Lin"):
    time = np.linspace(0, 10800, 3000)
    resultsDict = dict()
    cntr = 0
    for exp in experimentSet.experiments:
        cntr += 1
        for comp in exp.experimentComponents:
            for compName, params in zip(componentList, paramList):
                if(compName == comp.name):
                    result = Solver_Choice(solver, params, comp)[:, -1]
                    key = compName + "_" + str(cntr)
                    resultsDict[key] = result
                    plt.plot(time, result, label=key)
    plt.legend()
    plt.show()
    i = input("Save to csv?[Y - yes, N - no]")
    if i == "Y":
        path = input("Path?")
        pandasResult = pd.DataFrame({'time': time})
        for key, val in resultsDict.items():
            pandasResult[key] = val
        pandasResult.to_csv(path, index=False, compression=None)