import numpy as np
from matplotlib import pyplot as plt
from functions.solvers.Solver_Choice import Solver_Choice

def Model_Analysis(experimentComp, solver, params, webMode = False, title = False):
    df = experimentComp.concentrationTime
    modelCurve = Solver_Choice(solver, params, experimentComp)[:, -1]
    minTime = df.iat[0, 0]
    maxTime = df.iat[-1, 0]
    time = np.linspace(minTime, maxTime, modelCurve.size)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Concentration [mg/mL]")
    ax.plot(time, modelCurve)
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], color='r', marker=',', s=10)
    if not webMode:
        plt.show()