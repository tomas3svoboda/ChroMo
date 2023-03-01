import numpy as np
from matplotlib import pyplot as plt
from functions.solvers.Solver_Choice import Solver_Choice

def Model_Analysis(experimentComp, solver, params, spacialDiff = 30, timeDiff = 3000, time = 10800, webMode = False, title = False, full = False):
    df = experimentComp.concentrationTime
    model = Solver_Choice(solver, params, experimentComp, spacialDiff, timeDiff, time, full=full)
    if full:
        modelCurve = model[0][:, -1]
    else:
        modelCurve = model[:, -1]
    minTime = df.iat[0, 0]
    maxTime = df.iat[-1, 0]
    timeSpace = np.linspace(minTime, maxTime, modelCurve.size)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Concentration [mg/mL]")
    ax.plot(timeSpace, modelCurve)
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], color='r', marker=',', s=10)
    if full:
        fig2 = plt.figure(2)
        ax1 = fig2.add_subplot(projection='3d')
        length = experimentComp.experiment.experimentCondition.columnLength
        x = np.linspace(0, length, spacialDiff)
        t = np.linspace(0, time, timeDiff)
        X, Y = np.meshgrid(x, t)
        ax1.plot_surface(X, Y, model[0])
        ax1.set_xlabel('Lenght [mm]')
        ax1.set_ylabel('Time [s]')
        ax1.set_zlabel('Concentration [mg/mL]')

        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        ax3.plot(t, model[0][:, 0])
        ax3.set_title('Input concentration-time')

        if solver == "Nonlin":
            fig4 = plt.figure(4)
            ax4 = fig4.add_subplot(111)
            ax4.plot(t, model[2])  # Plot the residuals over time
            ax4.set_xlabel('Time [s]')
            ax4.set_ylabel('Residuals')

        dx = length / spacialDiff
        x_plot = np.round(np.linspace(0, spacialDiff, 10)).astype(int)
        fig5 = plt.figure(5)
        ax5 = fig5.add_subplot(111)
        ax5.plot(t, model[1], label='feed')
        ax5.plot(t, model[0][:, 0], label=(str(round(x_plot[0] * dx, 0)) + ' mm'))
        ax5.plot(t, model[0][:, (x_plot[1])], label=(str(round(x_plot[1] * dx, 0)) + ' mm'))
        ax5.plot(t, model[0][:, (x_plot[2])], label=(str(round(x_plot[2] * dx, 0)) + ' mm'))
        ax5.plot(t, model[0][:, (x_plot[3])], label=(str(round(x_plot[3] * dx, 0)) + ' mm'))
        ax5.plot(t, model[0][:, (x_plot[4])], label=(str(round(x_plot[4] * dx, 0)) + ' mm'))
        ax5.plot(t, model[0][:, (x_plot[5])], label=(str(round(x_plot[5] * dx, 0)) + ' mm'))
        ax5.plot(t, model[0][:, (x_plot[6])], label=(str(round(x_plot[6] * dx, 0)) + ' mm'))
        ax5.plot(t, model[0][:, (x_plot[7])], label=(str(round(x_plot[7] * dx, 0)) + ' mm'))
        ax5.plot(t, model[0][:, (x_plot[8])], label=(str(round(x_plot[8] * dx, 0)) + ' mm'))
        ax5.plot(t, model[0][:, -1], label=(str(round(spacialDiff * dx, 0)) + ' mm'))
        ax5.legend()
    if not webMode:
        plt.show()