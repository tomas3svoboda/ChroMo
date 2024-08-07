import numpy as np
import pandas as pd
import math
from scipy import optimize
import matplotlib.pyplot as plt
from numba import jit
import warnings

# ***Numerical solution of Equilibrium Dispersive Model of Chromatography***
#                       ***Langmuir isotherm***
#                 ***Cranck-Nicolson Implicid Method***
# Developed by Ing. Tomas Svoboda
# University of Chemistry and Technology in Prague, Czech Republic
# Department of Carbohydrates and Cereals
# tomas3.svoboda@vscht.cz
def Nonlin_Solver(flowRate = 500,       # Volume flowrate in [mL/h]
                  length=235,         # Length of the packed section in the column [mm]
                  diameter=16,        # Column diameter [mm]
                  feedVol=30,         # Feed injection volume [mL]
                  feedConc=150e-3,    # Concentration of the balanced component in the feed [g/mL] or [mg/mm^3]
                  porosity=0.4,       # Total porosity of the adsorbent packing [-]
                  langmuirConst=1.5,  # Langmuir's constant of the linear isotherm [-]
                  disperCoef=5,       # Axial dispersion coefficient [mm^2/s]
                  saturCoef=15,       # Saturation Coefficient
                  Nx=30,              # Number of spatial differences - Nx
                  Nt=3000,            # Number of time differences - Nt
                  time=3000,          # Finite time of the experiment [s]
                  denserFeed=True,
                  denseSparseRatio=0.4,  # define ratio between dense and sparse steps
                  debugPrint=False,
                  debugGraph=False,
                  full=False
                  ):
    '''This script implements Cranck-Nicolson implicit method for numerical solution of
    non-linear second order convection-diffusion PDE used to describe concentration
    wave propagation trough the chromatographic column. Numerical scheme utilizes
    averaged centred difference scheme in spatial direction and forward difference
    scheme in time direction. Danckwert's boudaries are implemented with usage of
    fictious point for left boundary.
    '''
    # Ignore runtime warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Calculation of the feed time [s]
    feedTime = (feedVol / flowRate) * 3600

    # Calculation of the flow speed [mm/s]
    flowSpeed = (flowRate * 1000/3600) / ((math.pi * (diameter**2) / 4) * porosity)
    if debugPrint:
        print('Flow speed:   ' + str(round(flowSpeed, 2)) + ' [mm/s]')
        print('Saturation Coefficient:   ' + str(round(saturCoef, 2)) + ' g/L')
        print('Langmuir Constant:   ' + str(round(langmuirConst, 2)))
        print('Dispersion Coefficient:   ' + str(round(disperCoef, 2)) + ' mm2/s')

    # ________________________________________________________________________
    # DATA STRUCTURES PREPARATION

    x = np.linspace(0, length, Nx)  # Preparation of space vector
    dx = length / Nx  # Calculating space step [mm]

    denseSparseRatio = 0.6  # define ratio between dense and sparse steps

    dense_steps = int(Nt * denseSparseRatio)  # Determine number of dense steps
    sparse_steps = Nt - dense_steps  # Determine number of sparse steps

    dense_time = feedTime + feedTime * ((time/feedTime)/20) # formula for calculating time of dense steps

    # Create time vector with varying step sizes
    t_dense = np.linspace(0, dense_time, dense_steps)  # Dense grid
    t_sparse = np.linspace(dense_time, time, sparse_steps)  # Sparse grid
    t = np.concatenate((t_dense, t_sparse))  # Combined time vector
    dt_dense = t_dense[1] - t_dense[0] # Time step size for dense grid
    dt_sparse = t_sparse[1] - t_sparse[0] # Time step size for sparse grid

    # Constructing pulse injection feed vector
    feedSteps = int(feedTime // dt_dense)  # Whole number of feed iterations
    feed = np.zeros(Nt)  # Initialize feed vector

    # Set feed concentration values
    for i in range(dense_steps):
        if i == 0:
            feed[i] = feedConc/2
        elif i <= feedSteps:
            feed[i] = feedConc
        elif i == feedSteps+1:
            feed[i] = feedConc/2
        else:
            feed[i] = 0

    # Preparation of the solution matrix
    c = np.zeros((Nt, Nx))
    # Initial conditions
    c0 = np.zeros(len(x))

    # ________________________________________________________________________
    # DISCRETIZATION

    # Following function evaluates each function in the resulting system
    # of nonlinear algebraic equations. The solution of this system is that c1
    # which corresponds to vector f filled with zeros. c1 is unknown next time step.
    # c0 is previous calculated time step. For time 0 c0 in zeros (initial condition)

    dx_squared = dx ** 2
    dx_twice = dx * 2

    @jit(nopython=True)
    def function(c1,
                 c0,
                 feedCur,
                 porosity,
                 langmuirConst,
                 saturCoef,
                 disperCoef,
                 flowSpeed,
                 dt):
        f = np.zeros(len(c0))  # Preparation of solution vector - will be optimized to 0
        for i in range(0, len(c0)):  # Main loop through all the vector's elements
            if i == 0:  # Left boundary
                f[0] = ((((c0[1] - c0[0]) / dx) + ((c1[1] - c1[0]) / dx)) / 2) - (flowSpeed * (c1[0] - feedCur))
            elif i > 0 and i < Nx - 1:
                denominator0 = ((1 - porosity) * saturCoef * langmuirConst) / (
                            (((langmuirConst * c0[i] + 1) ** 2) * porosity) + 1)
                denominator1 = ((1 - porosity) * saturCoef * langmuirConst) / (
                            (((langmuirConst * c1[i] + 1) ** 2) * porosity) + 1)
                secondDer0 = (c0[i - 1] - 2 * c0[i] + c0[i + 1]) / (dx_squared)
                secondDer1 = (c1[i - 1] - 2 * c1[i] + c1[i + 1]) / (dx_squared)
                firstDer0 = (c0[i + 1] - c0[i - 1]) / (dx_twice)
                firstDer1 = (c1[i + 1] - c1[i - 1]) / (dx_twice)
                timeDer = (c1[i] - c0[i]) / dt
                disperElem = ((disperCoef / denominator0 * secondDer0) + (disperCoef / denominator1 * secondDer1)) / 2
                convElem = ((flowSpeed / denominator0 * firstDer0) + (flowSpeed / denominator1 * firstDer1)) / 2
                f[i] = disperElem - convElem - timeDer
            elif i == Nx - 1:  # Right boundary
                f[Nx - 1] = (((c0[Nx - 1] - c0[Nx - 2]) / dx) + ((c1[Nx - 1] - c1[Nx - 2]) / dx)) / 2
        return f

    # ________________________________________________________________________
    # SOLUTION ALGORITHM

    residuals = np.zeros(Nt)  # Initialize a vector to store the residuals

    for i in range(1, Nt):
        if debugPrint:
            if i == 1:
                print('\nSolution algorithm has been started:')
            if i % (Nt // 20) == 0:
                print(str(i) + ' steps has been finished ... ' +
                      str(Nt - i) + ' steps remain.')

        if denserFeed:
            if i <= dense_steps:
                dt = dt_dense
            else:
                dt = dt_sparse

        options = {'fatol': 6e-4, 'jac_options': {'method': 'lgmres'}}

        sol = optimize.root(fun=function,
                            x0=c[i - 1, :],
                            method='krylov',
                            args=(c[i - 1, :], feed[i], porosity, langmuirConst, saturCoef, disperCoef, flowSpeed, dt),
                            options=options
                            )
        c[i, :] = sol.x
        residuals[i] = np.linalg.norm(sol.fun)  # Save the L2-norm of the residuals at each time step
    # ________________________________________________________________________
    # MASS BALANCE CHECK

    feedMass = feedVol * feedConc  # Calculating teoretical mass fed into system
    massCumulOut = 0  # Mass cumulation over time in outlet from the column
    massCumulIn = 0  # Mass cumulation over time in inlet to the column

    for i in range(0, Nt):  # Calculation of mass cumulation over time
        actConcOut = c[i, -1]
        massCumulOut += (dt * flowRate * actConcOut / 3600)

    # Calculation of differece between mass in feed and in the outlet
    massDifferenceOut = feedMass - massCumulOut
    massDifferenceIn = feedMass - massCumulIn
    massDifferencePerc = massDifferenceOut * 100 / feedMass
    # Display mass balance check
    if debugPrint:
        print('\nFeed Mass:   ' + str(round(feedMass, 2)) + ' mg')
        print('Outlet Mass:   ' + str(round(massCumulOut, 2)) + ' mg')
        print('Difference:   ' + str(round(-(massDifferenceOut), 2)) + ' mg   '
              + str(round((massDifferenceOut * 100 / feedMass), 2)) + ' %\n')

    if debugPrint:
        results = pd.DataFrame(c)
        print('Complete results mesh of size')
        print('Number of Elements: ' + str(results.size))
        print('Shape of the solution matrix: ' + str(results.shape))

    # ________________________________________________________________________
    # PLOTTING

    # Plotting solution 3D mesh
    if debugGraph:
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(projection='3d')
        X, Y = np.meshgrid(x, t)
        Z = c
        ax1.plot_surface(X, Y, Z)
        ax1.set_xlabel('Lenght [mm]')
        ax1.set_ylabel('Time [s]')
        ax1.set_zlabel('Concentration [mg/mL]')
        plt.savefig('3D_surface_plot_' + str(Nt) + 'x' + str(Nx) + '_' + str(int(round(Nt / Nx, 0))) \
                    + '.png')
        plt.show()

        # Ploting concentration-time curve in the column outlet
        x_plot = np.round(np.linspace(0, Nx, 10)).astype(int)
        fig2 = plt.figure(2)
        plt.plot(t, c[:, -1])
        plt.title('Outlet concentration-time')
        # plt.savefig('plot_' + str(Nt)+ 'x' +str(Nx) + '_' + str(int(round(Nt/Nx,0)))\
        #            + '.png')
        plt.show()

        # Ploting concentration-time curve in the column inlet
        fig3 = plt.figure(3)
        plt.plot(t, c[:, 0])
        plt.title('Input concentration-time')
        # plt.savefig('plot_' + str(Nt)+ 'x' +str(Nx) + '_' + str(int(round(Nt/Nx,0)))\
        #            + '.png')
        plt.show()

        # Ploting concentration time-curve in the outlet from the column
        x_plot = np.round(np.linspace(0, Nx, 10)).astype(int)
        fig2 = plt.figure(2)
        plt.plot(t, feed, label='feed')
        plt.plot(t, c[:, 0], label=(str(round(x_plot[0] * dx, 0)) + ' mm'))
        plt.plot(t, c[:, (x_plot[1])], label=(str(round(x_plot[1] * dx, 0)) + ' mm'))
        plt.plot(t, c[:, (x_plot[2])], label=(str(round(x_plot[2] * dx, 0)) + ' mm'))
        plt.plot(t, c[:, (x_plot[3])], label=(str(round(x_plot[3] * dx, 0)) + ' mm'))
        plt.plot(t, c[:, (x_plot[4])], label=(str(round(x_plot[4] * dx, 0)) + ' mm'))
        plt.plot(t, c[:, (x_plot[5])], label=(str(round(x_plot[5] * dx, 0)) + ' mm'))
        plt.plot(t, c[:, (x_plot[6])], label=(str(round(x_plot[6] * dx, 0)) + ' mm'))
        plt.plot(t, c[:, (x_plot[7])], label=(str(round(x_plot[7] * dx, 0)) + ' mm'))
        plt.plot(t, c[:, (x_plot[8])], label=(str(round(x_plot[8] * dx, 0)) + ' mm'))
        plt.plot(t, c[:, -1], label=(str(round(Nx * dx, 0)) + ' mm'))
        plt.legend()
        plt.savefig('Concentration_time_plot_' + str(Nt) + 'x' + str(Nx) + '_' + str(int(round(Nt / Nx, 0))) \
                    + '.png')
        plt.show()

        # Plotting iteration residuals
        plt.plot(t, residuals)  # Plot the residuals over time
        plt.xlabel('Time [s]')
        plt.ylabel('Residuals')
        plt.show()
    if full:
        return [c, t, feed, residuals, massDifferencePerc]
    return [c, t]
