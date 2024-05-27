import math
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

# ***Numerical solution of Equilibrium Dispersive Model of Chromatography***
#                  ***Lumped Model; Linear isotherm***
#                 ***Cranck-Nicolson Implicid Method***
# Developed by Ing. Tomas Svoboda
# University of Chemistry and Technology in Prague, Czech Republic
# Department of Carbohydrates and Cereals
# tomas3.svoboda@vscht.cz
'''
This script implements Cranck-Nicolson implicit method for numerical solution of
semi-linear second order convection-diffusion PDE used to describe concentration
wave propagation trough the chromatographic column. Numerical scheme utilizes
averaged centred difference scheme in spatial direction and forward difference
scheme in time direction. Danckwert's boudaries are implemented with usage of
fictious point for left boundary.
'''

def Lin_Solver(flowRate = 50,       # Volume flowrate in [mL/h]
               length=235,          # Length of the packed section in the column [mm]
               diameter=16,         # Column diameter [mm]
               feedVol=15,          # Feed injection volume [mL]
               feedConc=150e-3,     # Concentration of the balanced component in the feed [g/mL] or [mg/mm^3]
               porosity=0.4,        # Total porosity of the adsorbent packing [-]
               henryConst=1.9,      # Henry's constant of the linear isotherm [-]
               disperCoef=12,       # Axial dispersion coefficient [mm^2/s]
               Nx = 30,             # Number of spatial differences
               Nt = 3000,           # Number of time differences
               time=10800,          # Finite time of the experiment [s]
               debugPrint=False,
               debugGraph=False,
               full=False):

    # Calculation of the feed time [s]
    feedTime = (feedVol / flowRate) * 3600

    # Calculation of the flow speed [mm/s]
    flowSpeed = (flowRate * 1000/3600) / ((math.pi * (diameter**2) / 4) * porosity)

    if debugPrint:
        print('Flow speed:   ' + str(round(flowSpeed, 2)) + ' [mm/s]')
        print('Langmuir Constant:   ' + str(round(henryConst, 2)))
        print('Dispersion Coefficient:   ' + str(round(disperCoef, 2)) + ' mm2/s')

    # ________________________________________________________________________
    # DATA STRUCTURES PREPARATION

    x = np.linspace(0, length, Nx)  # Preparation of space vector
    dx = length / Nx  # Calculating space step [mm]

    denseSparseRatio = 0.7  # define ratio between dense and sparse steps

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
        if i <= feedSteps:
            feed[i] = feedConc
        else:
            feed[i] = 0

    # Preparation of the solution matrix
    c = np.zeros((Nt, Nx))
    # Initial conditions
    c0 = np.zeros(len(x))

    # ________________________________________________________________________
    # DISCRETIZATION AND SOLUTION

    for i in range(1, Nt):  # Advance in time
        if debugPrint:
            if i == 1:
                print('Solution algorithm has been started:')
            if i % (Nt // 20) == 0:
                print(str(i) + ' steps has been finished ... ' +
                      str(Nt - i) + ' steps remain.')

        if i <= dense_steps:
            dt = dt_dense
        else:
            dt = dt_sparse

        # Crank-Nicolson matrices preparation
        # A.c(t+1) = B.c(t)x, where c(t+1) and c(t) are vectors of c(x) values
        # Preparation of boundaries in matrix A
        A = np.zeros((Nx, Nx))  # A matrix data structure
        A[0, 0] = flowSpeed/disperCoef + 1/(2*dx)  # Left boundary
        A[0, 1] = -1/(2*dx)  # Left boundary
        A[Nx - 1, Nx - 2] = -1/(2*dx)  # Right boundary
        A[Nx - 1, Nx - 1] = 1/(2*dx)  # Right Boundary
        # Preparation of boundaries in matrix B
        B = np.zeros((Nx, Nx))  # B matrix data structure
        B[0, 0] = -1/(2*dx)  # Left boundary
        B[0, 1] = 1/(2*dx)  # Left boundary
        B[Nx - 1, Nx - 2] = 1/(2*dx)  # Right boundary
        B[Nx - 1, Nx - 1] = -1/(2*dx)  # Right Boundary
        # Filling up Matrices A and B
        # Defining constants
        a = disperCoef/((((1-porosity)*henryConst)/porosity)+1)  # *** !!! PODLE DOKUMENTU
        b = flowSpeed/((((1-porosity)*henryConst)/porosity)+1)  # *** !!! PODLE DOKUMENTU
        for j in range(1, Nx - 1):
            A[j, j - 1] = -((dt*a)/(2*(dx**2)))-((dt*b)/(4*dx))
            A[j, j] = 1 + ((dt*a)/(dx**2))
            A[j, j + 1] = -((dt*a)/(2*(dx**2)))+((dt*b)/(4*dx))
            B[j, j - 1] = ((dt*a)/(2*(dx**2)))+((dt*b)/(4*dx))
            B[j, j] = 1 - ((dt*a)/(dx**2))
            B[j, j + 1] = ((dt*a)/(2*(dx**2)))-((dt*b)/(4*dx))

        b = B.dot(c[i - 1, :])
        b[0] = b[0] + flowSpeed/disperCoef * feed[i]  # From left boundary
        c[i, :] = sparse.linalg.splu(A).solve(b) # Solve linear system of algebraic equations

    if debugPrint:
        feedMass = feedVol * feedConc  # Calculating theoretical mass fed into system
        massCumulOut = 0  # Mass accumulation over time in outlet from the column

        for i in range(0, Nt):  # Calculation of mass accumulation over time
            actConcOut = c[i, -1]
            massCumulOut += (dt * flowRate * actConcOut / 3600)

        # Calculation of difference between mass in feed and in the outlet
        massDifferenceOut = feedMass - massCumulOut
        # Display mass balance check
        print('\nFeed Mass:   ' + str(round(feedMass, 2)) + ' mg')
        print('Outlet Mass:   ' + str(round(massCumulOut, 2)) + ' mg')
        print('Difference:   ' + str(round(-(massDifferenceOut), 2)) + ' mg   '
              + str(round((massDifferenceOut * 100 / feedMass), 2)) + ' %\n')
    if debugGraph:
        '''fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(projection='3d')
        X, Y = np.meshgrid(x, t)
        Z = c
        ax1.plot_surface(X, Y, Z)
        ax1.set_xlabel('Lenght [mm]')
        ax1.set_ylabel('Time [s]')
        ax1.set_zlabel('Concentration [mg/mL]')
        plt.savefig('3D_surface_plot_' + str(Nt) + 'x' + str(Nx) + '_' + str(int(round(Nt / Nx, 0))) \
                    + '.png')
        '''
        plt.plot(t,c[:,-1])
        plt.show()
    if full:
        return [c, t, feed]
    return [c, t]
