import math
import numpy as np
from scipy import linalg
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Numerical solver of the EDM with Linear Isotherm - linear parabolic PDE

def Lin_Solver(flowRate = 50,       # Volume flowrate in [mL/h]
               length=235,          # Lenght of the packed section in the column [mm]
               diameter=16,         # Column diameter [mm]
               feedVol=15,          # Feed injection volume [mL]
               feedConc=150e-3,     # Concentration of the balanced component in the feed [g/mL] or [mg/mm^3]
               porosity=0.4,        # Total porosity of the sorbent packing [-]
               henryConst=1.9,      # Henry's constant of the linear isotherm [-]
               disperCoef=12,       # Axial dispersion coefficient [mm^2/s]
               Nx = 30,             # Number of spatial differences
               Nt = 3000,            # Number of time differences
               time=10800,          # Finite time of the experiment [s]
               debugPrint=False,
               debugGraph=False,
               full=False):

    def diagonal_form(a, lower=1, upper=1):
        # Transforms banded matrix into diagonal ordered form
        # allows to use scipy.linalg.solve_banded
        n = a.shape[1]
        assert (np.all(a.shape == (n, n)))
        ab = np.zeros((2 * n - 1, n))
        for i in range(n):
            ab[i, (n - 1) - i:] = np.diagonal(a, (n - 1) - i)
        for i in range(n - 1):
            ab[(2 * n - 2) - i, :i + 1] = np.diagonal(a, i - (n - 1))
        mid_row_inx = int(ab.shape[0] / 2)
        upper_rows = [mid_row_inx - i for i in range(1, upper + 1)]
        upper_rows.reverse()
        upper_rows.append(mid_row_inx)
        lower_rows = [mid_row_inx + i for i in range(1, lower + 1)]
        keep_rows = upper_rows + lower_rows
        ab = ab[keep_rows, :]
        return ab

    # Calculation of the feed time [s]
    feedTime = (feedVol / flowRate) * 3600
    # Calculation of the flow speed [mm/s]
    flowSpeed = (flowRate * 1000/3600) / ((math.pi * (diameter**2) / 4) * porosity)
    # 1 h = 3600 s
    # 1 mL = 1000 mm^3

    '''print("flowRate: " + str(flowRate))
    print("length: " + str(length))
    print("diameter: " + str(diameter))
    print("feedVol: " + str(feedVol))
    print("feedTime: " + str(feedTime))
    print("feedConc: " + str(feedConc))
    print("flowSpeed: " + str(flowSpeed))
    print("porosity: " + str(porosity))
    print("henryConst: " + str(henryConst))
    print("disperCoef: " + str(disperCoef))
    print("Nx: " + str(Nx))
    print("Nt: " + str(Nt))
    print("time: " + str(time))'''

    # Defining constants
    a = disperCoef/((((1-porosity)*henryConst)/porosity)+1)  # *** !!! PODLE DOKUMENTU
    b = flowSpeed/((((1-porosity)*henryConst)/porosity)+1)  # *** !!! PODLE DOKUMENTU


    x = np.linspace(0, length, Nx)  # Preparation of space vector
    dx = length / Nx  # Calculating space step [mm]
    t = np.linspace(0, time, Nt)  # Preparation of time vector
    dt = time / Nt  # Calculating time step [mm]

    '''if dt > feedTime/10:
        print("WARNING: discretization time step is more than 1/10 of feed time!")'''

    feedSteps = feedTime // dt  # Whole number of feed iterations
    feedTimeAprox = feedTime % dt  # approximation of division

    # Rounding iteration step based on defined feed parameters
    if feedTimeAprox >= dt/2:
        feedSteps += 1
    # Constructing pulse injection feed vector
    feed = np.linspace(0, time, Nt)
    for i in range(0, Nt):
        if i <= feedSteps:
            feed[i] = feedConc
        else:
            feed[i] = 0

    c = np.zeros((len(t), len(x)))

    C_0 = np.zeros(len(x))  # Implementing initial conditions
    c[0, :] = C_0

    # Crank-Nicolson matrices preparation
    # A.c(t+1) = B.c(t)x, where c(t+1) and c(t) are vectors of c(x) values
    # Preparation of boundaries in matrix A
    A = np.zeros((Nx, Nx))  # A matrix data structure
    A[0, 0] = flowSpeed + 1/(2*dx)  # Left boundary
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
    for i in range(1, Nx - 1):
        A[i, i - 1] = -((dt*a)/(2*(dx**2)))-((dt*b)/(4*dx))
        A[i, i] = 1 + ((dt*a)/(dx**2))
        A[i, i + 1] = -((dt*a)/(2*(dx**2)))+((dt*b)/(4*dx))
        B[i, i - 1] = ((dt*a)/(2*(dx**2)))+((dt*b)/(4*dx))
        B[i, i] = 1 - ((dt*a)/(dx**2))
        B[i, i + 1] = ((dt*a)/(2*(dx**2)))-((dt*b)/(4*dx))
    A_diag = diagonal_form(A)

    Aabs = np.abs(A)
    if debugPrint:
        for i in range(0, Nx):
            if Aabs[i, i] <= np.sum(Aabs[i, :]) - Aabs[i, i]:
                print('Matrix A is not strictly diagonally dominant in row ' +
                      str(i) + 'therefore, iterative method may not coverge.\n')
                break
            elif i == Nx - 1:
                print('Matrix A is strictly diagonally dominant,' +
                      'therefore, iterative method will converge!\n')

    # Implementing discretization
    for i in range(1, Nt):  # Advance in time
        b = B.dot(c[i - 1, :])
        b[0] = b[0] + flowSpeed * feed[i]  # From left boundary (start at 1 ???)
        c[i, :] = linalg.solve_banded((1, 1), A_diag, b)
        # c[i,:] = linalg.solve(A, b) # Solve linear system of algebraic equations
        if debugPrint:
            if i == 1:
                print('Solution algorithm has been started:')
            if i % (Nt // 20) == 0:
                print(str(i) + ' steps has been finished ... ' +
                      str(Nt - i) + ' steps remain.')
    if debugPrint:
        feedMass = feedVol * feedConc  # Calculating theoretical mass fed into system
        massCumulOut = 0  # Mass cumulation over time in outlet from the column

        for i in range(0, Nt):  # Calculation of mass cumulation over time
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
        return [c, feed]
    return c
