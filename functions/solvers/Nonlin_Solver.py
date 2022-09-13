import numpy as np
import pandas as pd
import math
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from objects.ExperimentSet import ExperimentSet
from objects.Experiment import Experiment
from objects.ExperimentComponent import ExperimentComponent
from functions.Fit_Gauss import Fit_Gauss
# Numerical solver of the EDM with Langmuir Isotherm - nonlinear parabolic PDE

def Nonlin_Solver(flowRate = 200, length = 235, diameter = 16, feedVol = 5, feedConc = 2, porosity = 0.5,
                  langmuirConst = 2.5, disperCoef = 0.95, saturationConst = 1, Nx = 180, Nt = 1000):
    def Danckwert_lb_c2ap(c_i1, c_i0, dx, dt, C1, C2, cIn):
        # First derivative defined as Um/Dax*(c[t,0]-cIn)
        # Second derivative used aproxmation of fictitious point c[t,0-1] = c[t,0+1]
        # Then second order centered difference for second derivative in x
        # First-order Forward difference for first derivative in t
        c_next = c_i0 + ((dt / C1) * (2 * c_i1 - 2 * c_i0) / (dx ** 2)) - ((dt * (C2 ** 2) * (c_i0 - cIn)) / C1)
        return c_next

    def Danckwert_rb_c2ap(c_iN1, c_iN, dx, dt, C1):
        # Second derivative used aproxmation of fictitious point c[t,N+1] = c[t,N-1]
        # Then second order centered difference for second derivative in x
        # First-order Forward difference for first derivative in t
        c_next = c_iN + ((dt * (2 * c_iN1 - 2 * c_iN)) / (C1 * (dx ** 2)))
        return c_next

    def Danckwert_lb_fwrd2(c_i1, c_i2, c_i0, dx, dt, C1, C2, cIn):
        # First derivative defined as Um/Dax*(c[t,0]-cIn) for x = 0
        # Second derivative uses second-order forward differential scheme with c[t,1] and c [t,2] in x
        # First-order Forward difference for first derivative in t
        c_next = c_i0 + ((dt / C1) * ((c_i2 - 2 * c_i1 + c_i0) / (dx ** 2))) - ((dt * (C2 ** 2) * (c_i0 - cIn)) / C1)
        return c_next

    # -------------------------------------------------------------------------------
    # BODIES

    def c1c2x_fwrdt(c_in, c_i1, c_in1, dx, dt, C1, C2):
        # First order centered difference for first order derivative in x
        # Second order centered difference for second derivative in x
        # First-order Forward difference for first derivative in t
        c_next = c_in + ((dt / C1) * ((c_i1 - 2 * c_in + c_in1) / (dx ** 2))) - (C2 * dt * (c_i1 - c_in1)) / (
                    2 * C1 * dx)
        return c_next

    def fwrd1fwrd2x_fwrdt(c_in, c_i1, c_i2, dx, dt, C1, C2):
        c_next = c_in + ((dt / C1) * ((c_i2 - 2 * c_i1 + c_in) / (dx ** 2))) - (
                    (C2 * dt * (c_i1 - c_in)) / (2 * C1 * dx))
        return c_next

    # Calculation of the feed time [s]
    feedTime = (feedVol / flowRate) * 3600
    # Calculation of the flow speed [mm/s]
    flowSpeed = (flowRate * 1000 / 3600) / (math.pi * ((diameter / 2) ** 2) * porosity)

    # Defining finite time of the experiment [s]
    time = 500
    # Defining number of spatial differences
    # Defining number of time differences
    # Preparation of space vector
    x = np.linspace(0, length, Nx)
    # Calculating space step [mm]
    dx = length / Nx

    # Preparation of time vector
    t = np.linspace(0, time, Nt)
    # Calculating space step [mm]
    dt = time / Nt

    # Preparation of solution matrix
    c = np.zeros((len(t), len(x)))

    # Implementing initial conditions
    C_0 = np.zeros(len(x))
    C_0[0] = (c[0, 1] + dx*flowSpeed*feedConc)/(1 + dx*flowSpeed)

    # c[0,0] = feedConc # For the left boudary
    c[0, :] = C_0  # First row (time = 0) are all elements C_0

    tStep = time / Nt  # time step [s]
    feedSteps = feedTime // tStep  # whole number of feed iterations
    feedTimeAprox = feedTime % tStep  # aproximation of division
    # rounding iteration step based on defined feed parameters
    if feedTimeAprox >= 0.5:
        feedSteps += 1

    feed = np.linspace(0, time, Nt)
    for i in range(0, Nt):
        if i <= feedSteps:
            feed[i] = feedConc
        else:
            feed[i] = 0

    # using Fit_Gauss to smooth out input feed
    """
    tmpFeedTime = np.linspace(0, time, Nt)
    d = {'time': tmpFeedTime, 'conc': feed}
    tmpDF = pd.DataFrame(data=d)
    tmpExperimentSet = ExperimentSet()
    tmpExperiment = Experiment()
    tmpComponent = ExperimentComponent()
    tmpComponent.concentrationTime = tmpDF
    tmpComponent.name = 'conc'
    tmpExperiment.experimentComponents.append(tmpComponent)
    tmpExperimentSet.experiments.append(tmpExperiment)
    print(tmpExperimentSet.experiments[0].experimentComponents[0].concentrationTime)
    tmpGauss = Fit_Gauss(tmpExperimentSet)
    feed = tmpGauss.experiments[0].experimentComponents[0].concentrationTime.iloc[:, 1].to_numpy()
    """
    # using scipy.interpolate.CubicSpline to smooth out input feed
    bp = 0
    tmpTimeStep = time/(Nt//1)
    tmpFeed = np.linspace(0, time, Nt//1)
    for i in range(0, Nt//1):
        if i*tmpTimeStep < feedTime:
            tmpFeed[i] = feedConc
        else:
            tmpFeed[i] = 0
            if bp == 0:
                bp = i

    tmpFeedTime = np.linspace(0, time, Nt//1)
    rmList = [range(bp-150, bp+149)]
    tmpFeed = np.delete(tmpFeed, rmList)
    tmpFeedTime = np.delete(tmpFeedTime, rmList)
    cs = CubicSpline(x=tmpFeedTime, y=tmpFeed, bc_type='natural')
    plt.plot(tmpFeedTime, tmpFeed)
    plt.show()
    feed = cs(np.linspace(0, time, Nt))
    plt.plot(np.linspace(0, time, Nt), feed)
    plt.show()


    # Implementing discretization
    for j in range(1, Nt-1):  # Advance in time
        # ----------------------------------------------------------------------
        # Feed pulse implementation
        cIn = feed[j]
        for i in range(1, Nx-2):
            divident1 = dt*disperCoef*(c[j-1, i+1] - 2*c[j-1, i] + c[j-1, i-1])
            divisor1 = (((1 - porosity)*saturationConst*langmuirConst)/(((-langmuirConst*c[j-1, i] + 1)**2)*porosity) + 1)*(dx**2)
            divident2 = dt*flowSpeed*(c[j-1, i+1] - c[j-1, i-1])
            divisor2 = (((1 - porosity)*saturationConst*langmuirConst)/(((-langmuirConst*c[j-1, i] + 1)**2)*porosity) + 1)*(dx*2)
            c[j, i] = (divident1/divisor1) - (divident2/divisor2) + c[j-1, i]
        c[j, 0] = (c[j, 1] + dx*flowSpeed*cIn)/(1 + dx*flowSpeed)
        c[j, Nx-1] = c[j, Nx-2]
    feedMass = feedVol * feedConc  # Calculating teoretical mass fed into system
    massCumulOut = 0  # Mass cumulation over time in outlet from the column
    massCumulIn = 0  # Mass cumulation over time in inlet to the column

    for i in range(0, Nt):  # Calculation of mass cumulation over time
        actConcOut = c[i, -1]
        massCumulOut += (dt * flowRate * actConcOut / 3600)

    # Calculation of differece between mass in feed and in the outlet
    massDifferenceOut = feedMass - massCumulOut
    massDifferenceIn = feedMass - massCumulIn
    # Display mass balance check
    print('\nFeed Mass:   ' + str(round(feedMass, 2)) + ' mg')
    print('Outlet Mass:   ' + str(round(massCumulOut, 2)) + ' mg')
    print('Difference:   ' + str(round(-(massDifferenceOut), 2)) + ' mg   '
          + str(round((massDifferenceOut * 100 / feedMass), 2)) + ' %\n')
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
    return c