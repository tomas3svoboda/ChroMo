import numpy as np
import math
# Numerical solver of the EDM with Langmuir Isotherm - nonlinear parabolic PDE

def Nonlin_Solver(flowRate = 800, length = 235, diameter = 16, feedVol = 5, feedConc = 2, porosity = 0.5, henryConst = 2.5, disperCoef = 0.95):
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

    C1 = (1 / disperCoef) + ((1 + porosity) * henryConst / (porosity * disperCoef))
    C2 = flowSpeed / disperCoef

    # Defining finite time of the experiment [s]
    time = 500
    # Defining number of spatial differences
    Nx = 180
    # Defining number of time differences
    Nt = 1000
    # Preparation of space vector
    x = np.linspace(0, length, Nx)
    # Calculating space step [mm]
    dx = length / Nx

    # Preparation of time vector
    t = np.linspace(0, time, Nt)
    # Calculating space step [mm]
    dt = time / Nx

    # Preparation of solution matrix
    c = np.zeros((len(t), len(x)))

    # Implementing initial conditions
    C_0 = np.zeros(len(x))
    C_0[0] = feedConc

    # c[0,0] = feedConc # For the left boudary
    c[0, :] = C_0  # First row (time = 0) are all elements C_0

    tStep = time / Nt  # time step [s]
    feedSteps = feedTime // tStep  # whole number of feed iterations
    feedTimeAprox = feedTime % tStep  # aproximation of division
    # rounding iteration step based on defined feed parameters
    if feedTimeAprox >= 0.5:
        feedSteps += 1

    # Implementing discretization
    for i in range(0, Nt - 1):  # Advance in time
        # ----------------------------------------------------------------------
        # Feed pulse implementation
        # !!! PŘEPSAT NA VECTOR
        if i <= feedSteps:
            cIn = feedConc
        else:
            cIn = 0
        # Calculating left boudary [x=0] in time ti
        c[i + 1, 0] = Danckwert_lb_c2ap(c[i, 1], c[i, 0], dx, dt, C1, C2, cIn)
        # ----------------------------------------------------------------------
        for n in range(1, Nx - 1):  # Calculating values of space differences in time ti
            # Calculating value for space xn in time ti
            c[i + 1, n] = c1c2x_fwrdt(c[i, n], c[i, n + 1], c[i, n - 1], dx, dt, C1, C2)
        # ----------------------------------------------------------------------
        # Calculating right boundary [x=L] in time ti
        c[i + 1, Nx - 2] = Danckwert_rb_c2ap(c[i, Nx - 2], c[i, Nx - 1], dx, dt, C1)
    return c