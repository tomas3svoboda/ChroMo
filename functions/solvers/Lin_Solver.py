import math
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import time as Time

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

#=========================================================================================================================
#=========================================================================================================================
def prepare_matrices(Nx, dt, dx_dense, dx_sparse, flowSpeed, disperCoef, henryConst, porosity, Nx_dense):
    """
    Prepares the coefficient matrices A and B for the Crank-Nicolson scheme, adapted for variable spatial steps.
    Parameters:
    Nx (int): Total number of spatial grid points.
    dt (float): Current time step size.
    dx_dense (float): Spatial step size in the densely discretized segment of the column.
    dx_sparse (float): Spatial step size in the sparsely discretized segment of the column.
    flowSpeed (float): Flow speed through the column [mm/s].
    disperCoef (float): Axial dispersion coefficient [mm^2/s].
    henryConst (float): Henry's constant of the linear isotherm [-].
    porosity (float): Total porosity of the adsorbent packing [-].
    Nx_dense (int): Number of spatial points in the dense segment.
    Returns:
    tuple: Tuple containing matrices A and B used in the Crank-Nicolson numerical scheme.
    """
    A = np.zeros((Nx, Nx))  # Matrix A for the next time step concentration
    B = np.zeros((Nx, Nx))  # Matrix B for the current time step concentration

    # Constants used in the discretization formula
    a = disperCoef / ((((1 - porosity) * henryConst) / porosity) + 1)
    b = flowSpeed / ((((1 - porosity) * henryConst) / porosity) + 1)
    # Define the dense segment end index
    dense_end = Nx_dense - 1
    # Set up boundary conditions
    A[0, 0] = flowSpeed / disperCoef + 1 / (2 * dx_dense)
    A[0, 1] = -1 / (2 * dx_dense)
    A[Nx - 1, Nx - 2] = -1 / (2 * dx_sparse)
    A[Nx - 1, Nx - 1] = 1 / (2 * dx_sparse)
    B[0, 0] = -1 / (2 * dx_dense)
    B[0, 1] = 1 / (2 * dx_dense)
    B[Nx - 1, Nx - 2] = 1 / (2 * dx_sparse)
    B[Nx - 1, Nx - 1] = -1 / (2 * dx_sparse)
    # Fill the interior points
    for j in range(1, Nx - 1):
        dx_current = dx_dense if j <= dense_end else dx_sparse
        A[j, j - 1] = -((dt * a) / (2 * (dx_current ** 2))) - ((dt * b) / (4 * dx_current))
        A[j, j] = 1 + ((dt * a) / (dx_current ** 2))
        A[j, j + 1] = -((dt * a) / (2 * (dx_current ** 2))) + ((dt * b) / (4 * dx_current))
        B[j, j - 1] = ((dt * a) / (2 * (dx_current ** 2))) + ((dt * b) / (4 * dx_current))
        B[j, j] = 1 - ((dt * a) / (dx_current ** 2))
        B[j, j + 1] = ((dt * a) / (2 * (dx_current ** 2))) - ((dt * b) / (4 * dx_current))
    return A, B

#=========================================================================================================================
#=========================================================================================================================
def discretize_and_solve(Nx, Nt, dx_dense, dx_sparse, dt_dense, dt_sparse, disperCoef, flowSpeed, henryConst, porosity, feed, c, debugPrint, Nx_dense, denseSparseRatio):
    """
    Discretizes the problem using the Crank-Nicolson method and solves the concentration profile over time.
    Parameters:
    Nx (int): Total number of spatial grid points.
    Nt (int): Total number of time steps.
    dx_dense (float): Spatial step size in the densely discretized segment of the column.
    dx_sparse (float): Spatial step size in the sparsely discretized segment of the column.
    dt_dense (float): Time step size used during dense temporal discretization phases.
    dt_sparse (float): Time step size used during sparse temporal discretization phases.
    disperCoef (float): Axial dispersion coefficient [mm^2/s].
    flowSpeed (float): Flow speed through the column [mm/s].
    henryConst (float): Henry's constant of the linear isotherm [-].
    porosity (float): Total porosity of the adsorbent packing [-].
    feed (array): Array containing the feed concentration over time.
    c (2D array): Array to store the concentration profile over time. Modified in-place.
    debugPrint (bool): If True, print debug statements.
    Nx_dense (int): Number of spatial points in the dense segment.
    Returns:
    2D array: Updated concentration profile array after solving the differential equation.
    """
    # Prepare initial matrices for Crank-Nicolson using the dense time step
    A, B = prepare_matrices(Nx, dt_dense, dx_dense, dx_sparse, flowSpeed, disperCoef, henryConst, porosity, Nx_dense)
    for i in range(1, Nt):
        if debugPrint:
            if i == 1:
                print('\nSolution algorithm has been started:')
            if i % (Nt // 20) == 0:
                print(str(i) + ' steps has been finished ... ' + str(Nt - i) + ' steps remain.')
        # Determine the current time step size
        dt_current = dt_dense if i <= (Nt * denseSparseRatio) else dt_sparse
        # Update matrices if the time step changes
        if i == 1 or dt_current != (dt_dense if i-1 <= (Nt * denseSparseRatio) else dt_sparse):
            A, B = prepare_matrices(Nx, dt_current, dx_dense, dx_sparse, flowSpeed, disperCoef, henryConst, porosity, Nx_dense)
        # Solve the system for the current step
        b = np.dot(B, c[i - 1, :])
        b[0] += flowSpeed / disperCoef * feed[i]
        c[i, :] = np.linalg.solve(A, b)
    return c

#=========================================================================================================================
#=========================================================================================================================
def data_struct_prep(length, Nx, Nt, feedTime, feedConc, time):
    """
    Prepare spatial and temporal grids, feed vector, and initial conditions for the chromatography model.
    Parameters:
    length (float): Total length of the column [mm]
    Nx (int): Number of spatial divisions
    Nt (int): Number of time steps
    feedTime (float): Duration of feed injection [s]
    feedConc (float): Concentration of the injected feed [g/mL]
    time (float): Total simulation time [s]
    Returns:
    tuple: (x, dx_dense, dx_sparse, t, dt_dense, dt_sparse, feed, c0)
    """
    #___________________________________________________
    # Spatial discretization configuration
    p_dense = 0.6  # percentage of the points in the dense segment
    dense_space_ratio = 0.2  # fraction of the column length for the dense segment

    # Calculate number of points and lengths for dense and sparse segments
    Nx_dense = int(round(Nx * p_dense))
    Nx_sparse = Nx - Nx_dense
    dense_length = length * dense_space_ratio
    sparse_length = length - dense_length

    # Calculate dx for dense and sparse segments
    dx_dense = dense_length / (Nx_dense - 1) if Nx_dense > 1 else dense_length
    dx_sparse = sparse_length / (Nx_sparse - 1) if Nx_sparse > 1 else sparse_length

    # Create spatial vector x
    x_dense = np.linspace(0, dense_length, Nx_dense, endpoint=False)
    x_sparse = np.linspace(dense_length, length, Nx_sparse, endpoint=True)
    x = np.concatenate((x_dense, x_sparse))

    #___________________________________________________
    # Time discretization
    denseSparseRatio = 0.8
    dense_steps = int(Nt * denseSparseRatio)
    sparse_steps = Nt - dense_steps
    dense_time = feedTime + feedTime * ((time / feedTime) / 20)
    t_dense = np.linspace(0, dense_time, dense_steps, endpoint=False)
    t_sparse = np.linspace(t_dense[-1], time, sparse_steps, endpoint=True)
    t = np.concatenate((t_dense, t_sparse))

    dt_dense = t_dense[1] - t_dense[0]
    dt_sparse = t_sparse[1] - t_sparse[0]

    # Constructing pulse injection feed vector
    feedSteps = int(feedTime // dt_dense)  # Whole number of feed iterations
    # Feed vector with ramp-up and ramp-down
    feed = np.zeros(Nt)
    # Set feed concentration values (get rid of extremely fast changes of feed concentration)
    for i in range(dense_steps):
        if i == 0:
            feed[i] = feedConc/10
        elif i == 1:
            feed[i] = feedConc/5
        elif i == 2:
            feed[i] = feedConc/2
        elif i == 3:
            feed[i] = feedConc/1.5
        elif i == 4:
            feed[i] = feedConc/1.1
        elif i <= feedSteps:
            feed[i] = feedConc
        elif i == feedSteps+1:
            feed[i] = feedConc/1.1
        elif i == feedSteps+2:
            feed[i] = feedConc/1.5
        elif i == feedSteps+3:
            feed[i] = feedConc/2
        elif i == feedSteps+4:
            feed[i] = feedConc/5
        elif i == feedSteps+5:
            feed[i] = feedConc/10
        else:
            feed[i] = 0

    # Initial conditions
    c0 = np.zeros(len(x))

    return x, dx_dense, dx_sparse, t, dt_dense, dt_sparse, feed, c0, Nx_dense, denseSparseRatio

#=========================================================================================================================
#=========================================================================================================================
def Lin_Solver(flowRate = 150,       # Volume flowrate in [mL/h]
               length=235,          # Length of the packed section in the column [mm]
               diameter=16,         # Column diameter [mm]
               feedVol=1,          # Feed injection volume [mL]
               feedConc=150e-3,     # Concentration of the balanced component in the feed [g/mL] or [mg/mm^3]
               porosity=0.2,        # Total porosity of the adsorbent packing [-]
               henryConst=0.5,      # Henry's constant of the linear isotherm [-]
               disperCoef=3,       # Axial dispersion coefficient [mm^2/s]
               Nx = 30,             # Number of spatial differences
               Nt = 3000,           # Number of time differences
               time=10800,          # Finite time of the experiment [s]
               debugPrint=False,
               debugGraph=False,
               full=False):

    # Calculation time measurement
    start_time = Time.time()  # Start time when function is called

    # Calculation of the feed time [s]
    feedTime = (feedVol / flowRate) * 3600

    # Calculation of the flow speed [mm/s]
    flowSpeed = (flowRate * 1000/3600) / ((math.pi * (diameter**2) / 4) * porosity) # 1000 for mL > mm^3

    if debugPrint:
        print('Flow speed:   ' + str(round(flowSpeed, 2)) + ' [mm/s]')
        print('Langmuir Constant:   ' + str(round(henryConst, 2)))
        print('Dispersion Coefficient:   ' + str(round(disperCoef, 2)) + ' mm2/s')

    # ______________________________________
    # DATA STRUCTURES PREPARATION

    x, dx_dense, dx_sparse, t, dt_dense, dt_sparse, feed, c0, Nx_dense,denseSparseRatio = data_struct_prep(length, Nx, Nt, feedTime, feedConc, time)
    # Preparation of the solution matrix
    c = np.zeros((Nt, Nx))

    # ______________________________________
    # DISCRETIZATION AND SOLUTION
    # Prepare matrices A and B for Crank-Nicolson method

    c = discretize_and_solve(Nx, Nt, dx_dense, dx_sparse, dt_dense, dt_sparse, disperCoef, flowSpeed, henryConst, porosity, feed, c, debugPrint, Nx_dense, denseSparseRatio)

    # Measure the elapsed time at the end of your function
    end_time = Time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_milliseconds = elapsed_time_seconds * 1000  # Convert to milliseconds

    if debugPrint:
      print(f"Elapsed time: {elapsed_time_milliseconds:.2f} ms")

    # ______________________________________
    # MASS BALANCE CHECK

    feedMass = feedVol * feedConc  # Calculating teoretical mass fed into system
    massCumulOut = 0  # Mass cumulation over time in outlet from the column
    massCumulIn = 0  # Mass cumulation over time in inlet to the column

    for i in range(0, Nt):  # Calculation of mass cumulation over time
        actConcOut = c[i, -1]
        massCumulOut += abs((t[i]-t[i-1]) * flowRate * actConcOut / 3600)

    # Calculation of differece between mass in feed and in the outlet
    massDifferenceOut = feedMass - massCumulOut
    massDifferenceIn = feedMass - massCumulIn
    massDifferencePerc = massDifferenceOut * 100 / feedMass
    # Display mass balance check
    if debugPrint:
        print('\nFeed volume:   ' + str(round(feedVol, 2)) + ' mg')
        print('\nFeed conc:   ' + str(round(feedConc, 2)) + ' mg')
        print('\nFeed Mass:   ' + str(round(feedMass, 2)) + ' mg')
        print('Outlet Mass:   ' + str(round(massCumulOut, 2)) + ' mg')
        print('Difference:   ' + str(round(-(massDifferenceOut), 2)) + ' mg   '
              + str(round((massDifferenceOut * 100 / feedMass), 2)) + ' %\n')

    if debugPrint:
        results = pd.DataFrame(c)
        print('Complete results mesh of size')
        print('Number of Elements: ' + str(results.size))
        print('Shape of the solution matrix: ' + str(results.shape))
        # Find the maximum value across the entire DataFrame
        overall_max = results.max().max()
        # Print the maximum value
        print('Maximum value in the DataFrame:', overall_max)

    # ______________________________________
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

        # Plotting the feed and various concentration profiles
        fig2, ax = plt.subplots()
        ax.plot(t, feed, label='Feed')
        # Selecting points along the column to plot
        indices_to_plot = np.round(np.linspace(0, len(x) - 1, 10)).astype(int)
        for idx in indices_to_plot:
            ax.plot(t, c[:, idx], label=f'{x[idx]:.2f} mm')  # Using exact x positions

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Concentration [mg/mL]')
        ax.set_title('Concentration-Time Curves at Various Column Positions')
        ax.legend()
        plt.savefig('Concentration_time_plot.png')
        plt.show()

    if full:
        return [c, t, feed]
    return [c, t]
