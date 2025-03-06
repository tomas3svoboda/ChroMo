import numpy as np
import pandas as pd
import math
from scipy import optimize
import matplotlib.pyplot as plt
from numba import jit

# ***Numerical solution of Equilibrium Dispersive Model of Chromatography***
#                       ***Langmuir isotherm***
#                 ***Cranck-Nicolson Implicid Method***
# Developed by Ing. Tomas Svoboda
# University of Chemistry and Technology in Prague, Czech Republic
# Department of Carbohydrates and Cereals
# tomas3.svoboda@vscht.cz
'''
This script implements Cranck-Nicolson implicit method for numerical solution of
non-linear second order convection-diffusion PDE used to describe concentration
wave propagation trough the chromatographic column. Numerical scheme utilizes
averaged centred difference scheme in spatial direction and forward difference
scheme in time direction. Danckwert's boudaries are implemented with usage of
fictious point for left boundary.
'''
def Nonlin_Solver(
                  flowRate = 150,       # Volume flowrate in [mL/h]
                  length = 320,         # Lenght of the packed section in the column [mm]
                  diameter = 10,        # Column diameter [mm]
                  feedVol = 4,         # Feed injection volume [mL]
                  feedConc = 6,    # Concentration of the balanced component in the feed [g/mL] or [mg/mm^3]
                  porosity = 0.4,       # Total porosity of the sorbent packing [-]
                  langmuirConst = 2,  # Langmuir's constant of the linear isotherm [-]
                  disperCoef = 2,       # Axial dispersion coefficient [mm^2/s]
                  saturCoef = 20,       # Saturation Coefficient
                  Nx = 30,              # Number of spatial differences - Nx
                  Nt = 3000,            # Number of time differences - Nt
                  time = 10800,          # Finite time of the experiment [s]
                  debugPrint=False,
                  debugGraph=False,
                  full=False
                  ):

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
    # Configuration
    Nx_total = Nx  # Total number of spatial points
    p_dense = 0.5  # percentage of the points are in the dense segment
    dense_space_ratio = 0.4  # fraction of the column lenght for the dense segment

    # The desired number of points in the dense segment
    Nx_dense_desired = int(round(Nx_total * p_dense))

    # Adjustments to ensure the correct distribution of points
    # Calculate the total number of points per unit length for dense and sparse areas
    total_points_per_unit_length_dense = Nx_dense_desired / dense_space_ratio

    # Determine the actual number of points for dense and sparse segments to match the length ratios
    Nx_dense_actual = int(round(total_points_per_unit_length_dense * dense_space_ratio))
    Nx_sparse_actual = Nx_total - Nx_dense_actual

    # Length of dense and sparse segments
    dense_length = length * dense_space_ratio  # Length of dense segment

    # Create space vectors for dense and sparse segments
    x_dense = np.linspace(0, dense_length, Nx_dense_actual, endpoint=False)  # Dense grid in space
    x_sparse = np.linspace(dense_length, length, Nx_sparse_actual)  # Sparse grid in space

    # Combine the dense and sparse space vectors
    x = np.concatenate((x_dense, x_sparse))  # Combined space vector
    dx = np.diff(x)
    dx = np.append(dx, dx[-1])

    # time grid
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

    if debugPrint:
        dense_dense = flowSpeed * dt_dense / dx[0]
        dense_sparse = flowSpeed * dt_dense / dx[-1]
        sparse_dense = flowSpeed * dt_sparse / dx[0]
        sparse_sparse = flowSpeed * dt_sparse / dx[-1]
        print("\nCourant-Friedrichs-Lewy Numbers Matrix:")
        print(f"{'':>15} | {'Dense Grid':>12} | {'Sparse Grid':>12}")
        print("-" * 40)
        print(f"{'Dense Time Step':>10} | {dense_dense:12.6f} | {dense_sparse:12.6f}")
        print(f"{'Sparse Time Step':>10} | {sparse_dense:12.6f} | {sparse_sparse:12.6f}")

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

    # Preparation of the solution matrix
    c = np.zeros((Nt, Nx))

    # DISCRETIZATION
    @jit(nopython=True)
    def discretization(c1,
                 c0,
                 feedCur,
                 porosity,
                 langmuirConst,
                 saturCoef,
                 disperCoef,
                 flowSpeed,
                 dt, # single value
                 dx # vector of dx values
                 ):
        f = np.zeros(len(c0))  # Preparation of solution vector - will be optimized to 0
        for i in range(0, len(c0)):  # Main loop through all the vector's elements
            # Adjusted derivative calculations to handle variable dx
            dx_i = dx[i]
            dx_squared_i = dx_i ** 2
            dx_twice_i = dx_i * 2
            if i == 0:  # Left boundary
                f[0] = ((((c0[1] - c0[0]) / dx_i) + ((c1[1] - c1[0]) / dx_i)) / 2) - \
                       (flowSpeed/disperCoef * (c1[0] - feedCur))
            elif i > 0 and i < Nx - 1:
                epsilon = 1e-10  # A small number to prevent division by zero
                denominator0 = ((1 - porosity) * saturCoef * langmuirConst) / \
                               ((((langmuirConst * c0[i] + 1) ** 2) * porosity) + epsilon)
                denominator1 = ((1 - porosity) * saturCoef * langmuirConst) / \
                               ((((langmuirConst * c1[i] + 1) ** 2) * porosity) + epsilon)
                secondDer0 = (c0[i - 1] - 2 * c0[i] + c0[i + 1]) / (dx_squared_i)
                secondDer1 = (c1[i - 1] - 2 * c1[i] + c1[i + 1]) / (dx_squared_i)
                firstDer0 = (c0[i + 1] - c0[i - 1]) / (dx_twice_i)
                firstDer1 = (c1[i + 1] - c1[i - 1]) / (dx_twice_i)
                timeDer = (c1[i] - c0[i]) / dt
                disperElem = ((disperCoef / denominator0 * secondDer0) + (disperCoef / denominator1 * secondDer1)) / 2
                convElem = ((flowSpeed / denominator0 * firstDer0) + (flowSpeed / denominator1 * firstDer1)) / 2
                f[i] = disperElem - convElem - timeDer
            elif i == Nx - 1:  # Right boundary
                f[Nx - 1] = (((c0[Nx - 1] - c0[Nx - 2]) / dx_i) + ((c1[Nx - 1] - c1[Nx - 2]) / dx_i)) / 2
        return f

    # SOLUTION ALGORITHM
    residuals = np.zeros(Nt)  # Initialize a vector to store the residuals
    for i in range(1, Nt):
        if debugPrint:
            if i == 1:
                print('\nSolution algorithm has started:')
            if i % (Nt // 20) == 0:
                print(str(i) + ' steps has been finished ... ' + str(Nt - i) + ' steps remain.')
        if i <= dense_steps:
            dt = dt_dense
        else:
            dt = dt_sparse
        options = {'col_deriv': True}
        sol = optimize.root(fun=discretization,
                            x0=c[i - 1, :],
                            method='hybr',
                            args=(c[i - 1, :],
                                  feed[i],
                                  porosity,
                                  langmuirConst,
                                  saturCoef,
                                  disperCoef,
                                  flowSpeed,
                                  dt,
                                  dx),
                            options=options
                            )
        c[i, :] = sol.x # Save solution concentrations matrix
        residuals[i] = np.linalg.norm(sol.fun)  # Save the L2-norm of the residuals at each time step
    # ________________________________________________________________________
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
        # Choose indices for plotting based on Nx
        x_plot_indices = np.round(np.linspace(0, len(x)-1, 10)).astype(int)  # Use x length, not Nx, for precise spacing
        fig2 = plt.figure(figsize=(10, 6))
        # Plot feed
        plt.plot(t, feed, label='Feed')
        # Plot concentration profiles at selected spatial positions
        for idx in x_plot_indices:
          label = f'{x[idx]:.2f} mm'  # Use actual x positions for labels
          plt.plot(t, c[:, idx], label=label)
        # Ensure to include the outlet concentration profile
        plt.plot(t, c[:, -1], label=f'{x[-1]:.2f} mm (Outlet)')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration')
        plt.title('Concentration vs. Time at Various Column Positions')
        plt.savefig(f'Concentration_time_plot_{Nt}x{len(x)}_{int(round(Nt / len(x), 0))}.png')
        plt.show()

        # Plotting iteration residuals
        plt.plot(t, residuals)  # Plot the residuals over time
        plt.xlabel('Time [s]')
        plt.ylabel('Residuals')
        plt.show()
    '''------------------------------------------------------------------------------'''
    # Create a DataFrame with Time, Concentration (last column), and Feed
    sheet = pd.DataFrame({
        'Time': [(x / 60) for x in t],
        'Concentration_Last': c[:, -1],
        'Feed': feed
        })

    # Debug: Print all function arguments
    print("\n=== Nonlin_Solver: Input Parameters ===")
    for key, value in locals().items():
        print(f"{key}: {value}")
    print("=======================================\n")

    '''# Save to Excel
    output_file = "genFit_feedConc" + str(round(feedConc)) + "_ disper" + str(round(disperCoef)) + "_flow" + str(round(flowRate)) + ".xlsx"
    sheet.to_excel(output_file, index=False)'''
    '''-------------------------------------------------------------------------------'''
    if full:
        return [c, t, feed, residuals, massDifferencePerc]
    return [c, t]
