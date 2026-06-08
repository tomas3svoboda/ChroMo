import numpy as np
import matplotlib.pyplot as plt
from ChroMo_SMB.SMB.LinColumn_working import LinColumn
from ChroMo_SMB.SMB.SMBStation import SMBStation
from ChroMo_SMB.SMB.Tube import Tube
from ChroMo_SMB.SMB.Lin_Solver import Lin_Solver

# === Simulation Parameters ===
flow_rates = [180, 0.1, 150, 0.1]  # Default SMB flow rates
porosity = 0.2  # Column porosity (dimensionless)
dt = 1.8  # Time step (s)
Nx = 60  # Number of discretization points in column
plot_interval = 10  # Plot update frequency (steps)
switch_interval = -1  # Disable switching for static separation setup

column_params = {
    "length": 310,  # Column length (mm)
    "diameter": 10,  # Column diameter (mm)
    "porosity": porosity,  # Column porosity (dimensionless)
    "dead_volume": 0.1  # Dead volume of tubing (mL)
}

def run_smb_simulation():
    """Runs a standard 4-zone SMB system with no switching, feed in Zone 3, and monitoring in the Raffinate stream."""
    smb = SMBStation()  # Create SMB system

    # Define components
    components = [
        {"name": "ComponentA", "feed_concentration": 7.27, "henry_constant": 1.5, "Di": 2},
    ]

    # Add Columns and Tubes to SMB System
    for zone in range(1, 5):
        smb.addColZone(zone,
                       LinColumn(column_params["length"],
                                 column_params["diameter"],
                                 column_params["porosity"]),
                       Tube(column_params["dead_volume"]))

    # Set Flow Rates per Zone
    for i, flow in enumerate(flow_rates, start=1):
        smb.setFlowRateZone(i, flow)

    # Set SMB system parameters
    smb.setSwitchInterval(switch_interval)  # No switching
    smb.setdt(dt)
    smb.setNx(Nx)

    # Add Components to SMB
    for comp in components:
        smb.createComponentAB(comp["name"],
                              comp["feed_concentration"],
                              comp["henry_constant"],
                              comp["Di"])

    # Initialize Columns
    smb.initCols()

    # Run Simulation Step-by-Step
    Nt = 6000  # Number of time steps
    c_matrix = np.zeros((Nt, Nx))  # Store concentration profiles

    for i in range(Nt):
        if i * dt < 1000:
            smb.updateComponentByName("ComponentA", feedConc=5)
        else:
            smb.updateComponentByName("ComponentA", feedConc=0.0)

            for zone in smb.zones:
                for tube in smb.zones[zone]:
                    if isinstance(tube, Tube):  # Only flush tubes, not columns
                        for comp in tube.components:
                            comp.c[:] = 0.0  # Reset tube concentration

        res = smb.step(steps=1)  # Step SMB system
        c_matrix[i, :] = res[3][1][0]  # Monitor Raffinate stream concentration (Zone 3, outlet to Zone 4)


    return c_matrix

def compare_solutions():
    """Runs both simulators and compares results."""

    # Run full simulation
    c_solver, t_solver = Lin_Solver(flow_rates[2], column_params["length"], column_params["diameter"],
                                     feedVol=10, feedConc=7.27,  # Set feed concentration directly
                                     porosity=porosity, henryConst=1.5, disperCoef=2,  # ComponentA parameters
                                     Nx=Nx, Nt=6000, time=10800, full=True)[:2]

    # Run SMB simulation
    c_stepwise = run_smb_simulation()

    # Compute error metrics
    abs_error = np.abs(c_solver - c_stepwise)
    rel_error = np.abs(c_solver - c_stepwise) / np.maximum(c_solver, 1e-9)  # Avoid division by zero
    rmse = np.sqrt(np.mean(abs_error**2))

    print(f"RMSE between simulations: {rmse:.6f}")

    # === Plotting Breakthrough Curves ===
    plt.figure(figsize=(10, 6))
    plt.plot(t_solver, c_solver[:, -1], label='Lin_Solver (Full)', linestyle='dashed')
    plt.plot(t_solver, c_stepwise[:, -1], label='SMB Simulation (Stepwise)', linestyle='solid')
    plt.xlabel("Time [s]")
    plt.ylabel("Outlet Concentration [g/mL]")
    plt.legend()
    plt.title("Breakthrough Curves Comparison")
    plt.show()

    # === Error Heatmap ===
    plt.figure(figsize=(10, 6))
    plt.imshow(abs_error.T, aspect='auto', cmap='hot', interpolation='none')
    plt.colorbar(label="Absolute Error")
    plt.xlabel("Time Step")
    plt.ylabel("Spatial Position")
    plt.title("Error Heatmap: Lin_Solver vs SMB Simulation")
    plt.show()

# Run the comparison
compare_solutions()
