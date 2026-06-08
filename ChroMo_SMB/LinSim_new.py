from EKF.SMB.SMBStation import SMBStation
from EKF.SMB.LinColumn import LinColumn
from EKF.SMB.Tube import Tube
import matplotlib.pyplot as plt
import numpy as np

# SMB setup
smb = SMBStation()

# Flow rates [mL/h]
flow_rate_zone = [172, 140, 160, 2]

# Switch interval [s]
switch_interval = 1894

# Discretization parameters
dt = 0.05
Nx = 100

# Column parameters
column_length = 320
column_diameter = 10
porosity = 0.376
dead_volume = 0.5

# Components definitions
components = [
    {"name": "Man", "feed_concentration": 9, "henry_constant": 4.55, "delta": 54, "Di": 0.0007},
    {"name": "Gal", "feed_concentration": 6, "henry_constant": 2.77, "delta": 84, "Di": 0.0007}
]

# Save and plot intervals
num_of_steps_to_save = 10
num_of_steps_to_plot = 20

# Initialize SMB columns and tubes
for zone in range(4):
    smb.addColZone(zone + 1, LinColumn(column_length, column_diameter, porosity), Tube(dead_volume))

# Set flow rates and parameters
for i, fr in enumerate(flow_rate_zone, 1):
    smb.setFlowRateZone(i, fr)
smb.setSwitchInterval(switch_interval)
smb.setdt(dt)
smb.setNx(Nx)

# Add components
for comp in components:
    smb.createComponentAB(comp["name"], comp["feed_concentration"], comp["henry_constant"], comp["delta"], comp["Di"])

smb.initCols()

# Results dictionary
result_dict = {
    "extract": {comp["name"]: [] for comp in components},
    "raffinate": {comp["name"]: [] for comp in components},
    "concentration_profile": {comp["name"]: [] for comp in components}
}

# Plot initialization
fig, ax = plt.subplots(figsize=(10, 5))
x_axis = np.linspace(0, 4 * column_length, 4 * Nx)
sim_time = 0
plot_counter = 0

# Simulation loop
while True:
    plot_counter += 1
    res = smb.step(num_of_steps_to_save)
    sim_time += num_of_steps_to_save * dt

    # Save Extract and Raffinate concentrations
    for idx, comp in enumerate(components):
        # correct: last column (= -1) in zone 1 & 3; then pick the component and its outlet value
        result_dict["extract"][comp["name"]].append(res[1][-1][idx][-1])
        result_dict["raffinate"][comp["name"]].append(res[3][-1][idx][-1])


    # Save spatial concentration profiles
    for idx, comp in enumerate(components):
        profiles = []
        for zone in range(4):
            arr = np.asarray(res[zone + 1][1])  # [1] for column (not tube)
            if arr.ndim == 2:
                profiles.append(arr[idx, :])
            elif arr.ndim == 1:
                profiles.append(arr)
            else:
                raise ValueError(f"Unexpected shape for arr: {arr.shape}")
        full_profile = np.hstack(profiles)
        result_dict["concentration_profile"][comp["name"]].append(full_profile)

    # Plotting at intervals
    if plot_counter >= num_of_steps_to_plot:
        plot_counter = 0

        ax.clear()
        for idx, comp in enumerate(components):
            latest_profile = result_dict["concentration_profile"][comp["name"]][-1]
            print(f"len(x_axis): {len(x_axis)}, len(latest_profile): {len(latest_profile)}")
            ax.plot(x_axis, latest_profile, label=comp["name"])

        # Mark column boundaries
        for boundary in range(1, 4):
            ax.axvline(boundary * column_length, color='black', linestyle='--', linewidth=0.8)

        ax.set_xlabel("Continuous Column Length [mm]")
        ax.set_ylabel("Concentration")
        ax.set_title(f"SMB Concentration Profile at time {sim_time:.2f} s")
        ax.legend()
        ax.grid(False)

        plt.tight_layout()
        plt.pause(0.005)

