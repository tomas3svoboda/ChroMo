import matplotlib.pyplot as plt
import pandas as pd
from ChroMo_SMB.SMB.SMBStation import SMBStation
from ChroMo_SMB.SMB.LinColumn_working import LinColumn
from ChroMo_SMB.SMB.Tube import Tube
from ChroMo_SMB.SMB.Lin_Solver import Lin_Solver

def simSMB(end_time,
           porosity=0.3756,
           name_compA="Man",
           henry_constantA=6.36,
           deltaA=13.51,
           name_compB="Gal",
           henry_constantB=3.62,
           deltaB=28.02,
           flow_rates=None,
           switch_interval=780,
           optimize_component='Man',
           plotting=False):
    """
    Runs the SMB chromatography simulation for a single specified component to save computational power.

    Args:
        end_time (float): The total simulation time in seconds.
        name_compA (str): Name of component A.
        henry_constantA (float): Henry constant of component A (dimensionless).
        deltaA (float): Delta value for component A (unitless parameter related to mass transfer kinetics).
        name_compB (str): Name of component B.
        henry_constantB (float): Henry constant of component B (dimensionless).
        deltaB (float): Delta value for component B (unitless parameter related to mass transfer kinetics).
        flow_rates (list, optional): Flow rates for the four zones in format [zone1 (mL/h), zone2 (mL/h), zone3 (mL/h), zone4 (mL/h)].
        switch_interval (int, optional): Column switching interval (s). Defaults to 780.
        optimize_component (str): 'A' to simulate only Component A, 'B' for Component B.

    Returns:
        tuple:
            - df_extract (pd.DataFrame): Extracted fraction time (s) and concentration (g/L).
            - df_raffinate (pd.DataFrame): Raffinate fraction time (s) and concentration (g/L).
            - collected_concentration_extract (float): Average collected concentration for extract (g/L).
            - collected_concentration_raffinate (float): Average collected concentration for raffinate (g/L).
    """
    smb = SMBStation()

    if flow_rates is None:
        flow_rates = [180, 93, 114, 45]

    dt = 3.6  # Time step (s)
    Nx = 100  # Number of discretization points in column
    plot_interval = 10  # Plot update frequency (steps)

    column_params = {
        "length": 310,  # Column length (mm)
        "diameter": 10,  # Column diameter (mm)
        "porosity": porosity,  # Column porosity (dimensionless)
        "dead_volume": 0.1  # Dead volume of tubing (mL)
    }

    components = []
    if optimize_component == name_compA:
        components.append({"name": name_compA,
                           "feed_concentration": 7.27,
                           "henry_constant": henry_constantA,
                           "delta": deltaA,
                           "Di": 0.0007})
    else:
        components.append({"name": name_compB,
                           "feed_concentration": 3.42,
                           "henry_constant": henry_constantB,
                           "delta": deltaB,
                           "Di": 0.0007})

    for zone in range(1, 5):
        smb.addColZone(zone,
                       LinColumn(column_params["length"],
                                 column_params["diameter"],
                                 column_params["porosity"]),
                       Tube(column_params["dead_volume"]))

    for i, flow in enumerate(flow_rates, start=1):
        smb.setFlowRateZone(i, flow)

    smb.setSwitchInterval(switch_interval)
    smb.setdt(dt)
    smb.setNx(Nx)

    for comp in components:
        smb.createComponentAB(comp["name"],
                              comp["feed_concentration"],
                              comp["henry_constant"],
                              comp["delta"],
                              comp["Di"])

    smb.initCols()

    times = []  # Time points (s)
    extract_values = []  # Extract concentration values (g/L)
    raffinate_values = []  # Raffinate concentration values (g/L)

    if plotting:
        plt.ion()
        fig, (ax_extract, ax_raffinate) = plt.subplots(2, 1, figsize=(8, 6))

        line_extract, = ax_extract.plot([], [], label=components[0]["name"])
        ax_extract.set_title("Extract Concentrations")
        ax_extract.set_xlabel("Time (s)")
        ax_extract.set_ylabel("Concentration (g/L)")
        ax_extract.legend()

        line_raffinate, = ax_raffinate.plot([], [], label=components[0]["name"])
        ax_raffinate.set_title("Raffinate Concentrations")
        ax_raffinate.set_xlabel("Time (s)")
        ax_raffinate.set_ylabel("Concentration (g/L)")
        ax_raffinate.legend()

    sim_time = 0
    step_counter = 0

    while sim_time < end_time:
        if sim_time > 240:
            smb.components[0].feedConc = 0.0
        res = smb.step(5)
        sim_time += 5 * dt

        times.append(sim_time)
        extract_values.append(res[1][-1][0][-1])
        raffinate_values.append(res[3][-1][0][-1])

        step_counter += 1

        if plotting:
            if step_counter % plot_interval == 0:
                line_extract.set_data(times, extract_values)
                ax_extract.relim()
                ax_extract.autoscale_view()

                line_raffinate.set_data(times, raffinate_values)
                ax_raffinate.relim()
                ax_raffinate.autoscale_view()

                plt.pause(0.001)

    df_extract = pd.DataFrame({"time": times, "concentration": extract_values})
    df_raffinate = pd.DataFrame({"time": times, "concentration": raffinate_values})

    if plotting:
        plt.ioff()
        plt.close(fig)

    # Delete all the instances of simulation classes
    for zone in smb.zones:
        for obj in smb.zones[zone]:
            del obj
        smb.zones[zone].clear()

    smb.components.clear()
    smb.cins.clear()
    smb.flowRates.clear()
    del smb

    return df_extract, df_raffinate

#=================================================================================
porosity = 0.3756
henry_constant = 6.36
Bo = 13.51
flowRate = 150
#=================================================================================
#=================================================================================
df_extract, df_raffinate= simSMB(10800,
                                 porosity=porosity,
                                 name_compA="Man",
                                 henry_constantA=henry_constant,
                                 deltaA=Bo,
                                 name_compB="Gal",
                                 henry_constantB=3.62,
                                 deltaB=28.02,
                                 flow_rates=[150, 0.01, flowRate, 0.01],
                                 switch_interval=20000,
                                 optimize_component='Man',
                                 plotting=True)

df_raffinate.to_excel('discont_testingSMB.xlsx', index=False)


#=================================================================================
#=================================================================================
import math
diameter = 10
length = 310
flowSpeed = (flowRate * 1000/3600) / ((math.pi * (diameter**2) / 4) * porosity)
D_app = length * flowSpeed / Bo

c_matrix, t_vector =Lin_Solver(flowRate=flowRate,       # Volume flowrate in [mL/h]
               length=length,          # Length of the packed section in the column [mm]
               diameter=diameter,         # Column diameter [mm]
               feedVol=10,          # Feed injection volume [mL]
               feedConc=7.27,     # Concentration of the balanced component in the feed [g/mL] or [mg/mm^3]
               porosity=porosity,        # Total porosity of the adsorbent packing [-]
               henryConst=henry_constant,      # Henry's constant of the linear isotherm [-]
               disperCoef=D_app,       # Axial dispersion coefficient [mm^2/s]
               Nx = 100,             # Number of spatial differences
               Nt = 3000,           # Number of time differences
               time=10800,          # Finite time of the experiment [s]
               debugPrint=False,
               debugGraph=False,
               full=False)

# Extract only the last column of the concentration matrix
c_output = c_matrix[:, -1]  # Shape (3000,)

# Convert time and output concentration to a DataFrame
df = pd.DataFrame({"Time": t_vector, "Output_Concentration": c_output})

# Save to Excel
df.to_excel('discont_testingChromMo.xlsx', index=False)

#=================================================================================
#=================================================================================

import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load your data
df1 = pd.read_excel('discont_testingSMB.xlsx')  # First dataset: Time1, Conc1
df2 = pd.read_excel('discont_testingChromMo.xlsx')  # Second dataset: Time2, Conc2

# Extract columns
time1, conc1 = df1["time"], df1["concentration"]
time2, conc2 = df2["Time"], df2["Output_Concentration"]

# Interpolate conc2 onto time1 scale (or vice versa)
interp_func = interp1d(time2, conc2, kind='linear', fill_value="extrapolate")
conc2_interpolated = interp_func(time1)  # Interpolating conc2 onto time1's grid

# Plot both curves
plt.figure(figsize=(10, 5))
plt.plot(time1, conc1, label="SMBStation", linestyle="-")
plt.plot(time1, conc2_interpolated, label="Lin_Column", linestyle="--")

# Formatting
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Comparison of Concentration Curves")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
#=================================================================================
#=================================================================================


