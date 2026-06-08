from SMB.SMBStation import SMBStation
from SMB.LinColumn_working import LinColumn
from SMB.NonlinColumn import NonLinColumn
from SMB.Tube import Tube
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
import pandas as pd

# Fixed constants for component parameters
FLOW_RATE_ZONE_1 = 341
FLOW_RATE_ZONE_2 = 109
FLOW_RATE_ZONE_3 = 120
FLOW_RATE_ZONE_4 = 48
SWITCHING_TIME = 1894
FEED_CONC_MAN = 9
FEED_CONC_GAL = 6
LANGMUIR_CONST = -1
SATUR_COEF = -1
DIFFUSION_COEF = 0.0007
NUM_OF_STEPS_TO_SAVE = 10
NUM_OF_STEPS_TO_PLOT = 20
CUTOFF_THRESHOLD = 500  # Maximum number of switches to store data

# Load experimental data
file_path = "SMB_ParamEstimation/SMB_onePeriond_experiment5.xlsx"
df_experiment = pd.read_excel(file_path, sheet_name="Sheet1")
df_experiment["Time_Ext (s)"] = df_experiment["Time_Ext (min)"] * 60
df_experiment["Time_Raff (s)"] = df_experiment["Time_Raff (min)"] * 60
exp_time_extract = df_experiment["Time_Ext (s)"].values
exp_conc_man_extract = df_experiment["Man_Ext (g/L)"].values
exp_conc_gal_extract = df_experiment["Gal_Ext (g/L)"].values
exp_time_raffinate = df_experiment["Time_Raff (s)"].values
exp_conc_man_raffinate = df_experiment["Man_Raff (g/L)"].values
exp_conc_gal_raffinate = df_experiment["Gal_Raff (g/L)"].values

# Function to initialize SMB system
def initialize_smb(henry_constants=[4.55, 2.77], delta_values=[54, 84]):
    smb = SMBStation()
    # flow rates in zones
    flow_rate_zone1 = 341 # [mL/h]
    flow_rate_zone2 = 109 # [mL/h]
    flow_rate_zone3 = 120 # [mL/h]
    flow_rate_zone4 = 48 # [mL/h]

    # switch interval
    switch_interval = 1894 # s

    # discretization params
    dt = 0.05 # s
    Nx = 100 # number of spatial points

    # column params
    column_length = 320 # [mm]
    column_diameter = 10 # [mm]
    porosity = 0.376
    dead_volume = 0.5 # [mL]

    # define component 1
    name1 = "Man"
    feed_concentration1 = 9 # [mL/g]
    henry_constant1 = henry_constants[0]
    delta1 = delta_values[0]
    Di1 = 0.0007 # [mm2/s]

    # define component 2
    name2 = "Gal"
    feed_concentration2 = 6 # [mL/g]
    henry_constant2 = henry_constants[1]
    delta2 = delta_values[1]
    Di2 = 0.0007 # [mm2/s]

    #for zone in range(1, 5):
    #    smb.addColZone(zone, LinColumn(column_length, column_diameter, porosity), Tube(dead_volume))
    smb.addColZone(1, LinColumn(column_length, column_diameter, porosity), Tube(dead_volume))
    smb.addColZone(2, LinColumn(column_length, column_diameter, porosity), Tube(dead_volume))
    smb.addColZone(3, LinColumn(column_length, column_diameter, porosity), Tube(dead_volume))
    smb.addColZone(4, LinColumn(column_length, column_diameter, porosity), Tube(dead_volume))

    smb.setFlowRateZone(1, flow_rate_zone1)
    smb.setFlowRateZone(2, flow_rate_zone2)
    smb.setFlowRateZone(3, flow_rate_zone3)
    smb.setFlowRateZone(4, flow_rate_zone4)
    smb.setSwitchInterval(switch_interval)
    smb.setdt(dt)
    smb.setNx(Nx)

    smb.createComponentAB(name1, feed_concentration1, henry_constant1, delta1, Di1)
    smb.createComponentAB(name2, feed_concentration2, henry_constant2, delta2, Di2)

    smb.initCols()
    return smb

# Function to run SMB simulation
def run_smb_simulation(henry_constants, delta_values):
    smb = initialize_smb(henry_constants, delta_values)  # Pass optimized parameters

    result_dict = {"extract": {0: [], 1: []}, "raffinate": {0: [], 1: []}}
    sim_time = 0
    num_switches = 11  # 10 for steady-state, 1 for objective calculation
    time_points = np.arange(0, num_switches * SWITCHING_TIME, 0.05)
    plt.figure(num='Optimization Progress')

    for _ in range(int(SWITCHING_TIME / 0.05)):
        res = smb.step(NUM_OF_STEPS_TO_SAVE)
        sim_time += NUM_OF_STEPS_TO_SAVE * 0.05

        for idx, x in enumerate(res[1][-1]):
            result_dict["extract"][idx].append(x[-1])
        for idx, x in enumerate(res[3][-1]):
            result_dict["raffinate"][idx].append(x[-1])

        # Plot results in real-time
    plt.clf()
    plt.plot(time_points[:len(result_dict["extract"][0])], result_dict["extract"][0], label='Sim Man Extract')
    plt.plot(time_points[:len(result_dict["extract"][1])], result_dict["extract"][1], label='Sim Gal Extract')
    plt.plot(time_points[:len(result_dict["raffinate"][0])], result_dict["raffinate"][0], label='Sim Man Raffinate')
    plt.plot(time_points[:len(result_dict["raffinate"][1])], result_dict["raffinate"][1], label='Sim Gal Raffinate')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (g/L)")
    plt.title("Real-time Optimization Progress")
    plt.pause(0.1)

    return result_dict["extract"], result_dict["raffinate"]

# Define optimization objective function
import time  # Import for tracking optimization progress

def objective(params):
    henry_constants = params[:2]
    delta_values = params[2:]
    global smb

    if 'smb' not in globals():
        smb = initialize_smb()

    sim_extract, sim_raffinate = run_smb_simulation(henry_constants, delta_values)

    # Ensure data is not empty before interpolation
    if not sim_extract[0] or not sim_extract[1] or not sim_raffinate[0] or not sim_raffinate[1]:
        return 1e6  # Return high error if no data is available

    sim_man_extract_interp = np.interp(exp_time_extract, np.linspace(0, SWITCHING_TIME, len(sim_extract[0])), sim_extract[0])
    sim_gal_extract_interp = np.interp(exp_time_extract, np.linspace(0, SWITCHING_TIME, len(sim_extract[1])), sim_extract[1])
    sim_man_raffinate_interp = np.interp(exp_time_raffinate, np.linspace(0, SWITCHING_TIME, len(sim_raffinate[0])), sim_raffinate[0])
    sim_gal_raffinate_interp = np.interp(exp_time_raffinate, np.linspace(0, SWITCHING_TIME, len(sim_raffinate[1])), sim_raffinate[1])

    mse = np.mean((exp_conc_man_extract - sim_man_extract_interp) ** 2) + \
          np.mean((exp_conc_gal_extract - sim_gal_extract_interp) ** 2) + \
          np.mean((exp_conc_man_raffinate - sim_man_raffinate_interp) ** 2) + \
          np.mean((exp_conc_gal_raffinate - sim_gal_raffinate_interp) ** 2)
    return mse

# Optimize parameters
initial_params = [4.55, 2.77, 54, 84]
bounds = [(1, 10), (1, 10), (10, 100), (10, 100)]
start_time = time.time()
result = minimize(objective, initial_params, bounds=bounds, method="L-BFGS-B", callback=lambda x: print(f"Iteration done, current params: {x}"))
elapsed_time = time.time() - start_time
print(f"Optimization completed in {elapsed_time:.2f} seconds")
print(f"Optimized Henry Constants: {result.x[:2]}")
print(f"Optimized Delta Values: {result.x[2:]}")
