from SMB.SMBStation import SMBStation
from SMB.NonlinColumn import NonLinColumn

from SMB.Tube import Tube
import math
import matplotlib.pyplot as plt

import numpy as np
import sys
import time

smb = SMBStation()

# flow rates in zones
flow_rate_zone1 = 180 # [mL/h]
flow_rate_zone2 = 93 # [mL/h]
flow_rate_zone3 = 114 # [mL/h]
flow_rate_zone4 = 45 # [mL/h]
'''
flow_rate_recycle = flow_rate_zone4
flow_rate_feed = flow_rate_zone3 - flow_rate_zone2
flow_rate_eluent = flow_rate_zone1 - flow_rate_zone4
flow_rate_extract = flow_rate_zone1 - flow_rate_zone2
flow_rate_raffinate = flow_rate_zone3 - flow_rate_zone4
'''

# switch interval
switch_interval = 780 # s

# discretization params
dt = 1 # s
Nx = 32 # number of spatial points

# column params
column_length = 280 # [mm]
column_diameter = 10 # [mm]
porosity = 0.376
dead_volume = 0.5 # [mL]

# define component 1
name1 = "Man"
feed_concentration1 = 9 # [mL/g]
langmuirConst1 = 0.74
saturCoef1 = 13.44
delta1 = 15.41
Di1 = 0.0007 # [mm2/s]

# define component 2
name2 = "Gal"
feed_concentration2 = 6 # [mL/g]
langmuirConst2 = 0.27
saturCoef2 = 18.13
delta2 = 15.41
Di2 = 0.0007 # [mm2/s]


num_of_steps_to_save = 10 # every Xth step will be saved to csv
num_of_steps_to_plot = 1 # every Xth save step will be shown on graph
# for example, if dt = 0.5, num_of_steps_to_save = 10 and num_of_steps_to_plot = 2
# calculation will run with step 0.5s
# saved will be each 10th step ... 5s
# ploted will be each 10*2 = 20th step ... 10s

cutoff_treshold = 500 # number of switches, after which the oldest values will be cutoff (set to sys.maxsize to prevent cutoff)

#for zone in range(1, 5):
#    smb.addColZone(zone, LinColumn(column_length, column_diameter, porosity), Tube(dead_volume))
smb.addColZone(1, NonLinColumn(column_length, column_diameter, porosity), Tube(dead_volume))
smb.addColZone(2, NonLinColumn(column_length, column_diameter, porosity), Tube(dead_volume))
smb.addColZone(3, NonLinColumn(column_length, column_diameter, porosity), Tube(dead_volume))
smb.addColZone(4, NonLinColumn(column_length, column_diameter, porosity), Tube(dead_volume))

smb.setFlowRateZone(1, flow_rate_zone1)
smb.setFlowRateZone(2, flow_rate_zone2)
smb.setFlowRateZone(3, flow_rate_zone3)
smb.setFlowRateZone(4, flow_rate_zone4)
smb.setSwitchInterval(switch_interval)
smb.setdt(dt)
smb.setNx(Nx)

smb.createNonLinComponentAB(name1, feed_concentration1, -1, delta1, Di1, langmuirConst1, saturCoef1)
smb.createNonLinComponentAB(name2, feed_concentration2, -1, delta2, Di2, langmuirConst2, saturCoef2)

smb.initCols()
print(smb.components)

file = open("result1.csv", "x")
file.write("time,man_extract,gal_extract,man_raffinate,gal_raffinate,switch_position\n")

result_dict = {}
result_dict["extract"] = {}
result_dict["raffinate"] = {}
result_dict["extract"][0] = []
result_dict["extract"][1] = []
result_dict["raffinate"][0] = []
result_dict["raffinate"][1] = []
maxval1 = 0
maxval2 = 0
f1 = plt.figure(num='Extract')
f2 = plt.figure(num='Raffinate')
sim_time = 0
plot_counter = 0

# --- keep the variables above as-is (sim_time, result_dict, etc.) ---

while True:
    # advance num_of_steps_to_save * dt with the FAST path
    for _ in range(num_of_steps_to_save):
        c_ex_man, c_ex_gal, c_ra_man, c_ra_gal = smb.step_fast_outlets()  # fast scalar outputs
        sim_time += dt

    # record last values
    result_dict["extract"][0].append(c_ex_man)
    result_dict["extract"][1].append(c_ex_gal)
    result_dict["raffinate"][0].append(c_ra_man)
    result_dict["raffinate"][1].append(c_ra_gal)

    # track maxima safely
    maxval1 = max(maxval1, c_ex_man, c_ex_gal)
    maxval2 = max(maxval2, c_ra_man, c_ra_gal)

    # log to csv
    file.write(f"{sim_time},{c_ex_man},{c_ex_gal},{c_ra_man},{c_ra_gal},{smb.switchState}\n")
    file.flush()

    # plotting cadence
    plot_counter += 1
    if plot_counter < num_of_steps_to_plot:
        # drop to keep one value per cadence (as your original code intended)
        result_dict["extract"][0].pop()
        result_dict["extract"][1].pop()
        result_dict["raffinate"][0].pop()
        result_dict["raffinate"][1].pop()
        continue
    plot_counter = 0

    # ---- plotting (unchanged structure, but with the safe ymax below) ----
    def _ymax(v): return 1.0 if v <= 0 else v * 1.2

    plt.figure('Extract')
    if sim_time <= cutoff_treshold * switch_interval:
        plt.axis([0, sim_time, 0, _ymax(maxval1)])
        x_axis = np.linspace(0, sim_time, len(result_dict["extract"][0]))
    else:
        plt.axis([sim_time - (cutoff_treshold * switch_interval), sim_time, 0, _ymax(maxval1)])
        result_dict["extract"][0].pop(0)
        result_dict["extract"][1].pop(0)
        result_dict["raffinate"][0].pop(0)
        result_dict["raffinate"][1].pop(0)
        x_axis = np.linspace(sim_time - (cutoff_treshold * switch_interval), sim_time, len(result_dict["extract"][0]))
    plt.plot(x_axis, result_dict["extract"][0], label=name1)
    plt.plot(x_axis, result_dict["extract"][1], label=name2)
    plt.legend(); plt.title("Extract")

    plt.figure('Raffinate')
    if sim_time <= cutoff_treshold * switch_interval:
        plt.axis([0, sim_time, 0, _ymax(maxval2)])        # <-- use maxval2 here
    else:
        plt.axis([sim_time - (cutoff_treshold * switch_interval), sim_time, 0, _ymax(maxval2)])  # <-- and here
    plt.plot(x_axis, result_dict["raffinate"][0], label=name1)
    plt.plot(x_axis, result_dict["raffinate"][1], label=name2)
    plt.legend(); plt.title("Raffinate")

    plt.pause(0.005)
    plt.clf(); plt.figure('Extract'); plt.clf()

