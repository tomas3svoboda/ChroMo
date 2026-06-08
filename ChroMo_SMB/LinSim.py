from EKF.SMB.SMBStation import SMBStation
from EKF.SMB.LinColumn import LinColumn
from EKF.SMB.Tube import Tube
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

# flow rates in zones
flow_rate_zone1 = 180 # [mL/h]
flow_rate_zone2 = 44.1 # [mL/h]
flow_rate_zone3 = 47.5 # [mL/h]
flow_rate_zone4 = 47.5 # [mL/h]
'''
flow_rate_recycle = flow_rate_zone4
flow_rate_feed = flow_rate_zone3 - flow_rate_zone2
flow_rate_eluent = flow_rate_zone1 - flow_rate_zone4
flow_rate_extract = flow_rate_zone1 - flow_rate_zone2
flow_rate_raffinate = flow_rate_zone3 - flow_rate_zone4
'''

# switch interval
switch_interval = 4300 # s

# discretization params
dt = 0.5 # s
Nx = 100 # number of spatial points

# column params
column_length = 320 # [mm]
column_diameter = 10 # [mm]
porosity = 0.376
dead_volume = 0.5 # [mL]

# define component 1
name1 = "Man"
feed_concentration1 = 9 # [mL/g]
henry_constant1 = 4.55
delta1 = 54
Di1 = 0.0007 # [mm2/s]

# define component 2
name2 = "Gal"
feed_concentration2 = 6 # [mL/g]
henry_constant2 = 2.77
delta2 = 84
Di2 = 0.0007 # [mm2/s]


num_of_steps_to_save = 10 # every Xth step will be saved to csv
num_of_steps_to_plot = 20 # every Xth save step will be shown on graph
# for example, if dt = 0.5, num_of_steps_to_save = 10 and num_of_steps_to_plot = 2
# calculation will run with step 0.5s
# saved will be each 10th step ... 5s
# ploted will be each 10*2 = 20th step ... 10s

cutoff_treshold = 500 # number of switches, after which the oldest values will be cutoff (set to sys.maxsize to prevent cutoff)

smb = SMBStation()

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
print(smb.components)

file = open("result10.csv", "x")
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
while True:
    plot_counter += 1
    res = smb.step(num_of_steps_to_save)
    sim_time += num_of_steps_to_save * dt
    for idx, x in enumerate(res[1][-1]): # loop through components in last column in zone 1
        result_dict["extract"][idx].append(x[-1]) # append last conc value of given component
    for idx, x in enumerate(res[3][-1]): # loop through components in last column in zone 3
        result_dict["raffinate"][idx].append(x[-1]) # append last conc value of given component
    if result_dict["extract"][0][-1] > maxval1:
        maxval1 = result_dict["extract"][0][-1]
    if result_dict["extract"][1][-1] > maxval1:
        maxval1 = result_dict["extract"][1][-1]

    file.write(str(sim_time) + "," + str(result_dict["extract"][0][-1]) + "," + str(result_dict["extract"][1][-1]) + "," + str(result_dict["raffinate"][0][-1]) + "," + str(result_dict["raffinate"][1][-1]) + "," + str(smb.switchState) + "\n")
    file.flush()

    if plot_counter < num_of_steps_to_plot:
        result_dict["extract"][0].pop()
        result_dict["extract"][1].pop()
        result_dict["raffinate"][0].pop()
        result_dict["raffinate"][1].pop()
        continue
    else:
        plot_counter = 0
        plt.figure('Extract')
        if sim_time <= cutoff_treshold * switch_interval:
            plt.axis([0, sim_time, 0, maxval1*1.2])
            x_axis = np.linspace(0, sim_time, len(result_dict["extract"][0]))
        else:
            plt.axis([sim_time - (cutoff_treshold * switch_interval), sim_time, 0, maxval1*1.2])
            result_dict["extract"][0].pop(0)
            result_dict["extract"][1].pop(0)
            result_dict["raffinate"][0].pop(0)
            result_dict["raffinate"][1].pop(0)
            x_axis = np.linspace(sim_time - (cutoff_treshold * switch_interval), sim_time, len(result_dict["extract"][0]))
        plt.plot(x_axis, result_dict["extract"][0], label = name1)
        plt.plot(x_axis, result_dict["extract"][1], label = name2)
        plt.legend()
        plt.title("Extract")
        if result_dict["raffinate"][0][-1] > maxval2:
            maxval2 = result_dict["raffinate"][0][-1]
        if result_dict["raffinate"][1][-1] > maxval2:
            maxval2 = result_dict["raffinate"][1][-1]
        plt.figure('Raffinate')
        if sim_time <= cutoff_treshold * switch_interval:
            plt.axis([0, sim_time, 0, maxval2*1.2])
        else:
            plt.axis([sim_time - (cutoff_treshold * switch_interval), sim_time, 0, maxval1*1.2])
        plt.plot(x_axis, result_dict["raffinate"][0], label = name1)
        plt.plot(x_axis, result_dict["raffinate"][1], label = name2)
        plt.legend()
        plt.title("Raffinate")
        plt.pause(0.005)
        plt.clf()
        plt.figure('Extract')
        plt.clf()
