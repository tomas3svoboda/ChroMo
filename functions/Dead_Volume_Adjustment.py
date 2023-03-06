import numpy as np

def Dead_Volume_Adjustment(dataArray, deadVolume, flowRate, dt = 3.6):
    t = (deadVolume/flowRate)*3600
    deadSteps = int(t//dt)
    remainder = t%dt
    if remainder >= dt/2:
        deadSteps += 1
    aditionalSteps = np.zeros(deadSteps)
    return np.concatenate([aditionalSteps, dataArray])
