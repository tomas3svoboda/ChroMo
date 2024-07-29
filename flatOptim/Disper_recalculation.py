import math

diameter = 16  # [mm]
length = 235 # [mm]
flowRate = 150 # [mL/h]
porosity = 0.3752  # [-]
Bo = 27.37

'''diameter = 16  # [mm]
length = 235  # [mm]
flowRate = 150  # [mL/h]
porosity = 0.3752  # [-]
Bo = 10'''

flowSpeed = (flowRate * 1000/3600) / ((math.pi * (diameter**2) / 4) * porosity)  # [mm/s]
disperCoef = (1/Bo) * length * flowSpeed  # [mm2/s]

print(disperCoef)

columnVolume = (math.pi * (diameter ** 2) / 4) * length  # [mm3]
print('column volume: ' + str(columnVolume) + ' mm3')
staticPhaseVolume = columnVolume - (columnVolume * porosity)  # [mm3]
voidVolume = columnVolume * porosity  # [mm3]
print('column void volume: ' + str(voidVolume) + ' mm3')
deadVolume = 2  # [mL]
deadVolume = deadVolume * 1000  # [mm3] unit conversion

staticPhaseCapacity = 376.53   # [g/L] upper limit for Sucrose
staticPhaseCapacity = staticPhaseCapacity / 10e6  # [g/mm3] unit conversion

maxAdsorbedMass = staticPhaseCapacity * staticPhaseVolume  # [g]

totalVoidVolume = deadVolume + voidVolume  # [mm3]
flowRate_conv = flowRate * 0.277778  # [mm3/s] unit conversion
deadTime = totalVoidVolume/flowRate_conv  # [s]

print('Dead time: ' + str(deadTime/60) + ' min')

print('Max adsorbed mass based on upper limit: ' + str(maxAdsorbedMass) + ' g')

estimatedQ = 6  # [g/L]
estimatedQ = estimatedQ / 10e6   # [g/mm3]

estimateAdsorbedMass = estimatedQ * staticPhaseVolume  # [g]

print('Max adsorbed mass based on estimated Q: ' + str(estimateAdsorbedMass) + ' g')

concFeed = 9.06  # [g/L] for sucrose
feedTime = 50  # [s]
flowRate_conv2 = flowRate / 3.6e+6  # [L/s]

feedMass = concFeed * feedTime * flowRate_conv2  # [g]
print('Fed mass: ' + str(feedMass) + ' g')

ratio = estimatedQ/staticPhaseCapacity
result = maxAdsorbedMass * ratio

print(result)
