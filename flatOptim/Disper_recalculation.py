import math

diameter = 16
length = 235
porosity = 0.3752
dispCorrelParam = 27

flowSpeed = (150 * 1000/3600) / ((math.pi * (diameter**2) / 4) * porosity)
disperCoef = (1/dispCorrelParam) * length * flowSpeed + 0.0001
print(disperCoef)


