from functions.Nonlin_Solver import Nonlin_Solver
import numpy as np
import math

def Covergence_Check(threshold = 0.01, flowRate = 800, length = 235, diameter = 16, feedVol = 5, feedConc = 2, porosity = 0.5,
                  langmuirConst = 2.5, disperCoef = 0.95, saturationConst = 1):
    feedTime = (feedVol / flowRate) * 3600
    flowSpeed = (flowRate * 1000 / 3600) / (math.pi * ((diameter / 2) ** 2) * porosity)
    def Loss_Func(Nt):
        mfeed = feedConc * feedTime * flowSpeed
        c = Nonlin_Solver(flowRate, length, diameter, feedVol, feedConc, porosity, langmuirConst, disperCoef,
                          saturationConst, 180, Nt)
        mn = (flowSpeed*math.pi*(diameter**2)*porosity)/4 * np.trapz(c[:, Nt])
        return abs(mfeed-mn)
    #use minimize_scalar
    convergenceCheck = 999999
    convergenceCount = 0
    k = 1000
    while True:
        x = Loss_Func(k)
        if x < threshold:
            return True
        if x > convergenceCheck:
            convergenceCount += 1
        else:
            convergenceCount = 0
        if convergenceCount > 10:
            return False
        convergenceCheck = x
        k = k*2
