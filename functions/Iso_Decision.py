import math

import functions.global_ as gl
from functions.singleLossFunctions.Single_Loss_Function_Choice import Single_Loss_Function_Choice


def Iso_Decision(expIso, params, lossFunc = 'Simple'):
    """Function witch assign linear or nonlinear isotherm to each compound
    based on ExpIso (one timeseries for each component).
    retired
    """
    choices = dict()
    for key, value in expIso.clusters.items():
        res = math.inf
        for choice in gl.solverChoices:
            lossFuncVal = Single_Loss_Function_Choice(lossFunc, params, value[0])
            if lossFuncVal < res:
                choices[key] = choice
                res = lossFuncVal
    return choices