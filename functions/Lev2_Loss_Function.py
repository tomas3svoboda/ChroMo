# Function calculates loss value for level 2 sub-optimization of the main bi-level optimization
from functions.Single_Loss_Function_Simple import Single_Loss_Function_Simple
from functions.Single_Loss_Function_Squares import Single_Loss_Function_Squares
from functions.Single_Loss_Function_LogSimple import Single_Loss_Function_LogSimple
from functions.Single_Loss_Function_LogSquares import Single_Loss_Function_LogSquares
import functions.global_ as gl

"""
Loss Function options:
    Default or 'Simple' - Single_Loss_Function_Simple
    'Squares' - Single_Loss_Function_Squares
    'LogSimple' - Single_Loss_Function_LogSimple
    'LogSquares' - Single_Loss_Function_LogSquares
"""
def Lev2_Loss_Function(params, experimentCluster, porosity, lossFunction = 'Simple'):
    params2 = [porosity, params[0], params[1]]
    func = Single_Loss_Function_Simple
    if lossFunction == 'Squares':
        func = Single_Loss_Function_Squares
    elif lossFunction == 'LogSimple':
        func = Single_Loss_Function_LogSimple
    elif lossFunction == 'LogSquares':
        func = Single_Loss_Function_LogSquares
    sum = 0
    for comp in experimentCluster:
        res = func(params2, comp)
        sum += res
    #print("lEVEL 2 Loss function value: " + str(sum))
    return sum
