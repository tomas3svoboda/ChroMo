# Function calculates loss value for level 2 sub-optimization of the main bi-level optimization

from functions.singleLossFunctions.Single_Loss_Function_Choice import Single_Loss_Function_Choice

"""
Loss Function options:
    Default or 'Simple' - Single_Loss_Function_Simple
    'Squares' - Single_Loss_Function_Squares
    'LogSimple' - Single_Loss_Function_LogSimple
    'LogSquares' - Single_Loss_Function_LogSquares
"""
def Lev2_Loss_Function(params, experimentCluster, porosity, lossFunction = 'Simple', factor = 1, solver = "Lin"):
    params2 = [porosity, params[0], params[1]]
    sum = 0
    for comp in experimentCluster:
        res = Single_Loss_Function_Choice(lossFunction, params2, comp, solver, factor)
        sum += res
    #print("lEVEL 2 Loss function value: " + str(sum))
    return sum
