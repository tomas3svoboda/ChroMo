# Function calculates loss value for level 2 sub-optimization of the main bi-level optimization
from functions.Single_Loss_Function_Simple import Single_Loss_Function_Simple
import functions.global_ as gl

def Lev2_Loss_Function(params, experimentCluster, porosity):
    params2 = [porosity, params[0], params[1]]
    #print(params2)
    sum = 0
    for comp in experimentCluster:
        res = Single_Loss_Function_Simple(params2, comp)
        sum += res
    #print("lEVEL 2 Loss function value: " + str(sum))
    return sum
