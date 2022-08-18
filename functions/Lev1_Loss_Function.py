# Function calculates loss value for the level 1 main bi-level optimization.
# The value is calculated as sum of loss values from level 2 sub-optimizations.
from functions.Lev2_Optim import Lev2_Optim
import functions.global_ as gl

def Lev1_Loss_Function(porosity, experimentClustersComp):
    sum = 0
    for key in experimentClustersComp.clusters:
        res = Lev2_Optim(porosity[0], experimentClustersComp.clusters[key], key)
        sum += res
    print("Loss function value: " + str(sum))
    return sum