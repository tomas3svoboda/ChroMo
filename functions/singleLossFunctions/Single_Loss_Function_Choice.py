
from functions.singleLossFunctions.Single_Loss_Function_Simple import Single_Loss_Function_Simple
from functions.singleLossFunctions.Single_Loss_Function_Squares import Single_Loss_Function_Squares
from functions.singleLossFunctions.Single_Loss_Function_LogSimple import Single_Loss_Function_LogSimple
from functions.singleLossFunctions.Single_Loss_Function_LogSquares import Single_Loss_Function_LogSquares

'''
Function allowing to choose between loss function based on choice parameter
Choices:
    'Simple' - Single_Loss_Function_Simple
    'Squares' - Single_Loss_Function_Squares
    'LogSimple' - Single_Loss_Function_LogSimple
    'LogSquares' - Single_Loss_Function_LogSquares
Other choices will raise an exception
'''
def Single_Loss_Function_Choice(choice, params, experimentComp, solver = 'Lin'):
    res = 0
    if choice == 'Simple':
        res = Single_Loss_Function_Simple(params, experimentComp, solver)
    elif choice == 'Squares':
        res = Single_Loss_Function_Squares(params, experimentComp, solver)
    elif choice == 'LogSimple':
        res = Single_Loss_Function_LogSimple(params, experimentComp, solver)
    elif choice == 'LogSquares':
        res = Single_Loss_Function_LogSquares(params, experimentComp, solver)
    else:
        raise Exception('Unknown Loss function in Single_Loss_Function_Choice')
    return res