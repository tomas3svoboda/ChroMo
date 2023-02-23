
from functions.solvers.Lin_Solver import Lin_Solver
from functions.solvers.Nonlin_Solver import Nonlin_Solver

'''
Function allowing to choose between solvers based on choice parameter
Choices:
    'Lin' - Lin_Solver
Other choices will raise an exception
IN PROGRESS - needs add more solvers
'''
def Solver_Choice(choice, params, experimentComp, spacialDiff = 30, timeDiff = 3000, time = 10800, full=False):
    if choice == 'Lin':
        res = Lin_Solver(experimentComp.experiment.experimentCondition.flowRate,
                            experimentComp.experiment.experimentCondition.columnLength,
                            experimentComp.experiment.experimentCondition.columnDiameter,
                            experimentComp.experiment.experimentCondition.feedVolume,
                            experimentComp.feedConcentration,
                            params[0],
                            params[1],
                            params[2],
                            spacialDiff,
                            timeDiff,
                            time,
                            full=full)
    elif choice == 'Nonlin':
        res = Nonlin_Solver(experimentComp.experiment.experimentCondition.flowRate,
                            experimentComp.experiment.experimentCondition.columnLength,
                            experimentComp.experiment.experimentCondition.columnDiameter,
                            experimentComp.experiment.experimentCondition.feedVolume,
                            experimentComp.feedConcentration,
                            params[0],
                            params[1],
                            params[2],
                            params[3],
                            spacialDiff,
                            timeDiff,
                            time,
                            full=full)
    else:
        raise Exception(str(choice) + ' - Unknown Solver in Solver_Choice')
    return res