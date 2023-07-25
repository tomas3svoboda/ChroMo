
from functions.solvers.Lin_Solver import Lin_Solver
from functions.solvers.Nonlin_Solver import Nonlin_Solver


def Solver_Choice(choice, params, experimentComp, spacialDiff = 30, timeDiff = 3000, time = 10800, debugPrint=False, full=False):
    """Function allowing to choose between solvers based on choice parameter
    Choices:
    'Lin' - Lin_Solver
    'Nonlin' - Nonlin_Solver
    Other choices will raise an exception.
    """
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
                            debugPrint=debugPrint,
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
                            debugPrint=debugPrint,
                            full=full)
    else:
        raise Exception(str(choice) + ' - Unknown Solver in Solver_Choice')
    return res