
from ChroMo_PE.functions.solvers.Lin_Solver import Lin_Solver
from ChroMo_PE.functions.solvers.Nonlin_Solver import Nonlin_Solver
import ChroMo_PE.functions.solvers.CADET_Solver as _cadet_module


def Solver_Choice(choice, params, experimentComp, spacialDiff=30, timeDiff=3000, time=10800, debugPrint=False, full=False):
    """Function allowing to choose between solvers based on choice parameter.
    Choices:
    'Lin'          - EDM Linear (Henry) solver
    'Nonlin'       - EDM Nonlinear (Langmuir) solver
    'CADET_Lin'    - CADET Linear (Henry) solver
    'CADET_Nonlin' - CADET Nonlinear (Langmuir) solver
    """
    if choice == 'Lin':
        res = Lin_Solver(experimentComp.experiment.experimentCondition.flowRate,
                         experimentComp.experiment.experimentCondition.columnLength,
                         experimentComp.experiment.experimentCondition.columnDiameter,
                         experimentComp.experiment.experimentCondition.feedVolume,
                         experimentComp.feedConcentration,
                         params[0], params[1], params[2],
                         spacialDiff, timeDiff, time,
                         debugPrint=debugPrint, full=full)

    elif choice == 'Nonlin':
        res = Nonlin_Solver(experimentComp.experiment.experimentCondition.flowRate,
                            experimentComp.experiment.experimentCondition.columnLength,
                            experimentComp.experiment.experimentCondition.columnDiameter,
                            experimentComp.experiment.experimentCondition.feedVolume,
                            experimentComp.feedConcentration,
                            params[0], params[1], params[2], params[3],
                            spacialDiff, timeDiff, time,
                            debugPrint=debugPrint, full=full)

    elif choice == 'CADET_Lin':
        molar_mass = _cadet_module.cadet_molar_mass.get(
            experimentComp.name, _cadet_module.CADET_DEFAULT_MOLAR_MASS)
        # ncol=None → solver reads cadet_ncol module variable (FVM cells, independent of spacialDiff)
        res = _cadet_module.CADET_Lin_Solver(
            experimentComp.experiment.experimentCondition.flowRate,
            experimentComp.experiment.experimentCondition.columnLength,
            experimentComp.experiment.experimentCondition.columnDiameter,
            experimentComp.experiment.experimentCondition.feedVolume,
            experimentComp.feedConcentration,
            params[0], params[1], params[2],
            ncol=None, Nt=timeDiff, time=time,
            debugPrint=debugPrint, full=full,
            molar_mass=molar_mass)

    elif choice == 'CADET_Nonlin':
        molar_mass = _cadet_module.cadet_molar_mass.get(
            experimentComp.name, _cadet_module.CADET_DEFAULT_MOLAR_MASS)
        res = _cadet_module.CADET_Nonlin_Solver(
            experimentComp.experiment.experimentCondition.flowRate,
            experimentComp.experiment.experimentCondition.columnLength,
            experimentComp.experiment.experimentCondition.columnDiameter,
            experimentComp.experiment.experimentCondition.feedVolume,
            experimentComp.feedConcentration,
            params[0], params[1], params[2], params[3],
            ncol=None, Nt=timeDiff, time=time,
            debugPrint=debugPrint, full=full,
            molar_mass=molar_mass)

    else:
        raise Exception(str(choice) + ' - Unknown Solver in Solver_Choice')
    return res
