from functions.singleLossFunctions.Single_Loss_Function_Choice import Single_Loss_Function_Choice
import math
from typing import List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

def Multi_Loss_Function_Wrapper(params: List[float],
                                B: List[float],
                                choice: str,
                                experimentCluster: Any,
                                solver: str = 'Lin',
                                factor: float = 3,
                                spacialDiff: int = 30,
                                timeDiff: int = 3000,
                                time: int = 10800,
                                fixedPorosity: float = 0,
                                fixedA: List[float] = []) -> float:
    """
    Function allowing to choose between loss function based on choice parameter.

    Args:
        params (List[float]): Model parameters.
        B (List[float]): Fixed diffusion coefficient.
        choice (str): Choice of loss function ('Simple', 'Squares', 'LogSimple', 'LogSquares').
        experimentCluster (Any): Data.
        solver (str): Solver selection ('Lin' or 'Nonlin'). Default is 'Lin'.
        factor (float): Normalization factor. Default is 3.
        spacialDiff (int): Number of spatial differences. Default is 30.
        timeDiff (int): Number of time differences. Default is 3000.
        time (int): Time span in seconds. Default is 10800.
        fixedPorosity (float): Fixed porosity value. Default is 0.
        fixedA (List[float]): Bodenstein numbers. Default is an empty list.

    Returns:
        float: Accumulated loss function result.
    """

    if choice not in ['Simple', 'Squares', 'LogSimple', 'LogSquares']:
        raise ValueError(f"Invalid choice of loss function: {choice}")

    porosity = params[0]
    addidx = 1 if not fixedPorosity else 0
    porosity = fixedPorosity if fixedPorosity else porosity

    def compute_single_loss(params, B, choice, experimentComp, solver, factor, spacialDiff, timeDiff, time, porosity, idx, addidx, fixedA, is_linear):
        flowRate = experimentComp.experiment.experimentCondition.flowRate
        diameter = experimentComp.experiment.experimentCondition.columnDiameter
        length = experimentComp.experiment.experimentCondition.columnLength
        flowSpeed = (flowRate * 1000 / 3600) / ((math.pi * (diameter ** 2) / 4) * porosity)

        if fixedA:
            disperCoef = (1 / fixedA[idx]) * length * flowSpeed + B[idx]
            tmp = [porosity, params[idx + addidx], disperCoef]
        else:
            disperCoef = (1 / params[idx * (2 if is_linear else 3) + addidx + 1]) * length * flowSpeed + B[idx]
            tmp = [porosity, params[idx * (2 if is_linear else 3) + addidx], disperCoef]
            if not is_linear:
                tmp.append(params[idx * 3 + addidx + 2])

        return Single_Loss_Function_Choice(choice, tmp, experimentComp, solver, factor, spacialDiff, timeDiff, time)

    is_linear = solver == "Lin"
    futures = []

    with ProcessPoolExecutor() as executor:
        for idx, pair in enumerate(experimentCluster.clusters.items()):
            for experimentComp in pair[1]:
                futures.append(executor.submit(compute_single_loss, params, B, choice, experimentComp, solver, factor, spacialDiff, timeDiff, time, porosity, idx, addidx, fixedA, is_linear))

        sum_loss = 0
        for future in as_completed(futures):
            sum_loss += future.result()

    return sum_loss
