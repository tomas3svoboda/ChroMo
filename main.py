from objects.Operator import Operator
from functions.WebServerStuff.Web_Server import Web_Server
import os

if __name__ == '__main__':
    operator = Operator()

    # path to directory with experiment files
    path = "C:\\Users\\Adam\\ChroMo\\docu\\LossFunctionExperimentSet"
    experimentSet = operator.Load_Experiment_Set(path)
    fit_gauss = True
    ret_time_corr = True
    ret_time_threshold = 0.005
    mass_bal_corr = True
    loss_function_type = 'Squares' # 'Simple', 'LogSimple', 'LogSquares'
    solver = 'Lin' # 'Nonlin
    factor = 1
    lvl1_dict = {'pinit': 0.5, 'prange': [0.4, 0.9]}
    lvl2_dict = {'Gal': {'kinit': 5.0, 'krange': [1.0, 10.0], 'dinit': 5.0, 'drange': [1.0, 10.0], 'qinit': 0, 'qrange': [0, 0]},
                'Man': {'kinit': 5.0, 'krange': [1.0, 10.0], 'dinit': 5.0, 'drange': [1.0, 10.0], 'qinit': 0, 'qrange': [0, 0]}}
    time_diff = 3000
    space_diff = 30
    time = 10800
    lvl1_optim_settings = {'algorithm': '2', 'settings': {'maxiter': '', 'maxfev': '', 'xatol': '', 'fatol': '', 'aptive': '0'}}
    lvl2_optim_settings = {'algorithm': '2', 'settings': {'maxiter': '', 'maxfev': '', 'xatol': '', 'fatol': '', 'aptive': '0'}}
    optim_type = "bilevel"
    fix_porosity = False

    # number of params in lvl1_dict and lvl2_dict has to match optim_type, fix_porosity and solver options combination
    result = operator.Web_Start(experimentSet,
                      fit_gauss,
                      ret_time_corr,
                      mass_bal_corr,
                      loss_function_type,
                      solver,
                      factor,
                      lvl1_dict,
                      lvl2_dict,
                      space_diff,
                      time_diff,
                      time,
                      1,
                      ret_time_threshold,
                      lvl1_optim_settings,
                      lvl2_optim_settings,
                      optim_type,
                      fix_porosity
                      )
    print(result)