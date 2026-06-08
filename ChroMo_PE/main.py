import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')   # must be before any other import touches matplotlib.pyplot

# Must run before any library (pandas, scipy, …) loads its own OpenMP runtime.
# Two OpenMP runtimes in the same Windows process cause an access-violation crash.
try:
    import importlib.util as _ilu, pathlib as _pl
    _cpp_root = _pl.Path(__file__).parent
    _cpp_lib = next((_cpp_root / f"edm_nonlinear_solver{ext}"
                     for ext in (".pyd", ".so")
                     if (_cpp_root / f"edm_nonlinear_solver{ext}").exists()), None)
    if _cpp_lib:
        _spec = _ilu.spec_from_file_location("edm_nonlinear_solver", str(_cpp_lib))
        _cpp_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_cpp_mod)
        sys.modules["edm_nonlinear_solver"] = _cpp_mod
        print("C++ solver library loaded.")
    del _ilu, _pl, _cpp_root, _cpp_lib, _spec, _cpp_mod
except Exception as _e:
    print(f"C++ solver not available: {_e}")

from ChroMo_PE.objects.Operator import Operator
from ChroMo_PE.functions.WebServerStuff.Web_Server import Web_Server

'''
YOU NEED TO START MONGO DB AS ADMINISTRATOR FIRST
- open command prompt as Administrator
- cd "C:\Program Files\MongoDB\Server\8.0\bin"
- .\mongod.exe --dbpath "C:\Program Files\MongoDB\Server\8.0\data" --port 27017 --logpath "C:\Program Files\MongoDB\Server\8.0\log\mongod.log" --logappend
- NOW IT WILL RUN!
- log in with admin admin
'''

def _port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) == 0


if __name__ == '__main__':
    SERVER_PORT = 6969
    if _port_in_use(SERVER_PORT):
        print()
        print("=" * 62)
        print(f"  ERROR: port {SERVER_PORT} is already in use!")
        print("  Another server instance is probably still running.")
        print("  Stop it first, then restart this script.")
        print()
        print("  To find and kill the process on Windows:")
        print(f"    netstat -ano | findstr :{SERVER_PORT}")
        print("    taskkill /PID <pid> /F")
        print("=" * 62)
        print()
        sys.exit(1)

    Web_Server()

    operator = Operator()

    # path to directory with experiment files
    path = "C:\\Users\\z004d8nt\\PycharmProjects\\DisertationTSV\\ChroMo_PE\\data\\Suc_Glu_GE"
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
