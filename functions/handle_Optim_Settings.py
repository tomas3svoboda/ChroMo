from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import shgo

def handle_Optim_Settings(func, x, args, bounds, optimInfo, default=0):
    """Function that handles calling correct optimization function based on input parameters"""
    if not optimInfo:
        if default == 0:
            return minimize(func,
                           x,
                           args=args,
                           bounds=bounds,
                           method='Nelder-Mead')
        elif default == 1:
            return shgo(func=lambda x: func(x, *args),
                        bounds=bounds,
                        args=())
    if optimInfo["algorithm"] == "1":
        if optimInfo["settings"]["Ns"]:
            return brute(func,
                           ranges=bounds,
                           args=args,
                           Ns=int(optimInfo["settings"]["Ns"]),
                           full_output=True,
                           finish=None)
        else:
            return brute(func,
                           ranges=bounds,
                           args=args,
                           full_output=True,
                           finish=None)
    elif optimInfo["algorithm"] == "2":
        options = {}
        if optimInfo["settings"]["maxiter"]:
            options["maxiter"] = int(optimInfo["settings"]["maxiter"])
        if optimInfo["settings"]["maxfev"]:
            options["maxfev"] = int(optimInfo["settings"]["maxfev"])
        if optimInfo["settings"]["xatol"]:
            options["xatol"] = float(optimInfo["settings"]["xatol"])
        if optimInfo["settings"]["fatol"]:
            options["fatol"] = float(optimInfo["settings"]["fatol"])
        if optimInfo["settings"]["aptive"]:
            options["adaptive"] = bool(int(optimInfo["settings"]["aptive"]))
        return minimize(func,
                       x,
                       args=args,
                       bounds=bounds,
                       method='Nelder-Mead',
                       options=options)
    elif optimInfo["algorithm"] == "3":
        n = 100
        if optimInfo["settings"]["n"]:
            n = optimInfo["settings"]["n"]
        iters = 1
        if optimInfo["settings"]["iters"]:
            iters = optimInfo["settings"]["iters"]
        options = {}
        if optimInfo["settings"]["maxev"]:
            options["maxev"] = int(optimInfo["settings"]["maxev"])
        if optimInfo["settings"]["maxiter"]:
            options["maxiter"] = int(optimInfo["settings"]["maxiter"])
        if optimInfo["settings"]["maxfev"]:
            options["maxfev"] = int(optimInfo["settings"]["maxfev"])
        if optimInfo["settings"]["maxtime"]:
            options["maxtime"] = float(optimInfo["settings"]["maxtime"])
        if optimInfo["settings"]["f_tol"]:
            options["f_tol"] = float(optimInfo["settings"]["f_tol"])
        if optimInfo["settings"]["f_min"]:
            options["f_min"] = float(optimInfo["settings"]["f_min"])
        return shgo(
            func=lambda x: func(x, *args),
            bounds=bounds,
            args=(),
            n=n,
            iters=iters,
            options=options)
    elif optimInfo["algorithm"] == "4":
        options = {}
        if optimInfo["settings"]["maxiter"]:
            options["maxiter"] = int(optimInfo["settings"]["maxiter"])
        print("x", x)
        print("args", args)
        print("bounds", bounds)
        return minimize(func,
                   x,
                   args=args,
                   bounds=bounds,
                   method='Powell',
                   options=options)
    else:
        raise "Unknown algorithm choice in handle_Optim_Settings"