from scipy.optimize import minimize
from objectiveSMB import objectiveSMB

def parameterEstimationSMB(
    experimental_data_path,
    end_time,
    initial_guess,          # [K_A, q_mA, K_B, q_mB, delta_shared]
    bounds,                 # list of 5 (low, high)
    flow_rates=None,
    switch_interval=780,
    dt=1, Nx=30, Di=7.0e-4,
    name_compA="Man", feed_concA=7.27,
    name_compB="Gal", feed_concB=3.42,
    # ---- progress controls ----
    print_eval=True,                # print each objective eval (params + MSE)
    print_iter=True,                # print each optimizer iteration (best-so-far xk)
    eval_log_csv=None,              # CSV path for logging all evals (None to disable)
):
    """
    Fit NonLin Langmuir params: K_A, q_mA, K_B, q_mB, delta_shared.
    Flow rates and switching time remain fixed.
    """

    # optional CSV header
    if eval_log_csv:
        with open(eval_log_csv, "w", encoding="utf-8") as f:
            f.write("eval_idx,K_A,q_mA,K_B,q_mB,delta_shared,MSE\n")

    eval_idx = [0]
    def obj(p):
        K_A, qm_A, K_B, qm_B, delta_shared = p
        mse = objectiveSMB(
            end_time=end_time,
            experimental_data_path=experimental_data_path,
            name_compA=name_compA, feed_concA=feed_concA,
            name_compB=name_compB, feed_concB=feed_concB,
            K_A=K_A, qm_A=qm_A, K_B=K_B, qm_B=qm_B,
            delta_shared=delta_shared,
            Di=Di,
            flow_rates=flow_rates,
            switch_interval=switch_interval,
            dt=dt, Nx=Nx,
            verbose=False,
        )
        eval_idx[0] += 1
        if print_eval:
            print(f"[eval {eval_idx[0]:04d}] MSE={mse:.6g} | "
                  f"K_A={K_A:.6g}, q_mA={qm_A:.6g}, K_B={K_B:.6g}, q_mB={qm_B:.6g}, δ={delta_shared:.6g}")
        if eval_log_csv:
            with open(eval_log_csv, "a", encoding="utf-8") as f:
                f.write(f"{eval_idx[0]},{K_A},{qm_A},{K_B},{qm_B},{delta_shared},{mse}\n")
        return mse

    it_idx = [0]
    def _callback(xk):
        if not print_iter:
            return
        it_idx[0] += 1
        print(f"[iter {it_idx[0]:03d}] best_so_far: "
              f"K_A={xk[0]:.6g}, q_mA={xk[1]:.6g}, K_B={xk[2]:.6g}, q_mB={xk[3]:.6g}, δ={xk[4]:.6g}")

    res = minimize(
        obj, x0=initial_guess, method="L-BFGS-B", bounds=bounds,
        options={
            "maxiter": 1000,
            "ftol": 1e-5,     # stop when improvement is tiny
            "eps": 1e-2       # finite-diff step; tune 1e-3–1e-2 in your params' units
        }
    )

    return {
        "K_A": float(res.x[0]),
        "q_mA": float(res.x[1]),
        "K_B": float(res.x[2]),
        "q_mB": float(res.x[3]),
        "delta_shared": float(res.x[4]),
        "MSE": float(res.fun),
        "success": bool(res.success),
        "message": res.message,
        "nfev": int(res.nfev),
        "nit": int(getattr(res, "nit", 0)),
    }

if __name__ == "__main__":
    # Example usage (adjust paths and guesses to your dataset)
    experimental_data_path = "SMB_onePeriond_experiment5_1.xlsx"
    end_time = 4700
    flow_rates = [180.0, 93.0, 114.0, 45.0]
    switch_interval = 780
    initial_guess = [0.74, 13.44, 0.27, 18.13, 15.41]  # [K_A, q_mA, K_B, q_mB, delta]
    bounds = [(0.1, 2.5), (1.0, 60.0), (0.1, 2.5), (1.0, 60.0), (5.0, 100.0)]

    result = parameterEstimationSMB(
        experimental_data_path=experimental_data_path,
        end_time=end_time,
        initial_guess=initial_guess,
        bounds=bounds,
        flow_rates=flow_rates,
        switch_interval=switch_interval,
        dt=1, Nx=30, Di=7.0e-4,
        name_compA="Man", feed_concA=7.27,
        name_compB="Gal", feed_concB=3.42,
        print_eval=True,
        print_iter=True,
        eval_log_csv=None,  # e.g., "/mnt/data/fit_log.csv"
    )
    print("Final:", result)

'''if __name__ == "__main__":
    experimental_data_path = "SMB_onePeriond_experiment5_1.xlsx"
    end_time = 7860
    flow_rates = [180.0, 93.0, 114.0, 45.0]
    switch_interval = 780
    # Reasonable starting guess for sugars on Dowex-like resin
    initial_guess = [7.14/5, 4.68/5, 13.41]   # [K_A, K_B, delta]
    bounds = [(0.25, 30.0), (0.25, 30.0), (3.0, 3000.0)]

    result = parameterEstimationSMB_linear(
        experimental_data_path=experimental_data_path,
        end_time=end_time,
        initial_guess=initial_guess,
        bounds=bounds,
        flow_rates=flow_rates,
        switch_interval=switch_interval,
        dt=0.5, Nx=100,
        name_compA="Man", feed_concA=7.27,
        name_compB="Gal", feed_concB=3.42,
        print_eval=True,
        print_iter=True,
        eval_log_csv=None,  # e.g., "/mnt/data/fit_log_linear.csv"
    )
    print("Final:", result)'''
