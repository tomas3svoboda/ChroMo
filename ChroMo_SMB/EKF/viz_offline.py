import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("1756478837.761219ekf_log.csv")
channels = ["ExMan", "ExGal", "RaMan", "RaGal"]
time_key = "twin_time" if "twin_time" in df.columns else df.columns[0]

# 1) Outlet fit (measured vs model)
plt.figure()
for name in channels:
    yhat = f"yhat_{name}"
    ymeas = f"ymeas_{name}"
    if yhat in df.columns:
        plt.plot(df[time_key], df[yhat], label=f"{name} model")
    if ymeas in df.columns:
        plt.plot(df[time_key], df[ymeas], linestyle="--", label=f"{name} meas")
plt.legend(); plt.xlabel("Twin time [s]"); plt.ylabel("Concentration"); plt.title("Outlet fit")

# 2) Innovations (z-scores if available, otherwise raw innovations)
plt.figure()
have_any_z = any(f"z_{n}" in df.columns for n in channels)
if have_any_z:
    for name in channels:
        zcol = f"z_{name}"
        if zcol in df.columns:
            plt.plot(df[time_key], df[zcol], label=name)
    plt.axhline(3, linestyle=":"); plt.axhline(-3, linestyle=":")
    plt.ylabel("z-score"); plt.title("Normalized innovations (z)")
else:
    for name in channels:
        icol = f"innov_{name}"
        if icol in df.columns:
            plt.plot(df[time_key], df[icol], label=name)
    plt.axhline(0, linestyle=":")
    plt.ylabel("Innovation"); plt.title("Innovations (raw) — z-scores not present in CSV")
plt.legend(); plt.xlabel("Twin time [s]")

# 3) NIS with chi-square bounds (dof inferred from available measurements)
from math import isfinite
channels = ["ExMan", "ExGal", "RaMan", "RaGal"]
n_meas = sum(1 for name in channels if f"ymeas_{name}" in df.columns)
dof = n_meas if n_meas > 0 else 4

try:
    from scipy.stats import chi2
    p95 = chi2.ppf(0.95, df=dof)
    p99 = chi2.ppf(0.99, df=dof)
except Exception:
    # Fallback for common dof values if SciPy is not available
    fallback = {
        1: (3.841, 6.635),
        2: (5.991, 9.210),
        3: (7.815, 11.345),
        4: (9.488, 13.277),
        5: (11.070, 15.086)
    }
    p95, p99 = fallback.get(dof, (9.488, 13.277))

if "nis" in df.columns:
    plt.figure()
    plt.plot(df[time_key], df["nis"])
    plt.axhline(p95, linestyle=":", label=f"95% (dof={dof})")
    plt.axhline(p99, linestyle="--", label=f"99% (dof={dof})")
    plt.legend(); plt.xlabel("Twin time [s]"); plt.ylabel("NIS"); plt.title("Normalized Innovation Squared")

plt.show()
