#!/usr/bin/env python3
"""
eval_timing.py — measure Objective.evaluate() time with a heartbeat.

- Loads warm_10si.pkl
- Times baseline and perturbed evaluations
- Prints progress every 5 seconds so you’re never "blind"

Adjust CONFIG below (HORIZON_SI, REPEATS, etc.).
"""

import time, pickle, threading
import numpy as np
from mpc_objective import Objective

# ===================== CONFIG =====================
SNAPSHOT_FILE = "snapshot_backup/warm_10si.pkl"
HORIZON_SI    = 8          # two full cycles for 4 columns
REPEATS_BASE  = 1          # keep 1 to avoid very long runs
REPEATS_PERT  = 1
PERTURB       = np.array([+5.0, -2.0, +0.3, +50.0], dtype=float)  # (D,E,Q4,SI)
PROGRESS_EVERY_S = 5.0
# ==================================================

def _heartbeat(start_time: float, label: str, stop_evt: threading.Event):
    next_tick = PROGRESS_EVERY_S
    while not stop_evt.is_set():
        elapsed = time.time() - start_time
        if elapsed >= next_tick:
            print(f"[progress] {label}: t={elapsed:.1f}s")
            next_tick += PROGRESS_EVERY_S
        time.sleep(0.2)

def time_one(obj: Objective, x: np.ndarray, label: str, repeats: int) -> float:
    tlist = []
    for r in range(repeats):
        stop_evt = threading.Event()
        t0 = time.time()
        thr = threading.Thread(target=_heartbeat, args=(t0, f"{label} (rep {r+1}/{repeats})", stop_evt), daemon=True)
        thr.start()
        J, m = obj.evaluate(x)
        stop_evt.set()
        thr.join(timeout=1.0)
        tlist.append(time.time() - t0)
        print(f"[done] {label} (rep {r+1}/{repeats}): J={J:.6g}, elapsed={tlist[-1]:.3f}s; metrics={m}")
    return sum(tlist)/len(tlist)

def main():
    with open(SNAPSHOT_FILE, "rb") as f:
        snap = pickle.load(f)
    smb     = snap["station"]
    active  = snap["active"]
    dt      = float(snap.get("dt", getattr(smb, "dt", 0.05)))
    si      = float(active["SI"])
    steps_per_SI   = int(round(si / dt))
    steps_per_eval = steps_per_SI * HORIZON_SI

    # Config banner
    print("[config] HORIZON_SI =", HORIZON_SI)
    print("[config] dt =", dt, "| SI =", si, "| steps/SI =", steps_per_SI, "| steps/eval ≈", steps_per_eval)

    obj = Objective(
        horizon_si=HORIZON_SI,
        weights=(1.0, 1.0, 1.0, 1.0),
        bounds={"D": (80.0,260.0), "E": (10.0,60.0), "Q4": (0.5,10.0), "SI": (1200.0,2600.0)}
    )
    obj.set_seed(smb, active)
    obj.prepare_t0()

    x_active = np.array([active["D"], active["E"], active["Q4"], active["SI"]], dtype=float)
    x_pert   = x_active + PERTURB

    # Baseline timing
    avg_base = time_one(obj, x_active, "baseline", REPEATS_BASE)

    # Perturbed timing
    avg_pert = time_one(obj, x_pert, "perturbed", REPEATS_PERT)

    print(f"[timing] baseline avg:  {avg_base:.3f} s/eval")
    print(f"[timing] perturbed avg: {avg_pert:.3f} s/eval")
    s_per_eval = max(avg_base, avg_pert)
    print(f"[timing] suggested NM budget for ~5 evals: {int(s_per_eval*5)}–{int(s_per_eval*6)} s")

if __name__ == "__main__":
    main()
