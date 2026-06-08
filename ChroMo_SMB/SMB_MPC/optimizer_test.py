#!/usr/bin/env python3
"""
optimizer_test.py — Offline SMB MPC benchmark (Q1,Q2,Q4,SI version)

Fixes vs your last file:
- OPC branch now reads ActiveQ1/ActiveQ2/ActiveQ4
- Objective uses horizon_si=HORIZON_SI (not hard-coded 8)
- Eval print shows Q1,Q2,Q4,SI + bound flags (no leftover D/E)
- Snapshot prints Q1,Q2 instead of D/E
"""

from __future__ import annotations
import os, time, pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

# ---- Project imports ---------------------------------------------------------
from mpc_objective import Objective
from mpc_optimizer import NMRunner, NMConfig

# Prefer local smb_engine helpers; fall back to package path if needed.
from ChroMo_SMB.SMB_SE.smb_engine import _build_station as build_station, ComponentSpec
from ChroMo_SMB.SMB_PlantSim.opcua_client import SMB_OPCUAClient


# =============================================================================
# CONFIG
# =============================================================================

MODE: str = "auto"          # "snapshot" | "run" | "auto"
SNAPSHOT_FILE: str = "warm_state.pkl"

# --- Snapshot (plant) settings ---
FROM_OPC: bool = False
OPC_ENDPOINT: str = "opc.tcp://127.0.0.1:4840"
OPC_OBJECT: str = "SMB_Simulation"

DT: float = 5.0         # base physics for the saved snapshot (keep physical)
NX: int = 100
SI: float = 1539.0
F: float = 21.0  # fixed parameter
Q1: float = 208.0
Q2: float = 135.0
Q4: float = 15.0
WARM_SI: int = 0       # warm-up length in SIs

# --- Optimizer settings ---
HORIZON_SI: int = 32    # default horizon for testing (two+ cycles)
BOUNDS: Dict[str, Tuple[float, float]] = {
    "Q1": (10.0, 400.0),
    "Q2": (10.0, 400.0),
    "Q4": (10.0, 400.0),
    "SI": (100.0, 10800.0),
}
# (w_dil_ex, w_dil_ra, w_pur_ex, w_pur_ra)
WEIGHTS: Tuple[float, float, float, float] = (2.5, 0.5, 5.0, 1.0)
TIME_BUDGET_S: float = 7200.0

NM_CFG = NMConfig(
    maxiter=400,
    xatol=1e-2,
    fatol=1e-2,
    simplex_rel_size=0.1,
    warm_start_blend=0.0,
    improve_tol=0.0001,
)


# =============================================================================
# INTERNALS
# =============================================================================

def _default_components():
    return [
        ComponentSpec(name="Man", feed_concentration=9.0, henry_constant=4.55, delta=54.0, Di=0.0007),
        ComponentSpec(name="Gal", feed_concentration=6.0, henry_constant=2.77, delta=84.0, Di=0.0007),
    ]

def _steps_for_si(si: float, dt: float, n_si: int) -> int:
    # allow zero warm-up; negative treated as zero
    n_si = max(0, int(n_si))
    steps_per_si = int(round(max(1.0, si) / max(dt, 1e-12)))
    return steps_per_si * n_si


@dataclass
class ProgressTap:
    t0: float
    best_J: Optional[float] = None
    n_eval: int = 0


class TeeObjective:
    """
    Wrap Objective to:
      • count evaluations
      • print ONE summary line per evaluation incl. x and bound hits
      • show KPIs (Pu/Di)
    """
    def __init__(self, base: Objective, tap: ProgressTap) -> None:
        self._base = base
        self._tap = tap
        self._nm_hook = None
        self._bounds = dict(getattr(base, "bounds", {}))

        # NEW: attributes for duplicate filtering
        self._round = (3, 3, 3, 1)   # decimal precision for Q1,Q2,Q4,SI
        self._last_key = None
        self._last_J = None

    def _clip_with_flags(self, x: np.ndarray):
        names = ["Q1", "Q2", "Q4", "SI"]
        lb = np.array([self._bounds[n][0] for n in names], dtype=float)
        ub = np.array([self._bounds[n][1] for n in names], dtype=float)
        xc = np.minimum(np.maximum(x, lb), ub)
        flags = []
        for i, n in enumerate(names):
            if xc[i] <= lb[i] + 1e-12:
                flags.append("lo")
            elif xc[i] >= ub[i] - 1e-12:
                flags.append("hi")
            else:
                flags.append("-")
        return xc, flags

    def set_seed(self, station_copy, active_setpoints: Dict[str, float]) -> None:
        self._base.set_seed(station_copy, active_setpoints)

    def prepare_t0(self) -> None:
        self._base.prepare_t0()

    def evaluate(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        xc, flags = self._clip_with_flags(x)
        J, metrics = self._base.evaluate(xc)

        key = (round(float(xc[0]), self._round[0]),
               round(float(xc[1]), self._round[1]),
               round(float(xc[2]), self._round[2]),
               round(float(xc[3]), self._round[3]))
        if self._last_key == key and self._last_J is not None and abs(J - self._last_J) <= 1e-12:
            # Return silently; don't count or print the duplicate
            return J, metrics
        self._last_key, self._last_J = key, J

        self._tap.n_eval += 1

        improved = False
        dJ = None
        if self._tap.best_J is None or J < self._tap.best_J:
            dJ = (self._tap.best_J - J) if self._tap.best_J is not None else None
            self._tap.best_J = float(J)
            improved = True

        if improved:
            Vex = metrics.get("vol_ex"); Vra = metrics.get("vol_ra")
            cAex = metrics.get("cA_ex_bar"); cBra = metrics.get("cB_ra_bar")
            print(f"         [detail] V_ex={Vex:.3g}, V_ra={Vra:.3g}, cA_ex_bar={cAex:.3g}, cB_ra_bar={cBra:.3g}")

        elapsed = time.time() - self._tap.t0
        Q1, Q2, Q4, SI = xc.tolist()
        b1, b2, bQ, bSI = flags

        Pu_ex = metrics.get("pur_ex", float("nan"))
        Pu_ra = metrics.get("pur_ra", float("nan"))
        Di_ex = metrics.get("dil_ex", float("nan"))
        Di_ra = metrics.get("dil_ra", float("nan"))

        msg = (f"[eval {self._tap.n_eval}] t={elapsed:.1f}s, "
               f"J={J:.6g}"
               f"  x: Q1={Q1:.3g}{'!'+b1 if b1!='-' else ''},"
               f" Q2={Q2:.3g}{'!'+b2 if b2!='-' else ''},"
               f" Q4={Q4:.3g}{'!'+bQ if bQ!='-' else ''},"
               f" SI={SI:.3g}{'!'+bSI if bSI!='-' else ''}"
               f"  KPIs: Pu_ex={Pu_ex:.3f}%, Pu_ra={Pu_ra:.3f}%,"
               f" Di_ex={Di_ex:.3f}%, Di_ra={Di_ra:.3f}%")
        if improved and dJ is not None:
            msg += f"  (ΔJ={dJ:.6g} NEW BEST)"
        print(msg)

        if self._nm_hook is not None:
            try:
                self._nm_hook({"x": xc, "J": J, "metrics": metrics, "is_best": improved})
            except Exception:
                pass
        return J, metrics

    def set_eval_hook(self, fn) -> None:
        self._nm_hook = fn
        try:
            self._base.set_eval_hook(fn)
        except Exception:
            pass

    def get_progress(self):
        try:
            return self._base.get_progress()
        except Exception:
            return {"n_eval": self._tap.n_eval, "best_J": self._tap.best_J}


def make_snapshot(outfile: str) -> None:
    """Build SMBStation, warm it, and pickle the station + ACTIVE."""
    if FROM_OPC:
        if SMB_OPCUAClient is None:
            raise RuntimeError("python-opcua not available; set FROM_OPC=False or install opcua")
        cli = SMB_OPCUAClient(endpoint=OPC_ENDPOINT, obj_browse_name=OPC_OBJECT)
        cli.connect()
        snap = cli.read_snapshot()
        cli.disconnect()
        F0  = float(snap.get("ActiveFeedFlow", F))
        Q10 = float(snap.get("ActiveQ1", Q1))
        Q20 = float(snap.get("ActiveQ2", Q2))
        Q40 = float(snap.get("ActiveQ4", Q4))
        SI0 = float(snap.get("ActiveSwitchInterval", SI))
        print(f"[snapshot] using ACTIVE from OPC  F={F0}, Q1={Q10}, Q2={Q20}, Q4={Q40}, SI={SI0}")
    else:
        F0, Q10, Q20, Q40, SI0 = F, Q1, Q2, Q4, SI
        print(f"[snapshot] using CONFIG  F={F0}, Q1={Q10}, Q2={Q20}, Q4={Q40}, SI={SI0}")

    # New scheme: control Q1, Q2, Q4; derive Q3 = Q2 + F
    zone_flows = [Q10, Q20, Q20 + F0, Q40]

    t0 = time.time()
    smb = build_station(
        dt=DT, Nx=NX, switch_interval=SI0,
        col_length=320.0, col_diameter=10.0, porosity=0.376, dead_volume=0.5,
        components=_default_components(), zone_flows=zone_flows,
    )

    n_steps = _steps_for_si(SI0, DT, WARM_SI)
    if n_steps > 0:
        print(f"[snapshot] warming for {WARM_SI} SI → {n_steps} steps (dt={DT}) ...")
        smb.step(n_steps)
    else:
        print("[snapshot] no warm-up (WARM_SI=0); starting from t=0")
    t1 = time.time()
    print(f"[snapshot] warm-up took {t1 - t0:.3f} s")

    payload = {
        "station": smb,
        "active": {"F": F0, "Q1": Q10, "Q2": Q20, "Q4": Q40, "SI": SI0},
        "dt": float(DT),
    }
    with open(outfile, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[snapshot] saved to {outfile}")


def run_optimizer(infile: str) -> None:
    """Load snapshot and run NM (uniform scoring; no heartbeat)."""
    with open(infile, "rb") as f:
        snap = pickle.load(f)

    smb = snap["station"]
    active = snap["active"]

    base_obj = Objective(
        horizon_si=HORIZON_SI,               # <- use the configured horizon
        weights=WEIGHTS,                     # (w_dil_ex, w_dil_ra, w_pur_ex, w_pur_ra)
        bounds=BOUNDS,
        feed_conc_A=9.0,                     # Mannose in feed
        feed_conc_B=6.0,                     # Galactose in feed
        # burn_in_si=0, score_si=None, taper=False  # defaults (uniform scoring)
    )

    # Prepare t0
    base_obj.set_seed(smb, active)
    base_obj.prepare_t0()

    x_active = np.array([active["Q1"], active["Q2"], active["Q4"], active["SI"]], dtype=float)

    # ---- config banner ----
    print("[config] horizon_si =", HORIZON_SI)
    print("[config] weights    =", {"w_dil_ex": WEIGHTS[0], "w_dil_ra": WEIGHTS[1], "w_pur_ex": WEIGHTS[2], "w_pur_ra": WEIGHTS[3]})
    print("[config] bounds     =", BOUNDS)
    print("[config] x0 (Q1,Q2,Q4,SI) =", tuple(float(v) for v in x_active))
    print("[config] NM         =", {
        "maxiter": NM_CFG.maxiter,
        "xatol": NM_CFG.xatol,
        "fatol": NM_CFG.fatol,
        "simplex_rel": NM_CFG.simplex_rel_size,
        "improve_tol": getattr(NM_CFG, "improve_tol", 0.0),
    })
    print("[config] time_budget_s =", TIME_BUDGET_S)

    # Baseline
    J0, m0 = base_obj.evaluate(x_active)
    print(f"[baseline] J0={J0:.6g}  metrics={m0}")

    # One-line summaries per eval
    tap = ProgressTap(t0=time.time(), best_J=J0, n_eval=0)
    obj = TeeObjective(base_obj, tap)

    runner = NMRunner()
    wall_t0 = time.time()
    res = runner.solve(x0=x_active, objective=obj, time_budget_s=TIME_BUDGET_S, cfg=NM_CFG)
    wall_t1 = time.time()

    print(f"[result] status={res.status} msg={res.message}")
    print(f"[result] elapsed={res.elapsed_s:.3f}s (wall {wall_t1 - wall_t0:.3f}s), n_iter={res.n_iter}, n_eval={res.n_eval}")
    if res.best_x is not None:
        bx = np.asarray(res.best_x, dtype=float)
        print(f"[result] best_x  Q1={bx[0]:.6g}, Q2={bx[1]:.6g}, Q4={bx[2]:.6g}, SI={bx[3]:.6g}")
    if res.best_J is not None:
        print(f"[result] best_J={res.best_J:.6g}  ΔJ={J0 - res.best_J:.6g}")
    if res.best_metrics:
        print(f"[result] metrics={res.best_metrics}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    mode = MODE.lower().strip()
    if mode not in ("snapshot", "run", "auto"):
        raise SystemExit(f"Invalid MODE='{MODE}'. Use 'snapshot' | 'run' | 'auto'.")

    if mode == "snapshot":
        make_snapshot(SNAPSHOT_FILE)
    elif mode == "run":
        if not os.path.exists(SNAPSHOT_FILE):
            raise SystemExit(f"Snapshot '{SNAPSHOT_FILE}' not found. Set MODE='snapshot' first.")
        run_optimizer(SNAPSHOT_FILE)
    else:  # auto
        if not os.path.exists(SNAPSHOT_FILE):
            print(f"[auto] snapshot '{SNAPSHOT_FILE}' not found → creating it...")
            make_snapshot(SNAPSHOT_FILE)
        print(f"[auto] running optimizer on '{SNAPSHOT_FILE}' ...")
        run_optimizer(SNAPSHOT_FILE)
