# smb_mhe.py
# -----------------------------------------------------------------------------
# Fixed-lag MHE (via Kalman smoothing) for the terminal column (zones 1 & 3)
# - Single source of truth: the engine (DT) advances physics.
# - At each observer tick we:
#     * append inlet and outlet to a sliding window
#     * run a linear-Gaussian MHE (equivalent to KF+RTS smoother) on that window
#     * write the smoothed current state back into the terminal column profile
#
# Model (within a zone, between switches):
#   x_{k+1} = F x_k + g_man * u_man,k + g_gal * u_gal,k + w_k
#   y_k     = H x_k + v_k
# with Q = sigma^2 I + alpha DᵀD  (per component), R = r_meas * I_2.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import numpy as np
import sys, time
from collections import deque
from scipy.linalg import lu_solve, block_diag

from smb_engine import SMBTwinEngine
from smb_sensor import SMBOutletSensor


# ---------- small numerics helpers ----------

def _first_diff_tridiag(n: int) -> np.ndarray:
    """T = DᵀD for first-difference (Neumann-like). Tri-diagonal SPD."""
    T = np.zeros((n, n), float)
    if n == 1:
        T[0, 0] = 1.0
        return T
    np.fill_diagonal(T, 2.0)
    T[0, 0] = 1.0
    T[-1, -1] = 1.0
    i = np.arange(n - 1)
    T[i, i + 1] = -1.0
    T[i + 1, i] = -1.0
    return T


def _chol_inv_2x2(M: np.ndarray) -> np.ndarray:
    """Stable inverse for tiny 2x2 SPD matrices."""
    try:
        L = np.linalg.cholesky(M)
        Linv = np.linalg.inv(L)
        return Linv.T @ Linv
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def _joseph(P: np.ndarray, K: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    I = np.eye(P.shape[0])
    IKH = I - K @ H
    return IKH @ P @ IKH.T + K @ R @ K.T


# ---------- core per-zone MHE ----------

class ColumnTailMHE:
    """
    Fixed-lag MHE for one terminal LinColumn (zone 1 or 3), solved via KF+RTS.

    State x ∈ R^{2*Nx} (Man[0..Nx-1], Gal[0..Nx-1])
    Measurement y ∈ R^2 (outlet Man/Gal = last cell of each component)
    Inputs u_k = (u_man, u_gal) = upstream tube last-node concentrations (scalars)
    """

    def __init__(
        self,
        smb_obj: object,
        zone: int,
        Nx: int,
        sensor: SMBOutletSensor,
        # Window / cadence
        horizon_steps: int = 50,        # fixed-lag window length (L)
        # Design weights
        q_sigma2: float = 2e-8,         # σ² in Q (per step)
        q_alpha: float = 5e-5,          # α on DᵀD in Q  (sets spatial spread)
        r_meas: float = 1e-4,           # (g/L)^2
        p0_sigma2: float = 1e-5,        # arrival P0 = p0_sigma2*I + p0_alpha*DᵀD
        p0_alpha: float = 5e-4,
        deadband_gpl: float = 0.05,     # ignore tiny |y - Hx| (sensor DL)
        # Write-back safety
        alpha_injection: float = 0.8,   # blend factor for applying smoothed delta
        max_cell_step_frac: float = 0.25,  # per-tick per-cell trust region
    ):
        assert zone in (1, 3), "zone must be 1 or 3"
        self.smb = smb_obj
        self.zone = int(zone)
        self.Nx = int(Nx)
        self.sensor = sensor

        # Measurement model: last cell of each component
        self.H = np.zeros((2, 2 * self.Nx))
        self.H[0, self.Nx - 1] = 1.0
        self.H[1, 2 * self.Nx - 1] = 1.0
        self.R = np.eye(2) * float(r_meas)
        self.deadband = float(deadband_gpl)

        # Process weights Q, arrival P0 (block-diag per component)
        T = _first_diff_tridiag(self.Nx)
        Qblk = float(q_sigma2) * np.eye(self.Nx) + float(q_alpha) * T
        P0blk = float(p0_sigma2) * np.eye(self.Nx) + float(p0_alpha) * T
        self.Q = block_diag(Qblk, Qblk)
        self.P0 = block_diag(P0blk, P0blk)

        # Safety on write-back
        self.alpha = float(np.clip(alpha_injection, 0.0, 1.0))
        self.max_cell_step_frac = float(max_cell_step_frac)

        # Bindings / model matrices
        self._tube_up = None
        self._col = None
        self._last_col_id = None
        self._F = None
        self._g_man = None
        self._g_gal = None

        # Sliding window buffers (u: inputs to propagate x_k→x_{k+1})
        L = int(horizon_steps)
        self.L = max(2, L)
        self._y_buf: deque[np.ndarray] = deque(maxlen=self.L)       # each shape (2,)
        self._u_buf: deque[Tuple[float, float]] = deque(maxlen=self.L)  # length L; last item unused in dyn
        self._x0: Optional[np.ndarray] = None    # arrival mean
        self._P0: Optional[np.ndarray] = None    # arrival covariance

        self._last_warn = 0.0
        self._bind_and_reset(force=True)

    # ----- binding & model matrices -----

    def _warn_once(self, msg: str):
        now = time.time()
        if now - getattr(self, "_last_warn", 0.0) > 2.0:
            print(f"[MHE z{self.zone}] {msg}", file=sys.stderr)
            self._last_warn = now

    def _pair_tube_and_column(self):
        try:
            objs: List[object] = self.smb.zones[self.zone]
            if len(objs) < 2:
                return None, None
            return objs[-2], objs[-1]
        except Exception as ex:
            self._warn_once(f"zone access failed: {ex}")
            return None, None

    def _read_engine_profile(self) -> np.ndarray:
        Nx = self.Nx
        try:
            m = np.asarray(self._col.components[0].c, float)
            g = np.asarray(self._col.components[1].c, float)
            if m.size != Nx or g.size != Nx:
                m = np.resize(m, Nx); g = np.resize(g, Nx)
        except Exception:
            m = np.zeros(Nx); g = np.zeros(Nx)
        return np.concatenate([m, g])

    def _write_engine_profile(self, x: np.ndarray):
        Nx = self.Nx
        try:
            self._col.components[0].c[:] = x[:Nx]
            self._col.components[1].c[:] = x[Nx:]
        except Exception:
            pass

    def _bind_and_reset(self, force: bool = False):
        prev_id = self._last_col_id
        tube, col = self._pair_tube_and_column()
        if col is None:
            return
        cid = id(col)
        switched = (prev_id is not None and cid != prev_id) or force
        self._last_col_id = cid
        self._tube_up, self._col = tube, col
        if switched:
            # Rebuild F and inlet responses g for new terminal column
            self._build_F_and_g()
            # Reset window with arrival taken from current engine profile
            self._y_buf.clear()
            self._u_buf.clear()
            self._x0 = self._read_engine_profile()
            self._P0 = self.P0.copy()

    def _build_F_and_g(self):
        """F ≈ A⁻¹B (block-diag) and g vectors (per component)."""
        Nx = self.Nx
        I = np.eye(Nx)
        # Man
        try:
            c0 = self._col.components[0]
            F0 = lu_solve(c0._lu, c0._B) if hasattr(c0, "_lu") and hasattr(c0, "_B") else I
            e0 = np.zeros(Nx); e0[0] = float(getattr(c0, "_inlet_coef", 0.0))
            g0 = lu_solve(c0._lu, e0) if hasattr(c0, "_lu") else np.zeros(Nx)
        except Exception:
            F0, g0 = I, np.zeros(Nx)
        # Gal
        try:
            c1 = self._col.components[1]
            F1 = lu_solve(c1._lu, c1._B) if hasattr(c1, "_lu") and hasattr(c1, "_B") else I
            e1 = np.zeros(Nx); e1[0] = float(getattr(c1, "_inlet_coef", 0.0))
            g1 = lu_solve(c1._lu, e1) if hasattr(c1, "_lu") else np.zeros(Nx)
        except Exception:
            F1, g1 = I, np.zeros(Nx)

        self._F = block_diag(F0, F1)
        self._g_man = g0
        self._g_gal = g1

    # ----- data capture -----

    def _read_inlet_pair(self) -> Tuple[float, float]:
        try:
            man_in = float(self._tube_up.components[0].c[-1])
            gal_in = float(self._tube_up.components[1].c[-1])
            return man_in, gal_in
        except Exception:
            return 0.0, 0.0

    def _read_measurement(self) -> Optional[np.ndarray]:
        try:
            y = self.sensor.read_zone(self.zone)
            if y is None:
                return None
            return np.array([float(y[0]), float(y[1])], float)
        except Exception as ex:
            self._warn_once(f"sensor read failed: {ex}")
            return None

    # ----- MHE solver (KF + RTS smoother over window) -----

    def _solve_window(self) -> Optional[np.ndarray]:
        """
        Run KF forward pass over window (arrival x0,P0), then RTS backward pass.
        Returns smoothed state at the LAST time of the window (x_hat_Lminus1).
        """
        if self._F is None or self._x0 is None or len(self._y_buf) == 0:
            return None

        F = self._F
        H = self.H
        Q = self.Q
        R = self.R

        Nx2 = 2 * self.Nx
        Lw = len(self._y_buf)            # window length actually available (<= L)
        y_seq = list(self._y_buf)
        u_seq = list(self._u_buf)

        # Arrival
        x_plus = self._x0.copy()
        P_plus = self._P0.copy()

        # Storage for RTS
        X_plus = [None] * Lw
        P_plus_list = [None] * Lw
        X_pred = [None] * Lw
        P_pred_list = [None] * Lw

        # Forward KF
        for k in range(Lw):
            # predict (except k=0 we predict from arrival; uses u_{k-1} if k>0)
            if k == 0:
                # measurement directly on arrival x0
                X_pred[k] = x_plus
                P_pred_list[k] = P_plus
            else:
                u_man, u_gal = u_seq[k - 1]  # input that drives k-1 -> k
                G_u = np.concatenate([self._g_man * u_man, self._g_gal * u_gal])
                X_pred[k] = F @ x_plus + G_u
                P_pred_list[k] = F @ P_plus @ F.T + Q
                P_pred_list[k] += np.eye(Nx2) * 1e-12

            # update with y_k (deadband gating)
            yk = y_seq[k]
            Hx = H @ X_pred[k]
            innov = yk - Hx
            R_eff = R.copy()
            BIG = 1e12
            for i in (0, 1):
                if abs(innov[i]) < self.deadband:
                    R_eff[i, i] = BIG
                    innov[i] = 0.0

            S = H @ P_pred_list[k] @ H.T + R_eff
            Sinv = _chol_inv_2x2(S)
            K = P_pred_list[k] @ H.T @ Sinv

            x_plus = X_pred[k] + K @ innov
            x_plus = np.maximum(x_plus, 0.0)  # physical clamp
            P_plus = _joseph(P_pred_list[k], K, H, R_eff)
            P_plus += np.eye(Nx2) * 1e-12

            X_plus[k] = x_plus
            P_plus_list[k] = P_plus

        # Backward RTS to get smoothed last state
        x_s = X_plus[-1].copy()
        P_s = P_plus_list[-1].copy()

        for k in range(Lw - 2, -1, -1):
            # Predict k -> k+1 (to build smoother gain)
            u_man, u_gal = u_seq[k]
            G_u = np.concatenate([self._g_man * u_man, self._g_gal * u_gal])
            x_pred_k1 = F @ X_plus[k] + G_u
            P_pred_k1 = F @ P_plus_list[k] @ F.T + Q
            P_pred_k1 += np.eye(Nx2) * 1e-12

            # Smoother gain
            try:
                Ck = P_plus_list[k] @ F.T @ np.linalg.inv(P_pred_k1)
            except np.linalg.LinAlgError:
                Ck = P_plus_list[k] @ F.T @ np.linalg.pinv(P_pred_k1)

            # RTS update
            x_s = X_plus[k] + Ck @ (x_s - x_pred_k1)
            P_s = P_plus_list[k] + Ck @ (P_s - P_pred_k1) @ Ck.T

        # x_s here is the smoothed estimate at time 0; we want LAST time (already stored)
        x_smoothed_last = X_plus[-1]  # (already equals RTS result at last index)
        return x_smoothed_last

    # ----- main step -----

    def step_once(self, sim_t: float) -> None:
        # Rebind (handles rotation & resets)
        self._bind_and_reset(force=False)
        if self._col is None or self._F is None:
            return

        # Capture y_k and u_k (u_k will drive k -> k+1)
        y = self._read_measurement()
        if y is None:
            return
        u = self._read_inlet_pair()

        # Append to buffers; if this is the FIRST item after reset, also set arrival x0,P0
        if len(self._y_buf) == 0:
            self._x0 = self._read_engine_profile()
            self._P0 = self.P0.copy()

        self._y_buf.append(y)
        self._u_buf.append(u)

        # Not enough data yet?
        if len(self._y_buf) < 2:
            return  # need at least arrival + one step

        # Solve window (KF+RTS)
        x_hat_last = self._solve_window()
        if x_hat_last is None:
            return

        # Write-back with trust region and blending
        x_now = self._read_engine_profile()
        delta = x_hat_last - x_now

        eps = 1e-9
        limit = self.max_cell_step_frac * (np.abs(x_now) + eps)
        delta = np.clip(delta, -limit, limit)

        x_new = np.maximum(x_now + self.alpha * delta, 0.0)
        self._write_engine_profile(x_new)

        # Slide the window by one step: set new arrival (x0,P0) to the filtered/smoothed prior
        # We reuse current engine profile for x0 to keep continuity.
        self._x0 = x_new.copy()
        self._P0 = self.P0.copy()  # simple arrival; could carry last P_s if desired
        # Keep buffers bounded automatically (deque maxlen)


# ---------- manager (two zones, orchestrator-friendly) ----------

class SMBMHEManager:
    """
    Hosts two ColumnTailMHEs (zone 1 = Extract, zone 3 = Raffinate) and
    runs them every mhe_dt_multiples * dt of simulated time.
    """
    def __init__(
        self,
        engine: SMBTwinEngine,
        sensor: SMBOutletSensor,
        Nx: int = 100,
        mhe_dt_multiples: int = 5,      # solve every 5*dt by default
        horizon_steps: int = 50,        # fixed-lag window (≈ desired spatial spread in cells)
        # weights (map to your KLO tuning intuition)
        q_sigma2: float = 2e-8,
        q_alpha: float = 5e-5,
        r_meas: float = 1e-4,
        p0_sigma2: float = 1e-5,
        p0_alpha: float = 5e-4,
        deadband_gpl: float = 0.05,
        # write-back safety
        alpha_injection: float = 0.8,
        max_cell_step_frac: float = 0.25,
    ):
        self.engine = engine
        self.sensor = sensor
        self.Nx = int(Nx)
        self._dt_mult = int(mhe_dt_multiples)
        self._next_sim_t: Optional[float] = None
        self._bound = False

        self.mhe_ex: Optional[ColumnTailMHE] = None
        self.mhe_ra: Optional[ColumnTailMHE] = None

        self._cfg = dict(
            horizon_steps=horizon_steps,
            q_sigma2=q_sigma2, q_alpha=q_alpha, r_meas=r_meas,
            p0_sigma2=p0_sigma2, p0_alpha=p0_alpha,
            deadband_gpl=deadband_gpl,
            alpha_injection=alpha_injection, max_cell_step_frac=max_cell_step_frac,
        )

        self.engine.subscribe(self._on_engine_update)

    def _ensure_bound(self):
        if self._bound:
            return
        smb = getattr(self.engine, "_smb", None)
        if smb is None:
            return
        self.mhe_ex = ColumnTailMHE(smb, zone=1, Nx=self.Nx, sensor=self.sensor, **self._cfg)
        self.mhe_ra = ColumnTailMHE(smb, zone=3, Nx=self.Nx, sensor=self.sensor, **self._cfg)
        dt = float(getattr(smb, "dt", 0.05)); sim_t = float(getattr(smb, "timer", 0.0))
        self._next_sim_t = sim_t + self._dt_mult * dt
        self._bound = True
        print(f"[MHE] initialized (every {self._dt_mult}*dt)")

    def _on_engine_update(self, _res, sim_t: float):
        self._ensure_bound()
        if not self._bound or self._next_sim_t is None or sim_t < self._next_sim_t:
            return

        try: self.mhe_ex.step_once(sim_t)
        except Exception as ex: print(f"[MHE] z1 error: {ex}", file=sys.stderr)
        try: self.mhe_ra.step_once(sim_t)
        except Exception as ex: print(f"[MHE] z3 error: {ex}", file=sys.stderr)

        smb = getattr(self.engine, "_smb", None)
        dt = float(getattr(smb, "dt", 0.05))
        self._next_sim_t = sim_t + self._dt_mult * dt

    # Optional hooks (for debugging / logging)
    def get_tail_state(self, zone: int) -> np.ndarray:
        self._ensure_bound()
        mhe = self.mhe_ex if zone == 1 else self.mhe_ra
        if mhe is None or mhe._col is None:
            return np.zeros(2 * self.Nx)
        x = mhe._read_engine_profile()
        return x

    def get_outlet_estimate(self, zone: int) -> Tuple[float, float]:
        self._ensure_bound()
        mhe = self.mhe_ex if zone == 1 else self.mhe_ra
        if mhe is None or mhe._col is None:
            return (0.0, 0.0)
        Nx = self.Nx
        c0 = float(mhe._col.components[0].c[Nx - 1])
        c1 = float(mhe._col.components[1].c[Nx - 1])
        return (c0, c1)

    def snapshot(self) -> Dict[str, object]:
        return {
            "Nx": self.Nx,
            "z1_outlet_hat": self.get_outlet_estimate(1),
            "z3_outlet_hat": self.get_outlet_estimate(3),
        }
