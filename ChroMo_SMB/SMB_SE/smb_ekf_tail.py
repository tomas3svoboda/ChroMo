# smb_ekf_tail.py — EKF/DT co-simulation: correct inlet forcing + switch handoff
# (keeps API: SMBTailKFManager)

from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import numpy as np
import sys, time
from scipy.linalg import lu_solve, block_diag

from smb_engine import SMBTwinEngine
from smb_sensor import SMBOutletSensor

# ---------- numerics helpers ----------
def joseph_update(P, K, H, R):
    I = np.eye(P.shape[0])
    IKH = I - K @ H
    return IKH @ P @ IKH.T + K @ R @ K.T

def first_diff_tridiag(n: int) -> np.ndarray:
    T = np.zeros((n, n), float)
    if n == 1:
        T[0, 0] = 1.0; return T
    np.fill_diagonal(T, 2.0)
    T[0, 0] = 1.0; T[-1, -1] = 1.0
    i = np.arange(n - 1)
    T[i, i + 1] = -1.0; T[i + 1, i] = -1.0
    return T

# ---------- core EKF over terminal column ----------
class ColumnTailEKF:
    """
    State x = [Man[0..Nx-1], Gal[0..Nx-1]]
    Measurement y = [Man_out, Gal_out] from terminal column last cell.
    """
    def __init__(
        self,
        smb_obj: object,
        zone: int,
        Nx: int,
        sensor: SMBOutletSensor,
        # process/measurement
        q_sigma2_per_dt: float = 1e-7,
        q_alpha_smooth: float = 5e-6,
        r_meas: float = 1e-4,
        deadband_gpl: float = 0.05,
        # prior + switch policy
        p0_sigma2: float = 1e-4,
        p0_alpha: float = 1e-3,
        write_back_on_switch: bool = True,   # NEW: push posterior into DT at rotation
    ):
        assert zone in (1, 3)
        self.smb = smb_obj
        self.zone = int(zone)
        self.Nx = int(Nx)
        self.sensor = sensor

        self._dt = float(getattr(self.smb, "dt", 0.05)) or 0.05

        # H picks terminal cell of each component
        self.H = np.zeros((2, 2 * self.Nx))
        self.H[0, self.Nx - 1] = 1.0
        self.H[1, 2 * self.Nx - 1] = 1.0
        self.R_base = np.eye(2) * float(r_meas)
        self.deadband = float(deadband_gpl)

        # Q and P0 (smoothness-penalized SPD, block-diag per component)
        T = first_diff_tridiag(self.Nx)
        self.Q_block = float(q_sigma2_per_dt) * np.eye(self.Nx) + float(q_alpha_smooth) * T
        self.P0_block = float(p0_sigma2) * np.eye(self.Nx) + float(p0_alpha) * T
        self.Q_base  = block_diag(self.Q_block, self.Q_block)
        self.P0_base = block_diag(self.P0_block, self.P0_block)

        self.x = np.zeros(2 * self.Nx)      # posterior
        self.P = self.P0_base.copy()        # posterior covariance

        self._col = None                    # bound terminal column object
        self._last_col_id = None
        self._last_sim_t: Optional[float] = None
        self._last_warn = 0.0

        self.write_back_on_switch = bool(write_back_on_switch)

        self._bind_terminal(force=True)

    # ----- binding & DT access -----
    def _warn_once(self, msg: str):
        now = time.time()
        if now - getattr(self, "_last_warn", 0.0) > 2.0:
            print(f"[EKF z{self.zone}] {msg}", file=sys.stderr)
            self._last_warn = now

    def _find_terminal_column(self):
        """Return (tube_before, terminal_column). In this project the zone list is [..., tube, column]."""
        try:
            objs: List[object] = self.smb.zones[self.zone]
            if len(objs) < 2:
                return None, None
            tube, col = objs[-2], objs[-1]       # tube upstream, then column (terminal)
            return tube, col
        except Exception as ex:
            self._warn_once(f"bind failed: {ex}")
            return None, None

    def _bind_terminal(self, force: bool = False):
        prev_col = self._col
        tube, col = self._find_terminal_column()
        if col is None:
            return
        col_id = id(col)
        switched = (self._last_col_id is not None and col_id != self._last_col_id) or force
        self._last_col_id = col_id
        self._col = col
        self._tube_up = tube

        if switched:
            # NEW: write back the *old* posterior into the old column so DT inherits it after rotate
            if self.write_back_on_switch and prev_col is not None:
                try:
                    Nx = self.Nx
                    prev_col.components[0].c[:] = self.x[:Nx]
                    prev_col.components[1].c[:] = self.x[Nx:]
                except Exception as ex:
                    self._warn_once(f"write-back on switch failed: {ex}")
            # reseed mean/cov from the NEW terminal column
            self.x = self._read_column_profile()
            self.P = self.P0_base.copy()
            self._last_sim_t = float(getattr(self.smb, "timer", 0.0))

    def _read_column_profile(self) -> np.ndarray:
        Nx = self.Nx
        try:
            man = np.asarray(self._col.components[0].c, float)
            gal = np.asarray(self._col.components[1].c, float)
            if man.size != Nx or gal.size != Nx:
                man = np.resize(man, Nx); gal = np.resize(gal, Nx)
        except Exception:
            man = np.zeros(Nx); gal = np.zeros(Nx)
        return np.concatenate([man, gal])

    def _read_inlet_pair(self) -> Tuple[float, float]:
        """Get inlet concentrations (Man, Gal) from the upstream tube’s last node."""
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

    # ----- CN transition & affine inlet forcing -----
    def _F_blocks(self):
        """Return (F0,F1) per component using cached LU/B; fall back to I."""
        Nx = self.Nx; I = np.eye(Nx)
        try:
            c0 = self._col.components[0]; c1 = self._col.components[1]
            F0 = lu_solve(c0._lu, c0._B) if hasattr(c0, "_lu") and hasattr(c0, "_B") else I
            F1 = lu_solve(c1._lu, c1._B) if hasattr(c1, "_lu") and hasattr(c1, "_B") else I
        except Exception:
            F0, F1 = I, I
        return F0, F1

    def _predict_mean_with_inlet(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        x^- = F x^+ + A^{-1} e0 * (_inlet_coef * cin)  (per component); returns (x_pred, F_blockdiag)
        Matches LinColumn.step() where comp._b[0] += comp._inlet_coef * cin.  """
        Nx = self.Nx
        F0, F1 = self._F_blocks()
        man_in, gal_in = self._read_inlet_pair()

        # Block predictions
        x0 = x[:Nx]; x1 = x[Nx:]
        x0_pred = F0 @ x0
        x1_pred = F1 @ x1

        # Affine inlet terms via same LU solve as the plant does
        try:
            e0 = np.zeros(Nx); e0[0] = float(self._col.components[0]._inlet_coef) * man_in
            u0 = lu_solve(self._col.components[0]._lu, e0)
        except Exception:
            u0 = np.zeros(Nx)
        try:
            e1 = np.zeros(Nx); e1[0] = float(self._col.components[1]._inlet_coef) * gal_in
            u1 = lu_solve(self._col.components[1]._lu, e1)
        except Exception:
            u1 = np.zeros(Nx)

        x_pred = np.concatenate([x0_pred + u0, x1_pred + u1])
        F = block_diag(F0, F1)
        return x_pred, F

    # ----- one EKF tick -----
    def step_once(self, sim_t: float) -> None:
        self._dt = float(getattr(self.smb, "dt", self._dt)) or self._dt
        self._bind_terminal(force=False)
        if self._col is None:
            return

        if self._last_sim_t is None:
            self._last_sim_t = float(sim_t)
        dts_steps = max(0.0, (float(sim_t) - self._last_sim_t) / max(self._dt, 1e-12))
        q_scale = max(1.0, dts_steps)

        # Predict (mean with inlet forcing; covariance with F)
        x_pred, F = self._predict_mean_with_inlet(self.x)
        P_pred = F @ self.P @ F.T + q_scale * self.Q_base
        P_pred += np.eye(P_pred.shape[0]) * 1e-12

        # Update
        y = self._read_measurement()
        if y is not None:
            innov = y - (self.H @ x_pred)
            R_eff = self.R_base.copy()
            BIG = 1e12
            for i in (0, 1):
                if abs(innov[i]) < self.deadband:
                    R_eff[i, i] = BIG
                    innov[i] = 0.0

            S = self.H @ P_pred @ self.H.T + R_eff
            try:
                L = np.linalg.cholesky(S)
                Sinv = np.linalg.inv(L).T @ np.linalg.inv(L)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)

            K = P_pred @ self.H.T @ Sinv
            x_post = np.maximum(x_pred + K @ innov, 0.0)    # nonnegativity projection
            P_post = joseph_update(P_pred, K, self.H, R_eff)
            P_post += np.eye(P_post.shape[0]) * 1e-12
        else:
            x_post, P_post = x_pred, P_pred

        self.x, self.P = x_post, P_post
        self._last_sim_t = float(sim_t)

    # hooks
    def outlet_estimate(self) -> Tuple[float, float]:
        y_hat = self.H @ self.x
        return float(y_hat[0]), float(y_hat[1])
    def state(self) -> np.ndarray: return self.x.copy()
    def covariance_diag(self) -> np.ndarray: return np.diag(self.P).copy()

# ---------- manager (API compatible) ----------
class SMBTailEKFManager:
    def __init__(
        self,
        engine: SMBTwinEngine,
        sensor: SMBOutletSensor,
        Nx: int = 100,
        kf_dt_multiples: int = 50,
        q_sigma2_per_dt: float = 1e-7,
        q_alpha_smooth: float = 5e-6,
        r_meas: float = 1e-4,
        deadband_gpl: float = 0.05,
        p0_sigma2: float = 1e-4,
        p0_alpha: float = 1e-3,
        write_back_on_switch: bool = True,   # propagate estimate into DT at rotation
    ):
        self.engine = engine
        self.sensor = sensor
        self.Nx = int(Nx)
        self._kf_dt_mult = int(kf_dt_multiples)
        self._next_sim_t: Optional[float] = None
        self.kf_ex: Optional[ColumnTailEKF] = None
        self.kf_ra: Optional[ColumnTailEKF] = None
        self._bound = False

        self.engine.subscribe(self._on_engine_update)
        self._cfg = dict(
            q_sigma2_per_dt=q_sigma2_per_dt,
            q_alpha_smooth=q_alpha_smooth,
            r_meas=r_meas,
            deadband_gpl=deadband_gpl,
            p0_sigma2=p0_sigma2,
            p0_alpha=p0_alpha,
            write_back_on_switch=write_back_on_switch,
        )

    def _ensure_bound(self):
        if self._bound: return
        smb = getattr(self.engine, "_smb", None)
        if smb is None: return
        self.kf_ex = ColumnTailEKF(smb, zone=1, Nx=self.Nx, sensor=self.sensor, **self._cfg)
        self.kf_ra = ColumnTailEKF(smb, zone=3, Nx=self.Nx, sensor=self.sensor, **self._cfg)
        dt = float(getattr(smb, "dt", 0.05)); sim_t = float(getattr(smb, "timer", 0.0))
        self._next_sim_t = sim_t + self._kf_dt_mult * dt
        self._bound = True
        print(f"[EKF] initialized (every {self._kf_dt_mult}*dt)")

    def _on_engine_update(self, _res, sim_t: float):
        self._ensure_bound()
        if not self._bound or self._next_sim_t is None or sim_t < self._next_sim_t:
            return
        try: self.kf_ex.step_once(sim_t)
        except Exception as ex: print(f"[EKF] z1 error: {ex}", file=sys.stderr)
        try: self.kf_ra.step_once(sim_t)
        except Exception as ex: print(f"[EKF] z3 error: {ex}", file=sys.stderr)

        smb = getattr(self.engine, "_smb", None)
        dt = float(getattr(smb, "dt", 0.05))
        self._next_sim_t = sim_t + self._kf_dt_mult * dt

    # hooks for orchestrator
    def get_tail_state(self, zone: int) -> np.ndarray:
        self._ensure_bound(); kf = self.kf_ex if zone == 1 else self.kf_ra
        return kf.state() if kf is not None else np.zeros(2 * self.Nx)
    def get_outlet_estimate(self, zone: int) -> Tuple[float, float]:
        self._ensure_bound(); kf = self.kf_ex if zone == 1 else self.kf_ra
        return kf.outlet_estimate() if kf is not None else (0.0, 0.0)
    def get_covariance_diag(self, zone: int) -> np.ndarray:
        self._ensure_bound(); kf = self.kf_ex if zone == 1 else self.kf_ra
        return kf.covariance_diag() if kf is not None else np.ones(2 * self.Nx)

# Back-compat shim
class SMBTailKFManager(SMBTailEKFManager):
    def __init__(
        self,
        engine: SMBTwinEngine,
        sensor: SMBOutletSensor,
        Nx: int = 100,
        kf_dt_multiples: int = 100,
        q_proc_per_dt: float = 1e-6,
        r_meas: float = 1e-4,
        deadband_gpl: float = 0.05,
        p0_reset: float = 1e-3,
        reset_P_on_switch: bool = True,
        q_alpha_smooth: float = 5e-6,
        p0_alpha: float = 1e-3,
        write_back_on_switch: bool = True,
    ):
        super().__init__(
            engine=engine, sensor=sensor, Nx=Nx,
            kf_dt_multiples=kf_dt_multiples,
            q_sigma2_per_dt=q_proc_per_dt,
            q_alpha_smooth=q_alpha_smooth,
            r_meas=r_meas,
            deadband_gpl=deadband_gpl,
            p0_sigma2=p0_reset,
            p0_alpha=p0_alpha,
            write_back_on_switch=write_back_on_switch,
        )
