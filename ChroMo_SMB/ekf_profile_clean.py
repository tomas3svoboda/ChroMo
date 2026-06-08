"""
Clean, rigorous EKF for SMB chromatography — with concise debug logging.

Key points
----------
- Theory-consistent: predict → gain → state update → covariance (Joseph form).
- Measurement models:
    • "ideal" (default): direct outlet pick, h(x) = S x.
    • "lag": first-order sensor dynamics per channel (time constants tau).
- Robust numerics: Cholesky solve for S^{-1}, Joseph form for P.
- Seamless integration: same public API used by GUI/MPC.

Public API
----------
- class SMBProfileEstimator(threading.Thread):
    .start() / .stop()
    .snapshot_for_mpc()                → deepCopy of SMB
    .snapshot_for_optimizer()          → (deepCopy, None)
    .bias()                            → None (kept for GUI compatibility)
    .resnorm()                         → last innovation/residual norm

Configuration knobs
-------------------
- ekf_period_s: EKF update period (s)
- R: 4-vector or 4×4 measurement covariance
- Q_profile: "flat" | "outlet_weighted" | array-like (per-node process noise)
- measurement_model: "ideal" (default) | "lag"
- tau: per-channel tau for lag model (s)
- qz: process noise on 4 sensor states (lag model)
- debug: True/False → logging verbosity
"""
from __future__ import annotations

import time
import threading
from typing import Tuple, Dict, Any, Optional, Sequence
import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_solve
import logging

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
logger = logging.getLogger("ekf")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[EKF] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# EKF core (pure math — no mode logic here)
# ----------------------------------------------------------------------------
class EKFCore:
    """Minimal EKF core with Joseph covariance update and Cholesky gain solve."""
    def __init__(self,
                 n_state: int,
                 *,
                 R: np.ndarray | Sequence[float],
                 Q: np.ndarray | float | Sequence[float],
                 init_var: float = 1e-2):
        self.n = int(n_state)
        self.x = np.zeros(self.n, float)
        self.P = np.eye(self.n) * float(init_var)

        # Q
        if np.isscalar(Q):
            self.Q = np.eye(self.n) * float(Q)
        else:
            q = np.asarray(Q, float).reshape(-1)
            assert q.size == self.n, f"Q length {q.size} != n_state {self.n}"
            self.Q = np.diag(q)

        # R (4×4)
        R = np.asarray(R, float)
        self.R = np.diag(R) if R.ndim == 1 else R.copy()
        assert self.R.shape == (4, 4), f"R must be 4×4, got {self.R.shape}"

    def set_state(self, x0: np.ndarray, P0_scale: float = 1.0):
        x0 = np.asarray(x0, float).reshape(-1)
        assert x0.size == self.n
        self.x[:] = x0
        if P0_scale != 1.0:
            self.P *= float(P0_scale)

    def predict(self, dt: float, A: np.ndarray | None = None):
        dt = float(dt)
        if A is None:
            # conservative random-walk model
            self.P = self.P + self.Q * dt
        else:
            # proper covariance propagation: P ← A P Aᵀ + Q Δt
            self.P = A @ self.P @ A.T + self.Q * dt

    def update(self, y_meas: np.ndarray, y_pred: np.ndarray, H: np.ndarray) -> Dict[str, float]:
        y_meas = np.asarray(y_meas, float).reshape(-1)
        y_pred = np.asarray(y_pred, float).reshape(-1)
        H = np.asarray(H, float)

        assert y_meas.shape == (4,), f"y_meas must be (4,), got {y_meas.shape}"
        assert y_pred.shape == (4,), f"y_pred must be (4,), got {y_pred.shape}"
        assert H.shape == (4, self.n), f"H must be (4, {self.n}), got {H.shape}"

        r = y_meas - y_pred                   # innovation (4,)
        S = H @ self.P @ H.T + self.R         # innovation covariance (4×4)

        # K = P H^T S^{-1}  → compute via Cholesky solve without explicit inverse
        cF = cho_factor(S, check_finite=False)
        # Solve S * X = (H P)  → X is 4×n, then K = X^T is n×4
        X = cho_solve(cF, H @ self.P, check_finite=False)
        K = X.T

        dx = K @ r
        self.x = np.maximum(self.x + dx, 0.0)  # non-negativity clamp (profiles)

        # Joseph form for P to preserve PSD
        I = np.eye(self.n)
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)     # enforce symmetry

        return {
            "r_norm": float(np.linalg.norm(r)),
            "dx_norm": float(np.linalg.norm(dx)),
            "trace_P": float(np.trace(self.P)),
        }

# ----------------------------------------------------------------------------
# Adapters to the SMB codebase (selection matrix, state IO)
# ----------------------------------------------------------------------------
class AdapterHooks:
    def outlet_y_pred(self, smb: Any) -> np.ndarray:  # ideal outlet from model
        raise NotImplementedError
    def get_state_vector(self, smb: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError
    def set_state_vector(self, smb: Any, x: np.ndarray, meta: Dict[str, Any]) -> None:
        raise NotImplementedError
    def build_S(self, smb: Any, meta: Dict[str, Any]) -> np.ndarray:  # selection rows
        raise NotImplementedError
    def permute_on_switch(self, smb: Any, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, P

class LinColumnAdapters(AdapterHooks):
    """State layout: x_prof = [Z1:c0 (Nx1), Z1:c1 (Nx1), Z3:c0 (Nx3), Z3:c1 (Nx3)].
    Measurements: [Ex.M, Ex.G, Ra.M, Ra.G] = last nodes.
    """
    def __init__(self, zone1_last_col_getter, zone3_last_col_getter):
        self._get_z1_last = zone1_last_col_getter
        self._get_z3_last = zone3_last_col_getter
        self._last_switch_state: Optional[int] = None

    @staticmethod
    def _Nx(col) -> int:
        return int(getattr(col, "Nx", len(col.components[0].c)))

    def outlet_y_pred(self, smb: Any) -> np.ndarray:
        z1 = self._get_z1_last(smb); z3 = self._get_z3_last(smb)
        return np.array([
            float(z1.components[0].c[-1]),
            float(z1.components[1].c[-1]),
            float(z3.components[0].c[-1]),
            float(z3.components[1].c[-1]),
        ], float)

    def get_state_vector(self, smb: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        z1 = self._get_z1_last(smb); z3 = self._get_z3_last(smb)
        Nx1 = self._Nx(z1); Nx3 = self._Nx(z3)
        x = np.concatenate([
            z1.components[0].c, z1.components[1].c,
            z3.components[0].c, z3.components[1].c,
        ]).astype(float)
        meta = {"Nx1": Nx1, "Nx3": Nx3, "n_prof": 2*Nx1 + 2*Nx3,
                "switch_state": getattr(smb, "switchState", None)}
        self._last_switch_state = meta["switch_state"]
        return x, meta

    def set_state_vector(self, smb: Any, x: np.ndarray, meta: Dict[str, Any]) -> None:
        Nx1, Nx3 = meta["Nx1"], meta["Nx3"]
        x = np.asarray(x, float)
        z1 = self._get_z1_last(smb); z3 = self._get_z3_last(smb)
        z1.components[0].c[:] = np.maximum(x[0:Nx1], 0.0)
        z1.components[1].c[:] = np.maximum(x[Nx1:2*Nx1], 0.0)
        z3.components[0].c[:] = np.maximum(x[2*Nx1:2*Nx1+Nx3], 0.0)
        z3.components[1].c[:] = np.maximum(x[2*Nx1+Nx3:2*Nx1+2*Nx3], 0.0)

    def build_S(self, smb: Any, meta: Dict[str, Any]) -> np.ndarray:
        Nx1, Nx3 = meta["Nx1"], meta["Nx3"]
        S = np.zeros((4, 2*Nx1 + 2*Nx3), float)
        S[0, Nx1-1]               = 1.0
        S[1, 2*Nx1-1]             = 1.0
        S[2, 2*Nx1 + Nx3 - 1]     = 1.0
        S[3, 2*Nx1 + 2*Nx3 - 1]   = 1.0
        return S

    def permute_on_switch(self, smb: Any, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cur_sw = getattr(smb, "switchState", None)
        if self._last_switch_state is None:
            self._last_switch_state = cur_sw
            return x, P
        if cur_sw != self._last_switch_state:
            x_new, _ = self.get_state_vector(smb)
            self._last_switch_state = cur_sw
            v = float(np.mean(np.diag(P))) if P.size else 1e-2
            P_new = np.eye(x_new.size) * max(1e-3, v)
            logger.info("Topology switch detected → resetting P")
            return x_new, P_new
        return x, P

    def build_A_map(self, smb: Any, meta: Dict[str, Any]) -> np.ndarray:
        """
        Build the state transition matrix A_map = A^{-1} B for the EKF predict step
        over the profile state x = [Z1:c0, Z1:c1, Z3:c0, Z3:c1].

        Uses per-component LU factors (comp._lu) and B (comp._B) precomputed in LinColumn.
        """
        Nx1, Nx3 = meta["Nx1"], meta["Nx3"]
        n = 2*Nx1 + 2*Nx3
        Amap = np.zeros((n, n), float)

        # helper to fetch last columns in zones 1 and 3
        z1_last = self._get_z1_last(smb)
        z3_last = self._get_z3_last(smb)

        # For each block (component & zone), solve A X = B  → X = A^{-1} B
        # 1) Zone 1, comp 0
        comp = z1_last.components[0]
        AinvB = lu_solve(comp._lu, comp._B)              # shape Nx1×Nx1
        Amap[0:Nx1, 0:Nx1] = AinvB

        # 2) Zone 1, comp 1
        comp = z1_last.components[1]
        AinvB = lu_solve(comp._lu, comp._B)
        Amap[Nx1:2*Nx1, Nx1:2*Nx1] = AinvB

        # 3) Zone 3, comp 0
        comp = z3_last.components[0]
        AinvB = lu_solve(comp._lu, comp._B)
        Amap[2*Nx1:2*Nx1+Nx3, 2*Nx1:2*Nx1+Nx3] = AinvB

        # 4) Zone 3, comp 1
        comp = z3_last.components[1]
        AinvB = lu_solve(comp._lu, comp._B)
        Amap[2*Nx1+Nx3:2*Nx1+2*Nx3, 2*Nx1+Nx3:2*Nx1+2*Nx3] = AinvB

        return Amap


# ----------------------------------------------------------------------------
# Real-time EKF thread
# ----------------------------------------------------------------------------
class SMBProfileEstimator(threading.Thread):
    def __init__(self,
                 smb_template: Any,
                 opc_client: Any,
                 adapters: AdapterHooks,
                 *,
                 dt_model: Optional[float] = None,
                 ekf_period_s: float = 15.0,
                 R: Sequence[float] | np.ndarray = (2e-4, 2e-4, 4e-4, 4e-4),
                 Q_profile: str | float | Sequence[float] = "outlet_weighted",
                 q_flat: float = 1e-6,
                 q_lo: float = 5e-7,
                 q_hi: float = 2e-6,
                 init_var: float = 1e-2,
                 measurement_model: str = "ideal",      # default ideal for simulator
                 tau: Sequence[float] = (20.0, 20.0, 25.0, 25.0),
                 qz: float = 1e-6,
                 debug: bool = False):
        super().__init__(daemon=True)
        if debug:
            logger.setLevel(logging.DEBUG)

        self._opc = opc_client
        self._dt = float(dt_model if dt_model is not None else smb_template.settings.get("dt", 0.5))
        self._period = float(ekf_period_s)
        self._ad = adapters
        self._running = threading.Event(); self._running.set()
        self._last_res: Optional[float] = None

        self._mm = str(measurement_model).lower()      # "ideal" | "lag"
        self._tau = np.array(tau, float).reshape(4)
        self._alpha_period = np.exp(-self._period / self._tau)

        # Working copy of the plant
        self._smb = smb_template.deepCopy() if hasattr(smb_template, "deepCopy") else smb_template

        # Initial state (profile) and selection matrix
        x_prof0, self._meta = self._ad.get_state_vector(self._smb)
        n_prof = self._meta["n_prof"]
        S = self._ad.build_S(self._smb, self._meta)
        y_star0 = S @ x_prof0

        # Build Q (profile) and optionally augment with sensor process noise
        q_vec_prof = self._make_q_vec(n_prof, Q_profile, q_flat, q_lo, q_hi, self._meta)
        if self._mm == "lag":
            x0 = np.concatenate([x_prof0, y_star0])             # [x_prof ; z]
            q_vec = np.concatenate([q_vec_prof, np.ones(4) * float(qz)])
        elif self._mm == "ideal":
            x0 = x_prof0
            q_vec = q_vec_prof
        else:
            raise ValueError("measurement_model must be 'ideal' or 'lag'")

        self._n_prof = n_prof
        self._z_slice = slice(n_prof, n_prof + 4) if self._mm == "lag" else slice(n_prof, n_prof)

        self._ekf = EKFCore(n_state=x0.size, R=R, Q=q_vec, init_var=init_var)
        self._ekf.set_state(x0)

        logger.info(f"EKF init: model_dt={self._dt:.3f}s, period={self._period:.3f}s, "
                    f"n_prof={n_prof}, mode={self._mm}")

    # ---- helpers ----
    def _make_q_vec(self, n: int, mode: str | float | Sequence[float],
                    q_flat: float, q_lo: float, q_hi: float,
                    meta: Dict[str, Any]) -> np.ndarray:
        if isinstance(mode, (int, float)):
            return np.ones(n, float) * float(mode)
        if isinstance(mode, (list, tuple, np.ndarray)):
            v = np.asarray(mode, float).reshape(-1)
            assert v.size == n
            return v
        m = str(mode).lower()
        if m == "flat":
            return np.ones(n, float) * float(q_flat)
        if m == "outlet_weighted":
            Nx1, Nx3 = meta["Nx1"], meta["Nx3"]
            def ramp(lo, hi, N): return np.linspace(lo, hi, N)
            return np.r_[
                ramp(q_lo, q_hi, Nx1),  # Z1 c0
                ramp(q_lo, q_hi, Nx1),  # Z1 c1
                ramp(q_lo, q_hi, Nx3),  # Z3 c0
                ramp(q_lo, q_hi, Nx3),  # Z3 c1
            ]
        raise ValueError("Unknown Q_profile mode")

    def bias(self): return None
    def resnorm(self): return self._last_res

    def _read_measurement(self) -> Optional[np.ndarray]:
        try:
            s = self._opc.read_snapshot()
            return np.array([
                float(s["ExtractConcentration_Man"]),
                float(s["ExtractConcentration_Gal"]),
                float(s["RaffinateConcentration_Man"]),
                float(s["RaffinateConcentration_Gal"]),
            ], float)
        except Exception:
            logger.warning("Measurement read failed.")
            return None

    # ----------------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------------
    def run(self):
        last_update = time.time()
        while self._running.is_set():
            # 1) Advance simulator by one step (keeps plant evolving)
            if hasattr(self._smb, "step_fast_outlets"):
                self._smb.step_fast_outlets()
            else:
                self._smb.step(1)

            # 2) For 'lag' model, propagate sensor lag EVERY model dt (prevents residual drift)
            if self._mm == "lag" and self._ekf.x.size >= self._n_prof + 4:
                y_star_dt = self._ad.outlet_y_pred(self._smb)            # ideal outlets at this dt
                alpha_dt  = np.exp(-self._dt / self._tau)                # per-channel (4,)
                z         = self._ekf.x[self._z_slice]                   # (4,)
                self._ekf.x[self._z_slice] = alpha_dt * z + (1.0 - alpha_dt) * y_star_dt

            # 3) EKF update at coarse period
            now = time.time()
            if (now - last_update) >= self._period:

                # Build the state-transition map from LinColumn
                A_prof = self._ad.build_A_map(self._smb, self._meta)   # shape (n_prof × n_prof)

                # If lag sensors are active, augment A with the z-dynamics:
                #   z_{k+1} = α z_k + (1-α) S x_prof  (discrete over ekf_period_s)
                if self._mm == "lag":
                    S = self._ad.build_S(self._smb, self._meta)        # (4 × n_prof)
                    alpha = np.exp(-self._period / self._tau).reshape(4)
                    A = np.zeros((self._ekf.n, self._ekf.n), float)     # (n_prof+4) × (n_prof+4)
                    # top-left: profile dynamics
                    A[:self._n_prof, :self._n_prof] = A_prof
                    # bottom-left: coupling from profile to sensors
                    A[self._n_prof:, :self._n_prof] = (1.0 - alpha)[:, None] * S
                    # bottom-right: sensor self-dynamics
                    A[self._n_prof:, self._n_prof:] = np.diag(alpha)
                else:
                    A = A_prof

                # Predict covariance with A
                self._ekf.predict(self._period, A)

                # Handle topology change (re-extract state, reset P moderately)
                self._ekf.x, self._ekf.P = self._ad.permute_on_switch(self._smb, self._ekf.x, self._ekf.P)

                # If lag-mode and sensor states lost at switch, re-augment [x_prof ; z]
                if self._mm == "lag" and self._ekf.x.size == self._n_prof:
                    y_star_switch = self._ad.outlet_y_pred(self._smb)  # (4,)
                    x_aug = np.concatenate([self._ekf.x, y_star_switch])
                    P_old = self._ekf.P
                    P_new = np.zeros((self._n_prof + 4, self._n_prof + 4), float)
                    P_new[:self._n_prof, :self._n_prof] = P_old
                    z_var = float(np.mean(np.diag(P_old))) if P_old.size else 1e-3
                    P_new[self._n_prof:, self._n_prof:] = np.eye(4) * z_var
                    self._ekf.x, self._ekf.P = x_aug, P_new

                # Build selection and ideal outlet from current model
                S = self._ad.build_S(self._smb, self._meta)
                y_star = self._ad.outlet_y_pred(self._smb)

                # Build H and y_pred per mode
                if self._mm == "ideal":
                    H = np.zeros((4, self._ekf.n), float)
                    H[:, :self._n_prof] = S
                    y_pred = y_star
                else:  # lag
                    H = np.zeros((4, self._ekf.n), float)
                    H[:, self._n_prof:self._n_prof+4] = np.eye(4)
                    # y_pred is the current sensor state already propagated each dt
                    assert self._ekf.x.size >= self._n_prof + 4, "Sensor states missing; re-augment after switch."
                    y_pred = self._ekf.x[self._z_slice]

                # Measurement and EKF update
                y_meas = self._read_measurement()
                if y_meas is not None:
                    stats = self._ekf.update(y_meas, y_pred, H)
                    self._last_res = stats["r_norm"]
                    # write corrected profiles back into the model (only profile slice)
                    self._ad.set_state_vector(self._smb, self._ekf.x[:self._n_prof], self._meta)
                    logger.info(f"Dt={self._period:.1f}s | ||r||={stats['r_norm']:.3e} | "
                                f"||dx||={stats['dx_norm']:.3e} | tr(P)={stats['trace_P']:.3e}")
                last_update = now

            time.sleep(max(0.0, self._dt * 0.5))

    def stop(self):
        self._running.clear()

    # Snapshots for MPC/optimizer
    def snapshot_for_mpc(self) -> Any:
        return self._smb.deepCopy() if hasattr(self._smb, "deepCopy") else self._smb
    def snapshot_for_optimizer(self):
        smb_copy = self._smb.deepCopy() if hasattr(self._smb, "deepCopy") else self._smb
        return smb_copy, None
