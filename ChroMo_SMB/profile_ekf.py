"""
Profile‑Aware Extended Kalman Filter for SMB (ideal detector; no delay model).

This version is wired to your current codebase:
- **SMBStation** zones lists (last object is the column): `smb.zones[1][-1]`, `smb.zones[3][-1]`.
- **LinColumn** keeps numeric caches on each component as `col.components[i]._lu` and `._B`,
  and the spatial profile in `col.components[i].c` (numpy 1D array of length `Nx`).
- Measurements are the *ideal* detector readings (last node of the outlet columns).

It runs a coarse EKF update every `ekf_period_s` (e.g., 10–20 s) while the simulator
advances each `dt` (e.g., 0.5 s). The EKF corrects the **full outlet profiles** of the
last column in Zone 1 (→ Extract) and Zone 3 (→ Raffinate) for both components.

GUI compatibility:
- Exposes `.bias()` → returns `None` (no bias state here) and `.resnorm()` → last residual norm
  so your existing labels keep working.
- The MPC seed uses `snapshot_for_mpc()` just like your previous estimator hook.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    from scipy.linalg import lu_solve
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# EKF core
# ---------------------------------------------------------------------------

class ProfileAwareEKF:
    """Lightweight EKF over a large state x (concatenated outlet-column profiles).

    The fine-grained plant propagation is handled by the simulator; between EKF updates
    we treat the process as a random walk and only inflate P by Q·Δt.
    """
    def __init__(self, n_state: int, *, r_meas: float = 1e-4, q_proc: float = 1e-6, init_var: float = 1e-2):
        self.n = int(n_state)
        self.x = np.zeros(self.n, float)
        self.P = np.eye(self.n) * float(init_var)
        self.Q = np.eye(self.n) * float(q_proc)
        self.R = np.eye(4) * float(r_meas)  # 4 outlets: [Ex.M, Ex.G, Ra.M, Ra.G]

    def set_state(self, x0: np.ndarray, P0_scale: float = 1.0):
        x0 = np.asarray(x0, float).reshape(-1)
        assert x0.size == self.n
        self.x[:] = x0
        if P0_scale != 1.0:
            self.P *= float(P0_scale)

    def predict(self, dt_ekf: float):
        self.P = self.P + self.Q * float(dt_ekf)

    def update(self, y_meas: np.ndarray, y_pred: np.ndarray, H: np.ndarray) -> np.ndarray:
        y_meas = np.asarray(y_meas, float).reshape(-1)
        y_pred = np.asarray(y_pred, float).reshape(-1)
        H = np.asarray(H, float)
        assert y_meas.size == 4 and y_pred.size == 4
        assert H.shape == (4, self.n)

        innov = y_meas - y_pred  # r = y - h(x^-)
        S = H @ self.P @ H.T + self.R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S + 1e-9*np.eye(4))

        dx = K @ innov
        self.x = np.maximum(self.x + dx, 0.0)  # clamp to ≥0
        self.P = (np.eye(self.n) - K @ H) @ self.P
        return dx

# ---------------------------------------------------------------------------
# Adapter API → wired to your SMBStation + LinColumn
# ---------------------------------------------------------------------------

class AdapterHooks:
    def outlet_y_pred(self, smb: Any) -> np.ndarray:
        raise NotImplementedError

    def get_state_vector(self, smb: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    def set_state_vector(self, smb: Any, x: np.ndarray, meta: Dict[str, Any]) -> None:
        raise NotImplementedError

    def build_H(self, smb: Any, meta: Dict[str, Any], m_steps: int) -> np.ndarray:
        raise NotImplementedError

    def permute_on_switch(self, smb: Any, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, P

@dataclass
class _ColRef:
    col: Any
    comp_idx: int
    Nx: int

class LinColumnAdapters(AdapterHooks):
    """Adapter for your project.

    • Zone 1 last column → Extract, Zone 3 last column → Raffinate.
    • Component data live on `col.components[i]` with fields: `c`, `_lu`, `_B`.
    • State layout x = [Z1:c0 (Nx1), Z1:c1 (Nx1), Z3:c0 (Nx3), Z3:c1 (Nx3)].
    """
    def __init__(self, zone1_last_col_getter, zone3_last_col_getter):
        self._get_z1_last = zone1_last_col_getter
        self._get_z3_last = zone3_last_col_getter
        self._last_switch_state: Optional[int] = None

    # --- helpers ---
    @staticmethod
    def _last_node(col, comp_idx: int) -> float:
        return float(col.components[comp_idx].c[-1])

    @staticmethod
    def _Nx(col) -> int:
        return int(getattr(col, "Nx", len(col.components[0].c)))

    # --- AdapterHooks ---
    def outlet_y_pred(self, smb: Any) -> np.ndarray:
        z1_last = self._get_z1_last(smb)
        z3_last = self._get_z3_last(smb)
        return np.array([
            self._last_node(z1_last, 0),  # Ex.M
            self._last_node(z1_last, 1),  # Ex.G
            self._last_node(z3_last, 0),  # Ra.M
            self._last_node(z3_last, 1),  # Ra.G
        ], float)

    def get_state_vector(self, smb: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        z1_last = self._get_z1_last(smb)
        z3_last = self._get_z3_last(smb)
        Nx1 = self._Nx(z1_last); Nx3 = self._Nx(z3_last)
        x = np.concatenate([
            z1_last.components[0].c,
            z1_last.components[1].c,
            z3_last.components[0].c,
            z3_last.components[1].c,
        ]).astype(float)
        meta = {
            "z1": [_ColRef(z1_last, 0, Nx1), _ColRef(z1_last, 1, Nx1)],
            "z3": [_ColRef(z3_last, 0, Nx3), _ColRef(z3_last, 1, Nx3)],
            "offsets": {
                "z1c0": 0,
                "z1c1": Nx1,
                "z3c0": Nx1*2,
                "z3c1": Nx1*2 + Nx3,
                "n": Nx1*2 + Nx3*2,
            },
            "switch_state": getattr(smb, "switchState", None),
        }
        # remember last switch state for permutation detection
        self._last_switch_state = meta["switch_state"]
        return x, meta

    def set_state_vector(self, smb: Any, x: np.ndarray, meta: Dict[str, Any]) -> None:
        offs = meta["offsets"]; n = offs["n"]
        x = np.asarray(x, float).reshape(-1)
        assert x.size == n
        z1c0 = x[offs["z1c0"]: offs["z1c0"] + meta["z1"][0].Nx]
        z1c1 = x[offs["z1c1"]: offs["z1c1"] + meta["z1"][1].Nx]
        z3c0 = x[offs["z3c0"]: offs["z3c0"] + meta["z3"][0].Nx]
        z3c1 = x[offs["z3c1"]: offs["z3c1"] + meta["z3"][1].Nx]
        meta["z1"][0].col.components[0].c[:] = np.maximum(z1c0, 0.0)
        meta["z1"][1].col.components[1].c[:] = np.maximum(z1c1, 0.0)
        meta["z3"][0].col.components[0].c[:] = np.maximum(z3c0, 0.0)
        meta["z3"][1].col.components[1].c[:] = np.maximum(z3c1, 0.0)

    # ---- Row propagation for H without forming F ----
    @staticmethod
    def _row_propagate(col, comp_idx: int, m_steps: int) -> np.ndarray:
        Nx = int(getattr(col, "Nx", len(col.components[0].c)))
        r = np.zeros(Nx, float); r[-1] = 1.0  # select last node
        if m_steps <= 0:
            return r
        comp = col.components[comp_idx]
        if hasattr(comp, "_B") and (_HAVE_SCIPY and hasattr(comp, "_lu")):
            B = comp._B
            for _ in range(m_steps):
                # Aᵀ z = r → z; then rᵀ ← zᵀ B
                z = lu_solve(comp._lu, r, trans=1)
                r = (z @ B)
            return r
        if hasattr(comp, "_B") and hasattr(comp, "_A"):
            B = comp._B; AT = comp._A.T
            for _ in range(m_steps):
                z = np.linalg.solve(AT, r)
                r = (z @ B)
            return r
        raise AttributeError("Need comp._lu/_B or comp._A/_B for row propagation.")

    def build_H(self, smb: Any, meta: Dict[str, Any], m_steps: int) -> np.ndarray:
        z1_last = meta["z1"][0].col
        z3_last = meta["z3"][0].col
        Nx1 = meta["z1"][0].Nx; Nx3 = meta["z3"][0].Nx
        r_z1_c0 = self._row_propagate(z1_last, 0, m_steps)
        r_z1_c1 = self._row_propagate(z1_last, 1, m_steps)
        r_z3_c0 = self._row_propagate(z3_last, 0, m_steps)
        r_z3_c1 = self._row_propagate(z3_last, 1, m_steps)
        H = np.zeros((4, Nx1*2 + Nx3*2), float)
        H[0, 0:Nx1] = r_z1_c0
        H[1, Nx1:2*Nx1] = r_z1_c1
        H[2, 2*Nx1:2*Nx1+Nx3] = r_z3_c0
        H[3, 2*Nx1+Nx3:2*Nx1+2*Nx3] = r_z3_c1
        return H

    def permute_on_switch(self, smb: Any, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cur_sw = getattr(smb, "switchState", None)
        if self._last_switch_state is None:
            self._last_switch_state = cur_sw
            return x, P
        if cur_sw != self._last_switch_state:
            # Topology changed → rebuild mapping from model and reset covariance moderately
            x_new, meta_new = self.get_state_vector(smb)
            self._last_switch_state = cur_sw
            # reset P to a conservative diagonal around the previous mean variance
            v = float(np.mean(np.diag(P))) if P.size else 1e-2
            P_new = np.eye(x_new.size) * max(1e-3, v)
            return x_new, P_new
        return x, P

# ---------------------------------------------------------------------------
# Real‑time estimator thread
# ---------------------------------------------------------------------------

class SMBProfileEstimator(threading.Thread):
    """Runs the simulator and periodically applies profile‑aware EKF corrections.

    • Steps the working SMB copy by `dt_model`.
    • Every `ekf_period_s` seconds, builds H over the m fine steps since last update,
      compares ideal-detector outlets with plant measurements, and corrects the
      outlet profiles of Zone 1/Zone 3 last columns.
    • Provides `snapshot_for_mpc()` for warm‑starting the MPC.
    """
    def __init__(
        self,
        smb_template: Any,
        opc_client: Any,
        adapters: AdapterHooks,
        *,
        dt_model: Optional[float] = None,
        ekf_period_s: float = 15.0,
        r_meas: float = 1e-4,
        q_proc: float = 1e-6,
        init_var: float = 1e-2,
    ):
        super().__init__(daemon=True)
        self._opc = opc_client
        self._dt = float(dt_model if dt_model is not None else smb_template.settings.get("dt", 0.5))
        self._period = float(ekf_period_s)
        self._ad = adapters
        self._lock = threading.Lock()
        self._running = threading.Event(); self._running.set()
        self._last_res: Optional[float] = None

        # Working model copy
        self._smb = smb_template.deepCopy() if hasattr(smb_template, "deepCopy") else smb_template
        # Init EKF state from the current model
        x0, meta = self._ad.get_state_vector(self._smb)
        self._meta = meta
        self._ekf = ProfileAwareEKF(n_state=x0.size, r_meas=r_meas, q_proc=q_proc, init_var=init_var)
        self._ekf.set_state(x0)

    # --- optional GUI helpers ---
    def bias(self):
        return None
    def resnorm(self):
        return self._last_res

    # --- internal helpers ---
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
            return None

    def run(self):
        last_update = time.time()
        while self._running.is_set():
            # 1) Advance simulator by one dt and get predicted outlets
            if hasattr(self._smb, "step_fast_outlets"):
                y_pred = np.asarray(self._smb.step_fast_outlets(), float)
            else:
                self._smb.step(1)
                y_pred = self._ad.outlet_y_pred(self._smb)

            # 2) EKF update at coarse period
            now = time.time()
            if (now - last_update) >= self._period:
                m_steps = max(1, int(round(self._period / max(self._dt, 1e-9))))
                # handle topology change
                self._ekf.x, self._ekf.P = self._ad.permute_on_switch(self._smb, self._ekf.x, self._ekf.P)
                # predict covariance & build H
                self._ekf.predict(self._period)
                H = self._ad.build_H(self._smb, self._meta, m_steps)
                # measurement
                y_meas = self._read_measurement()
                if y_meas is not None:
                    self._ekf.update(y_meas, y_pred, H)
                    self._last_res = float(np.linalg.norm(y_meas - y_pred))
                    # write corrected profiles back into the model
                    self._ad.set_state_vector(self._smb, self._ekf.x, self._meta)
                last_update = now

            time.sleep(max(0.0, self._dt * 0.5))

    def stop(self):
        self._running.clear()

    # --- MPC handoff ---
    def snapshot_for_mpc(self) -> Any:
        with self._lock:
            return self._smb.deepCopy() if hasattr(self._smb, "deepCopy") else self._smb

    # compatibility with existing optimizer hook
    def snapshot_for_optimizer(self):
        """Return (smb_copy, bias) where bias=None to match old outlet-bias API."""
        with self._lock:
            smb_copy = self._smb.deepCopy() if hasattr(self._smb, "deepCopy") else self._smb
        return smb_copy, None

# ---------------------------------------------------------------------------
# Minimal usage sketch (names match your codebase)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def get_zone1_last_col(smb):
        # zones[1] is [Tube, Column, Tube, Column, ...]; last is Column
        return smb.zones[1][-1]
    def get_zone3_last_col(smb):
        return smb.zones[3][-1]

    adapters = LinColumnAdapters(get_zone1_last_col, get_zone3_last_col)
    # est = SMBProfileEstimator(smb_template, opc_client, adapters, dt_model=None, ekf_period_s=15.0)
    # est.start()
    pass
"""
Profile‑Aware Extended Kalman Filter for SMB (ideal detector; no delay model).

This version is wired to your current codebase:
- **SMBStation** zones lists (last object is the column): `smb.zones[1][-1]`, `smb.zones[3][-1]`.
- **LinColumn** keeps numeric caches on each component as `col.components[i]._lu` and `._B`,
  and the spatial profile in `col.components[i].c` (numpy 1D array of length `Nx`).
- Measurements are the *ideal* detector readings (last node of the outlet columns).

It runs a coarse EKF update every `ekf_period_s` (e.g., 10–20 s) while the simulator
advances each `dt` (e.g., 0.5 s). The EKF corrects the **full outlet profiles** of the
last column in Zone 1 (→ Extract) and Zone 3 (→ Raffinate) for both components.

GUI compatibility:
- Exposes `.bias()` → returns `None` (no bias state here) and `.resnorm()` → last residual norm
  so your existing labels keep working.
- The MPC seed uses `snapshot_for_mpc()` just like your previous estimator hook.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    from scipy.linalg import lu_solve
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# EKF core
# ---------------------------------------------------------------------------

class ProfileAwareEKF:
    """Lightweight EKF over a large state x (concatenated outlet-column profiles).

    The fine-grained plant propagation is handled by the simulator; between EKF updates
    we treat the process as a random walk and only inflate P by Q·Δt.
    """
    def __init__(self, n_state: int, *, r_meas: float = 1e-4, q_proc: float = 1e-6, init_var: float = 1e-2):
        self.n = int(n_state)
        self.x = np.zeros(self.n, float)
        self.P = np.eye(self.n) * float(init_var)
        self.Q = np.eye(self.n) * float(q_proc)
        self.R = np.eye(4) * float(r_meas)  # 4 outlets: [Ex.M, Ex.G, Ra.M, Ra.G]

    def set_state(self, x0: np.ndarray, P0_scale: float = 1.0):
        x0 = np.asarray(x0, float).reshape(-1)
        assert x0.size == self.n
        self.x[:] = x0
        if P0_scale != 1.0:
            self.P *= float(P0_scale)

    def predict(self, dt_ekf: float):
        self.P = self.P + self.Q * float(dt_ekf)

    def update(self, y_meas: np.ndarray, y_pred: np.ndarray, H: np.ndarray) -> np.ndarray:
        y_meas = np.asarray(y_meas, float).reshape(-1)
        y_pred = np.asarray(y_pred, float).reshape(-1)
        H = np.asarray(H, float)
        assert y_meas.size == 4 and y_pred.size == 4
        assert H.shape == (4, self.n)

        innov = y_meas - y_pred  # r = y - h(x^-)
        S = H @ self.P @ H.T + self.R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S + 1e-9*np.eye(4))

        dx = K @ innov
        self.x = np.maximum(self.x + dx, 0.0)  # clamp to ≥0
        self.P = (np.eye(self.n) - K @ H) @ self.P
        return dx

# ---------------------------------------------------------------------------
# Adapter API → wired to your SMBStation + LinColumn
# ---------------------------------------------------------------------------

class AdapterHooks:
    def outlet_y_pred(self, smb: Any) -> np.ndarray:
        raise NotImplementedError

    def get_state_vector(self, smb: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    def set_state_vector(self, smb: Any, x: np.ndarray, meta: Dict[str, Any]) -> None:
        raise NotImplementedError

    def build_H(self, smb: Any, meta: Dict[str, Any], m_steps: int) -> np.ndarray:
        raise NotImplementedError

    def permute_on_switch(self, smb: Any, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, P

@dataclass
class _ColRef:
    col: Any
    comp_idx: int
    Nx: int

class LinColumnAdapters(AdapterHooks):
    """Adapter for your project.

    • Zone 1 last column → Extract, Zone 3 last column → Raffinate.
    • Component data live on `col.components[i]` with fields: `c`, `_lu`, `_B`.
    • State layout x = [Z1:c0 (Nx1), Z1:c1 (Nx1), Z3:c0 (Nx3), Z3:c1 (Nx3)].
    """
    def __init__(self, zone1_last_col_getter, zone3_last_col_getter):
        self._get_z1_last = zone1_last_col_getter
        self._get_z3_last = zone3_last_col_getter
        self._last_switch_state: Optional[int] = None

    # --- helpers ---
    @staticmethod
    def _last_node(col, comp_idx: int) -> float:
        return float(col.components[comp_idx].c[-1])

    @staticmethod
    def _Nx(col) -> int:
        return int(getattr(col, "Nx", len(col.components[0].c)))

    # --- AdapterHooks ---
    def outlet_y_pred(self, smb: Any) -> np.ndarray:
        z1_last = self._get_z1_last(smb)
        z3_last = self._get_z3_last(smb)
        return np.array([
            self._last_node(z1_last, 0),  # Ex.M
            self._last_node(z1_last, 1),  # Ex.G
            self._last_node(z3_last, 0),  # Ra.M
            self._last_node(z3_last, 1),  # Ra.G
        ], float)

    def get_state_vector(self, smb: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        z1_last = self._get_z1_last(smb)
        z3_last = self._get_z3_last(smb)
        Nx1 = self._Nx(z1_last); Nx3 = self._Nx(z3_last)
        x = np.concatenate([
            z1_last.components[0].c,
            z1_last.components[1].c,
            z3_last.components[0].c,
            z3_last.components[1].c,
        ]).astype(float)
        meta = {
            "z1": [_ColRef(z1_last, 0, Nx1), _ColRef(z1_last, 1, Nx1)],
            "z3": [_ColRef(z3_last, 0, Nx3), _ColRef(z3_last, 1, Nx3)],
            "offsets": {
                "z1c0": 0,
                "z1c1": Nx1,
                "z3c0": Nx1*2,
                "z3c1": Nx1*2 + Nx3,
                "n": Nx1*2 + Nx3*2,
            },
            "switch_state": getattr(smb, "switchState", None),
        }
        # remember last switch state for permutation detection
        self._last_switch_state = meta["switch_state"]
        return x, meta

    def set_state_vector(self, smb: Any, x: np.ndarray, meta: Dict[str, Any]) -> None:
        offs = meta["offsets"]; n = offs["n"]
        x = np.asarray(x, float).reshape(-1)
        assert x.size == n
        z1c0 = x[offs["z1c0"]: offs["z1c0"] + meta["z1"][0].Nx]
        z1c1 = x[offs["z1c1"]: offs["z1c1"] + meta["z1"][1].Nx]
        z3c0 = x[offs["z3c0"]: offs["z3c0"] + meta["z3"][0].Nx]
        z3c1 = x[offs["z3c1"]: offs["z3c1"] + meta["z3"][1].Nx]
        meta["z1"][0].col.components[0].c[:] = np.maximum(z1c0, 0.0)
        meta["z1"][1].col.components[1].c[:] = np.maximum(z1c1, 0.0)
        meta["z3"][0].col.components[0].c[:] = np.maximum(z3c0, 0.0)
        meta["z3"][1].col.components[1].c[:] = np.maximum(z3c1, 0.0)

    # ---- Row propagation for H without forming F ----
    @staticmethod
    def _row_propagate(col, comp_idx: int, m_steps: int) -> np.ndarray:
        Nx = int(getattr(col, "Nx", len(col.components[0].c)))
        r = np.zeros(Nx, float); r[-1] = 1.0  # select last node
        if m_steps <= 0:
            return r
        comp = col.components[comp_idx]
        if hasattr(comp, "_B") and (_HAVE_SCIPY and hasattr(comp, "_lu")):
            B = comp._B
            for _ in range(m_steps):
                # Aᵀ z = r → z; then rᵀ ← zᵀ B
                z = lu_solve(comp._lu, r, trans=1)
                r = (z @ B)
            return r
        if hasattr(comp, "_B") and hasattr(comp, "_A"):
            B = comp._B; AT = comp._A.T
            for _ in range(m_steps):
                z = np.linalg.solve(AT, r)
                r = (z @ B)
            return r
        raise AttributeError("Need comp._lu/_B or comp._A/_B for row propagation.")

    def build_H(self, smb: Any, meta: Dict[str, Any], m_steps: int) -> np.ndarray:
        z1_last = meta["z1"][0].col
        z3_last = meta["z3"][0].col
        Nx1 = meta["z1"][0].Nx; Nx3 = meta["z3"][0].Nx
        r_z1_c0 = self._row_propagate(z1_last, 0, m_steps)
        r_z1_c1 = self._row_propagate(z1_last, 1, m_steps)
        r_z3_c0 = self._row_propagate(z3_last, 0, m_steps)
        r_z3_c1 = self._row_propagate(z3_last, 1, m_steps)
        H = np.zeros((4, Nx1*2 + Nx3*2), float)
        H[0, 0:Nx1] = r_z1_c0
        H[1, Nx1:2*Nx1] = r_z1_c1
        H[2, 2*Nx1:2*Nx1+Nx3] = r_z3_c0
        H[3, 2*Nx1+Nx3:2*Nx1+2*Nx3] = r_z3_c1
        return H

    def permute_on_switch(self, smb: Any, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cur_sw = getattr(smb, "switchState", None)
        if self._last_switch_state is None:
            self._last_switch_state = cur_sw
            return x, P
        if cur_sw != self._last_switch_state:
            # Topology changed → rebuild mapping from model and reset covariance moderately
            x_new, meta_new = self.get_state_vector(smb)
            self._last_switch_state = cur_sw
            # reset P to a conservative diagonal around the previous mean variance
            v = float(np.mean(np.diag(P))) if P.size else 1e-2
            P_new = np.eye(x_new.size) * max(1e-3, v)
            return x_new, P_new
        return x, P

# ---------------------------------------------------------------------------
# Real‑time estimator thread
# ---------------------------------------------------------------------------

class SMBProfileEstimator(threading.Thread):
    """Runs the simulator and periodically applies profile‑aware EKF corrections.

    • Steps the working SMB copy by `dt_model`.
    • Every `ekf_period_s` seconds, builds H over the m fine steps since last update,
      compares ideal-detector outlets with plant measurements, and corrects the
      outlet profiles of Zone 1/Zone 3 last columns.
    • Provides `snapshot_for_mpc()` for warm‑starting the MPC.
    """
    def __init__(
        self,
        smb_template: Any,
        opc_client: Any,
        adapters: AdapterHooks,
        *,
        dt_model: Optional[float] = None,
        ekf_period_s: float = 15.0,
        r_meas: float = 1e-4,
        q_proc: float = 1e-6,
        init_var: float = 1e-2,
    ):
        super().__init__(daemon=True)
        self._opc = opc_client
        self._dt = float(dt_model if dt_model is not None else smb_template.settings.get("dt", 0.5))
        self._period = float(ekf_period_s)
        self._ad = adapters
        self._lock = threading.Lock()
        self._running = threading.Event(); self._running.set()
        self._last_res: Optional[float] = None

        # Working model copy
        self._smb = smb_template.deepCopy() if hasattr(smb_template, "deepCopy") else smb_template
        # Init EKF state from the current model
        x0, meta = self._ad.get_state_vector(self._smb)
        self._meta = meta
        self._ekf = ProfileAwareEKF(n_state=x0.size, r_meas=r_meas, q_proc=q_proc, init_var=init_var)
        self._ekf.set_state(x0)

    # --- optional GUI helpers ---
    def bias(self):
        return None
    def resnorm(self):
        return self._last_res

    # --- internal helpers ---
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
            return None

    def run(self):
        last_update = time.time()
        while self._running.is_set():
            # 1) Advance simulator by one dt and get predicted outlets
            if hasattr(self._smb, "step_fast_outlets"):
                y_pred = np.asarray(self._smb.step_fast_outlets(), float)
            else:
                self._smb.step(1)
                y_pred = self._ad.outlet_y_pred(self._smb)

            # 2) EKF update at coarse period
            now = time.time()
            if (now - last_update) >= self._period:
                m_steps = max(1, int(round(self._period / max(self._dt, 1e-9))))
                # handle topology change
                self._ekf.x, self._ekf.P = self._ad.permute_on_switch(self._smb, self._ekf.x, self._ekf.P)
                # predict covariance & build H
                self._ekf.predict(self._period)
                H = self._ad.build_H(self._smb, self._meta, m_steps)
                # measurement
                y_meas = self._read_measurement()
                if y_meas is not None:
                    self._ekf.update(y_meas, y_pred, H)
                    self._last_res = float(np.linalg.norm(y_meas - y_pred))
                    # write corrected profiles back into the model
                    self._ad.set_state_vector(self._smb, self._ekf.x, self._meta)
                last_update = now

            time.sleep(max(0.0, self._dt * 0.5))

    def stop(self):
        self._running.clear()

    # --- MPC handoff ---
    def snapshot_for_mpc(self) -> Any:
        with self._lock:
            return self._smb.deepCopy() if hasattr(self._smb, "deepCopy") else self._smb

# ---------------------------------------------------------------------------
# Minimal usage sketch (names match your codebase)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def get_zone1_last_col(smb):
        # zones[1] is [Tube, Column, Tube, Column, ...]; last is Column
        return smb.zones[1][-1]
    def get_zone3_last_col(smb):
        return smb.zones[3][-1]

    adapters = LinColumnAdapters(get_zone1_last_col, get_zone3_last_col)
    # est = SMBProfileEstimator(smb_template, opc_client, adapters, dt_model=None, ekf_period_s=15.0)
    # est.start()
    pass
