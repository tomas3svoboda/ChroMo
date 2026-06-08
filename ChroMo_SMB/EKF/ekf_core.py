"""
EKF Core — rigorous Extended Kalman Filter for SMB column profiles with sensor lag
----------------------------------------------------------------------------------

Implements the EKF math exactly as in the dissertation excerpt, using Joseph form
for numerical stability. The filter maintains an augmented state [x; z], where:
  • x ∈ R^N  – concatenated axial concentration profiles for both components
  • z ∈ R^4  – first-order sensor states (Ex/Ra × Man/Gal)

Full-state mode
---------------
When EKFConfig.full_state == True, the EKF reads/writes the *entire* 4-column ring state
via twin.get_profiles_full() / twin.set_profiles_full(). Otherwise it uses the legacy
2-column path twin.get_profiles() / twin.set_profiles().

Prediction over an EKF period [t_start, t_end]:
  • Covariance is propagated using CN transition matrices A_seq (one per dt).
  • Sensor block uses a closed-form F,G (or model-provided matrices).
  • Process noises Qx, Qz are integrated over Δt.

Update at t_end:
  • IDEAL mode: y = S x
  • 1st-order sensors: y = z,  z_{k+1} = F z_k + G (S x_k)
  • Joseph form for covariance.

Interfaces expected by EKFManager.tick(...):
  tick(A_seq, S, y_meas, sensor_model, twin, dt_model, t_start, t_end) -> diagnostics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

LOGGER_NAME = "EKF"
log = logging.getLogger(LOGGER_NAME)


# -------------------------- Utilities --------------------------
def _is_pos_semidef(M: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(M + 1e-15 * np.eye(M.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False


def _chol_solve_spd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A X = B for SPD A using Cholesky; falls back to pinv if needed."""
    try:
        L = np.linalg.cholesky(A)
        Y = np.linalg.solve(L, B)
        X = np.linalg.solve(L.T, Y)
        return X
    except np.linalg.LinAlgError:
        log.warning("[ekf] Cholesky failed; falling back to pinv (matrix may be ill-conditioned)")
        return np.linalg.pinv(A) @ B


def _block_diag(*blocks: np.ndarray) -> np.ndarray:
    sizes = [b.shape[0] for b in blocks]
    N = sum(sizes)
    out = np.zeros((N, N), dtype=blocks[0].dtype)
    off = 0
    for b in blocks:
        n = b.shape[0]
        out[off:off+n, off:off+n] = b
        off += n
    return out


# ---------------------- Hyperparameters -----------------------
@dataclass
class EKFConfig:
    # Initial covariance (per-state variances)
    init_var_profile: float = 1e-4
    init_var_sensor: float = 1e-6

    # Process noise *rates* (units per second). Integrated over Δt during predict.
    qx_rate: float = 1e-8
    qz_rate: float = 1e-8

    # Measurement noise (sensor) — diagonal by default
    r_meas_diag: tuple[float, float, float, float] = (1e-6, 1e-6, 1e-6, 1e-6)

    # Sensor dynamics fallback
    tau_default: float = 0.1

    # Numerics
    min_variance: float = 1e-12
    clamp_profiles_nonnegative: bool = True

    # Full-state toggle
    full_state: bool = True

    # ---------------- Robust update controls ----------------
    # Skip update if a permutation occurred within this many seconds before t_end
    switch_blackout: float = 1.0       # seconds (set 0 to disable)
    blackout_skip_update: bool = True  # True: skip update; False: just inflate R at switch
    switch_R_scale: float = 1e3        # used only if blackout_skip_update=False

    # NIS (Mahalanobis) gating: if pre-gating NIS > threshold, inflate R on-the-fly
    nis_gate: float = 50.0             # 4-dof chi^2 ~18.47 (99.9%); 50 is conservative
    nis_inflate_factor: float = 10.0   # multiply R by up to (nis/nis_gate)*this
    nis_R_scale_max: float = 1e6       # cap for per-tick inflation

    # Optional: re-seed covariance when a switch is inside the last blackout window (rarely needed)
    reset_P_on_switch: bool = False

    # --- nové přepínače ---
    disable_update: bool = True       # True => čistě model (K=0), žádná korekce
    write_back_profiles: bool = False    # False => nikdy nezapisuj x_post zpět do twin
    dump_diagnostics: bool = True       # True => vrátí rozšířené diag (Syy, S P_xx S^T, atd.)

# -------------------------- EKF Core ---------------------------
class EKFCore:
    def __init__(self, config: EKFConfig | None = None) -> None:
        self.cfg = config or EKFConfig()
        # Lazy-initialized dimensions/state
        self._N: Optional[int] = None  # profile dimension
        self._P: Optional[np.ndarray] = None  # full covariance (N+4)x(N+4)
        self._z: Optional[np.ndarray] = None  # sensor state (4,)
        # Stats
        self._tick_idx = 0

    # --------------- Public API ---------------
    def tick(
        self,
        *,
        A_seq: list[np.ndarray],
        S: np.ndarray,
        y_meas: list[float] | np.ndarray,
        sensor_model: Any,
        twin: Any,
        dt_model: float,
        t_start: float,
        t_end: float,
    ) -> dict:
        """EKF predict+update; umí i čistý 'model-only' běh (disable_update=True)."""
        def _power_with_Q(T: np.ndarray, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            n = T.shape[0]
            Tk = np.eye(n, dtype=T.dtype)
            Sk = np.zeros((n, n), dtype=T.dtype)
            Tb = T.copy()
            Sb = Q.copy()
            kk = int(k)
            while kk > 0:
                if kk & 1:
                    Sk = Sk + Tk @ Sb @ Tk.T
                    Tk = Tk @ Tb
                Sb = Sb + Tb @ Sb @ Tb.T
                Tb = Tb @ Tb
                kk >>= 1
            return Tk, Sk

        def _is_perm(A: np.ndarray) -> bool:
            if A.ndim != 2 or A.shape[0] != A.shape[1]: return False
            if not ((A == 0.0) | (A == 1.0)).all():     return False
            return np.allclose(A.sum(axis=0), 1.0) and np.allclose(A.sum(axis=1), 1.0)

        y_meas = np.asarray(y_meas, dtype=float).reshape(4)

        # --- načti aktuální stav z twin (plný stav) ---
        x_prior = self._get_profiles(twin).astype(float).reshape(-1)
        Nx = x_prior.size; Nz = 4

        if self._N is None:
            self._initialize(N=Nx, y0=y_meas)
            log.info("[ekf] Initialized: N=%d, P shape=%s", Nx, self._P.shape)
        elif Nx != self._N:
            raise ValueError(f"Profile dimension changed from {self._N} to {Nx}; cannot continue.")

        Δt = max(0.0, float(t_end - t_start))
        if Δt < 1e-12:
            return {"status": "skipped", "reason": "zero_dt"}

        # --- per-step Q a F,G ---
        qx_step = max(self.cfg.qx_rate * float(dt_model), self.cfg.min_variance)
        qz_step = max(self.cfg.qz_rate * float(dt_model), self.cfg.min_variance)
        Fz_step, Gz_step = self._sensor_FG(sensor_model, float(dt_model))

        # ---------------- PREDICT kovariance (rychlý běh přes bloky) ----------------
        P_full = self._P.copy()
        if A_seq:
            Q_joint = np.zeros((Nx + Nz, Nx + Nz))
            Q_joint[:Nx, :Nx] = qx_step * np.eye(Nx)
            Q_joint[Nx:, Nx:] = qz_step * np.eye(Nz)
            # B_step závisí na S
            B_step = Gz_step @ S

            # komprese do běhů stejného A
            runs = []
            i = 0
            while i < len(A_seq):
                Ai = A_seq[i]; j = i + 1
                while j < len(A_seq) and (A_seq[j] is Ai): j += 1
                runs.append((np.asarray(Ai), j - i)); i = j

            for A_run, k in runs:
                if A_run is None or A_run.shape != (Nx, Nx) or not np.isfinite(A_run).all():
                    A_run = np.eye(Nx)
                if _is_perm(A_run):
                    Pperm = np.eye(Nx + Nz); Pperm[:Nx, :Nx] = A_run
                    P_full = Pperm @ P_full @ Pperm.T
                else:
                    T = np.zeros((Nx + Nz, Nx + Nz))
                    T[:Nx, :Nx] = A_run
                    T[Nx:, :Nx] = B_step
                    T[Nx:, Nx:] = Fz_step
                    Tk, Sk = _power_with_Q(T, Q_joint, int(k))
                    P_full = Tk @ P_full @ Tk.T + Sk
        else:
            P_full[:Nx, :Nx] += max(self.cfg.qx_rate * Δt, self.cfg.min_variance) * np.eye(Nx)
            P_full[Nx:, Nx:] += max(self.cfg.qz_rate * Δt, self.cfg.min_variance) * np.eye(Nz)

        # rozbal bloky
        P_xx = P_full[:Nx, :Nx]
        P_xz = P_full[:Nx, Nx:]
        P_zz = P_full[Nx:, Nx:]

        # ---------------------- „MODEL-ONLY“ BRÁNA ----------------------
        from sensors import IDEAL_SENSOR  # runtime toggle
        if self.cfg.disable_update:
            # žádná korekce; jen diagnostika
            if IDEAL_SENSOR:
                y_pred = (S @ x_prior).reshape(4)
                r = y_meas - y_pred
                Syy = S @ P_xx @ S.T + np.diag(np.asarray(self.cfg.r_meas_diag, float))
            else:
                y_true = (S @ x_prior).reshape(4)
                z_pred = Fz_step @ self._z + Gz_step @ y_true
                y_pred = z_pred
                r = y_meas - z_pred
                Syy = P_zz + np.diag(np.asarray(self.cfg.r_meas_diag, float))

            # ulož P, případně *ne*zapisuj do twin (write_back_profiles=False)
            self._P = _symmetrize(P_full)
            self._tick_idx += 1
            if self.cfg.write_back_profiles:
                self._set_profiles(twin, x_prior)

            diag = {
                "status": "predict_only",
                "tick": self._tick_idx,
                "Δt": Δt,
                "residual": float(np.linalg.norm(r)),
                "innov": r.tolist(),
                "nis": float(r @ _chol_solve_spd(Syy, np.eye(4)) @ r),
                "y_pred": y_pred.tolist(),
                "y_meas": y_meas.tolist(),
                "cond": float(_safe_cond(Syy)),
            }
            if self.cfg.dump_diagnostics:
                diag["Syy_diag"] = np.diag(Syy).astype(float).tolist()
                diag["SPSt_diag"] = np.diag(S @ P_xx @ S.T).astype(float).tolist()
                diag["R_diag"] = list(self.cfg.r_meas_diag)
            return diag
        # ---------------------- /MODEL-ONLY ----------------------

        # ---------------------- STANDARD UPDATE (IDEAL SENSOR) ----------------------
        R = np.diag(np.asarray(self.cfg.r_meas_diag, dtype=float))

        if IDEAL_SENSOR:
            y_pred = (S @ x_prior).reshape(4)
            r = y_meas - y_pred
            Syy = S @ P_xx @ S.T + R
            if not np.isfinite(Syy).all():
                log.warning("[ekf] Syy non-finite; skipping update.")
                self._P = _symmetrize(P_full); self._tick_idx += 1
                return {"status": "skipped_bad_Syy"}

            Syy_inv = _chol_solve_spd(Syy, np.eye(4))
            Kx = P_xx @ S.T @ Syy_inv
            Kz = P_xz.T @ S.T @ Syy_inv

            x_post = x_prior + Kx @ r
            if self.cfg.clamp_profiles_nonnegative:
                eps = 1e-6
                neg = x_post < 0
                x_post[neg & (x_post >= -eps)] = (x_post[neg & (x_post >= -eps)] ** 2) / (2 * eps)
                x_post[x_post < -eps] = 0.0
            z_post = (S @ x_post).reshape(4)

            HP = np.hstack([S @ P_xx, S @ P_xz])
            P_post_full = P_full - HP.T @ (Syy_inv @ HP)

        else:
            # první řád: y = z
            y_true = (S @ x_prior).reshape(4)
            z_pred = Fz_step @ self._z + Gz_step @ y_true
            r = y_meas - z_pred

            Syy = P_zz + R
            if not np.isfinite(Syy).all():
                log.warning("[ekf] Syy non-finite; skipping update.")
                self._P = _symmetrize(P_full); self._tick_idx += 1
                return {"status": "skipped_bad_Syy"}

            Syy_inv = _chol_solve_spd(Syy, np.eye(4))
            Kx = P_xz @ Syy_inv
            Kz = P_zz @ Syy_inv

            x_post = x_prior + Kx @ r
            z_post = z_pred + Kz @ r
            if self.cfg.clamp_profiles_nonnegative:
                eps = 1e-6
                neg = x_post < 0
                x_post[neg & (x_post >= -eps)] = (x_post[neg & (x_post >= -eps)] ** 2) / (2 * eps)
                x_post[x_post < -eps] = 0.0

            HP = np.hstack([np.zeros((4, Nx), dtype=P_zz.dtype), P_zz])
            P_post_full = P_full - HP.T @ (Syy_inv @ HP)

        # --- persist & (ne)zapisuj do twin ---
        self._P = _symmetrize(P_post_full)
        self._z = z_post
        self._tick_idx += 1
        if self.cfg.write_back_profiles and np.isfinite(x_post).all():
            self._set_profiles(twin, x_post)

        S_diag = np.clip(np.diag(Syy), 1e-15, None)
        innov_z = (r / np.sqrt(S_diag)).astype(float)
        diag = {
            "status": "ok",
            "tick": self._tick_idx,
            "Δt": Δt,
            "residual": float(np.linalg.norm(r)),
            "innov": r.tolist(),
            "innov_z": innov_z.tolist(),
            "nis": float(r @ Syy_inv @ r),
            "y_pred": ((S @ x_post).reshape(4) if IDEAL_SENSOR else z_pred).tolist(),
            "y_meas": y_meas.tolist(),
            "cond": float(_safe_cond(Syy)),
        }
        if self.cfg.dump_diagnostics:
            diag["Syy_diag"]  = np.diag(Syy).astype(float).tolist()
            diag["SPSt_diag"] = np.diag(S @ P_xx @ S.T).astype(float).tolist()
            diag["R_diag"]    = list(self.cfg.r_meas_diag)
            diag["Kx_norm"]   = float(np.linalg.norm(Kx, 2))
            diag["Pxx_trace"] = float(np.trace(P_xx))
        return diag



    def _power_with_Q(self, T: np.ndarray, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (T^k, Σ_k) where Σ_k = sum_{i=0}^{k-1} T^i Q (Tᵀ)^i using doubling."""
        n = T.shape[0]
        Tk  = np.eye(n, dtype=T.dtype)
        Sk  = np.zeros((n, n), dtype=T.dtype)
        Tb  = T.copy()
        Sb  = Q.copy()
        kk  = int(k)
        while kk > 0:
            if kk & 1:
                # accumulate Sk ← Sk + Tk Sb Tkᵀ ; Tk ← Tk Tb
                Sk = Sk + Tk @ Sb @ Tk.T
                Tk = Tk @ Tb
            # square: Tb ← Tb^2 ; Sb ← Sb + Tb Sb Tbᵀ
            Sb = Sb + Tb @ Sb @ Tb.T
            Tb = Tb @ Tb
            kk >>= 1
        return Tk, Sk

    def reset_profile_covariance(self, var: float | None = None) -> None:
        """Re-seed P_xx at a switch boundary (kept for compatibility; not needed in full-state)."""
        if self._P is None or self._N is None:
            return
        Nx = self._N
        sigma2 = float(var) if (var is not None) else max(self.cfg.init_var_profile, self.cfg.min_variance)
        self._P[:Nx, :Nx] = np.eye(Nx) * sigma2
        self._P[:Nx, Nx:] = 0.0
        self._P[Nx:, :Nx] = 0.0

    # --------------- Internal helpers ---------------
    def _initialize(self, N: int, y0: np.ndarray) -> None:
        self._N = N
        self._z = y0.astype(float).reshape(4)
        var_x = max(self.cfg.init_var_profile, self.cfg.min_variance)
        var_z = max(self.cfg.init_var_sensor, self.cfg.min_variance)
        P_xx = np.eye(N) * var_x
        P_zz = np.eye(4) * var_z
        self._P = _block_diag(P_xx, P_zz)

    def _sensor_FG(self, sensor_model: Any, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(sensor_model, "matrices"):
            F, G = sensor_model.matrices(dt)
            return np.asarray(F, float), np.asarray(G, float)
        tau = None
        if hasattr(sensor_model, "tau"):
            tau = np.asarray(getattr(sensor_model, "tau"), dtype=float).reshape(4)
        if tau is None:
            tau = np.full(4, float(self.cfg.tau_default), dtype=float)
        tau = np.maximum(tau, 1e-6)
        alpha = np.exp(-dt / tau)
        F = np.diag(alpha)
        G = np.eye(4) - F
        return F, G

    # ----------- NEW: twin I/O helpers for profile vectors -----------
    def _get_profiles(self, twin: Any) -> np.ndarray:
        return np.asarray(twin.get_profiles_full(), dtype=float)

    def _set_profiles(self, twin: Any, x: np.ndarray) -> None:
        twin.set_profiles_full(x)

    # -----------------------------------------------------------------


# -------------------------- Numerics ---------------------------
def _safe_cond(M: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(M))
    except Exception:
        return float("inf")


def _symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)
