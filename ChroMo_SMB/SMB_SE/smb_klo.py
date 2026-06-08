# smb_klo.py
# -----------------------------------------------------------------------------
# Single-source-of-truth Kalman/Luenberger observer:
#   • Engine (DT) owns the states and advances physics (CN).
#   • This module computes a fixed, steady-state gain K_inf per terminal column.
#   • Each observer tick: read outlet error e = y - H c, then do c += alpha * K_inf @ e.
#   • Profiles are updated IN-PLACE in the engine's terminal columns (zones 1 & 3).
#
# Assumptions satisfied by your codebase:
#   - LinColumn caches A's LU (comp._lu), B (comp._B), and _inlet_coef (for the plant).
#   - smb_engine provides zones[zone] with [..., upstream_tube, terminal_column].
#   - smb_sensor.SMBOutletSensor.read_zone(zone) -> (Man, Gal) outlet readings.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import numpy as np
import sys, time
from scipy.linalg import lu_solve, block_diag

from smb_engine import SMBTwinEngine
from smb_sensor import SMBOutletSensor


# ------------------------ small numerics helpers ------------------------

def _first_diff_tridiag(n: int) -> np.ndarray:
    """T = DᵀD for first-difference operator (Neumann-like ends). Tri-diagonal SPD."""
    T = np.zeros((n, n), dtype=float)
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
    """Stable inverse for 2x2 SPD matrices (S = H P Hᵀ + R)."""
    try:
        L = np.linalg.cholesky(M)
        Linv = np.linalg.inv(L)
        return Linv.T @ Linv
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


# ------------------------ Column-local KLO (no internal state) ------------------------

class ColumnTailKLO:
    """
    Steady-state Luenberger observer over one terminal LinColumn (zone 1 or 3),
    directly correcting the engine's column profile in place.

    State in engine: c = [Man[0..Nx-1], Gal[0..Nx-1]]
    Meas: y = [Man_outlet, Gal_outlet] (last cell of each component)
    Update each tick (after engine advances):
        e = y - H c
        c <- c + alpha * K_inf @ e
    """

    def __init__(
        self,
        smb_obj: object,
        zone: int,                 # 1 = Extract, 3 = Raffinate
        Nx: int,
        sensor: SMBOutletSensor,
        # Kalman design (steady-state):
        q_sigma2: float = 1e-7,    # σ²·I in Q
        q_alpha: float = 5e-6,     # α·(DᵀD) in Q  (damps high-k outlet mode)
        r_meas: float = 1e-4,      # (g/L)^2
        # runtime behavior:
        deadband_gpl: float = 0.05,   # ignore tiny |e| to avoid micro-chatter
        alpha_injection: float = 1.0, # 0..1 — scale the injection if you want gentler nudges
        max_cell_step_frac: float = 0.30,  # trust region per cell (fraction of |c|+eps)
        # Riccati solver:
        riccati_maxit: int = 200,
        riccati_relax: float = 0.1,
        riccati_tol: float = 1e-9,
        # Diagnostics (throttled):
        diag: bool = False,
        diag_interval_s: float = 2.0,
    ):
        assert zone in (1, 3), "zone must be 1 (Extract) or 3 (Raffinate)"
        self.smb = smb_obj
        self.zone = int(zone)
        self.Nx = int(Nx)
        self.sensor = sensor

        # measurement model H (last cell of each component)
        self.H = np.zeros((2, 2 * self.Nx))
        w = min(3, self.Nx)      # average over last w cells
        wgt = 1.0 / w
        # Man
        self.H[0, self.Nx - w : self.Nx] = wgt
        # Gal
        self.H[1, 2*self.Nx - w : 2*self.Nx] = wgt

        # noise models
        self.R = np.eye(2) * float(r_meas)
        T = _first_diff_tridiag(self.Nx)
        Q_blk = float(q_sigma2) * np.eye(self.Nx) + float(q_alpha) * T
        self.Q = block_diag(Q_blk, Q_blk)

        # runtime knobs
        self.deadband = float(deadband_gpl)
        self.alpha = float(np.clip(alpha_injection, 0.0, 1.0))
        self.max_cell_step_frac = float(max_cell_step_frac)

        # Riccati cfg
        self._riccati_maxit = int(riccati_maxit)
        self._riccati_relax = float(riccati_relax)
        self._riccati_tol = float(riccati_tol)

        # handles
        self._tube_up = None
        self._col = None
        self._last_col_id = None
        self._last_warn = 0.0

        # design matrices/gain
        self._F = None
        self._K = None

        # --- diagnostics state ---
        self._diag_enabled = bool(diag)
        self._diag_interval = float(diag_interval_s)
        self._diag_last_ts = 0.0
        self._r_meas = float(r_meas)
        self._q_sigma2 = float(q_sigma2)
        self._q_alpha = float(q_alpha)

        self._bind_and_design(force=True)

    # ---------- binding & design ----------

    def _warn_once(self, msg: str):
        now = time.time()
        if now - getattr(self, "_last_warn", 0.0) > 2.0:
            print(f"[KLO z{self.zone}] {msg}", file=sys.stderr)
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

    def _bind_and_design(self, force: bool = False):
        tube, col = self._pair_tube_and_column()
        if col is None:
            return
        col_id = id(col)
        switched = (self._last_col_id is not None and col_id != self._last_col_id) or force
        self._last_col_id = col_id
        self._tube_up, self._col = tube, col

        if switched or self._F is None or self._K is None:
            self._build_F()
            self._compute_K_inf()

    def _build_F(self):
        """F ≈ A⁻¹B (block-diag) from LinColumn caches; falls back to identity."""
        Nx = self.Nx
        I = np.eye(Nx)

        # component 0 (Man)
        try:
            c0 = self._col.components[0]
            F0 = lu_solve(c0._lu, c0._B) if hasattr(c0, "_lu") and hasattr(c0, "_B") else I
        except Exception:
            F0 = I

        # component 1 (Gal)
        try:
            c1 = self._col.components[1]
            F1 = lu_solve(c1._lu, c1._B) if hasattr(c1, "_lu") and hasattr(c1, "_B") else I
        except Exception:
            F1 = I

        self._F = block_diag(F0, F1)

    def _compute_K_inf(self):
        """
        Steady-state Riccati:
            P = FPFᵀ + Q − FPHᵀ (H P Hᵀ + R)^{-1} H P Fᵀ
            K = P Hᵀ (H P Hᵀ + R)^{-1}
        Run at bind/switch; result is 2*Nx x 2 gain mapping outlet errors → full profile.
        """
        assert self._F is not None
        F, H, Q, R = self._F, self.H, self.Q, self.R
        n = F.shape[0]

        P = Q.copy()  # reasonable start
        relax = self._riccati_relax
        tol = self._riccati_tol

        for _ in range(self._riccati_maxit):
            HPHT = H @ P @ H.T
            S = HPHT + R
            Sinv = _chol_inv_2x2(S)
            PFHT = P @ H.T
            FPFt = F @ P @ F.T
            P_new = FPFt + Q - (F @ PFHT) @ Sinv @ (H @ P @ F.T)
            P_new = 0.5 * (P_new + P_new.T)      # symmetrize
            P = (1 - relax) * P + relax * (P_new + 1e-12 * np.eye(n))
            if np.allclose(H @ P @ H.T, HPHT, rtol=0, atol=tol):
                break

        S = H @ P @ H.T + R
        Sinv = _chol_inv_2x2(S)
        self._K = P @ H.T @ Sinv

    # ---------- runtime tick (no internal state; operate on engine) ----------

    def _read_measurement(self) -> Optional[np.ndarray]:
        try:
            y = self.sensor.read_zone(self.zone)  # (Man, Gal)
            if y is None:
                return None
            return np.array([float(y[0]), float(y[1])], float)
        except Exception as ex:
            self._warn_once(f"sensor read failed: {ex}")
            return None

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

    def step_once(self, sim_t: float) -> None:
        # Rebind (handles rotation) and redesign if column changed
        self._bind_and_design(force=False)
        if self._col is None or self._K is None:
            return

        # Read DT state and outlet measurement
        x = self._read_engine_profile()
        y = self._read_measurement()
        if y is None:
            return

        # Residual at outlets (model vs measurement)
        Hx = self.H @ x
        e = y - Hx
        for i in (0, 1):
            if abs(e[i]) < self.deadband:
                e[i] = 0.0

        # Smooth spatial correction
        delta_unclipped = self._K @ e

        # Trust region cap (avoid shocking the PDE)
        eps = 1e-9
        limit = self.max_cell_step_frac * (np.abs(x) + eps)
        delta = np.clip(delta_unclipped, -limit, limit)

        # Apply scaled injection and keep physical
        upd = self.alpha * delta
        x_new = np.maximum(x + upd, 0.0)

        Nx = self.Nx  # ← ADD THIS LINE

        # 1) Enforce outlet BC: zero gradient → last two equal
        # Man
        x_new[Nx - 1] = x_new[Nx - 2]
        # Gal
        x_new[2*Nx - 1] = x_new[2*Nx - 2]

        k = min(10, Nx - 1)   # ensure start index ≥ 1
        beta = 0.9
        # Man
        target = x_new[Nx - 2]
        for i in range(Nx - k, Nx - 1):
            w = (i - (Nx - k)) / max(k - 1, 1)
            x_new[i] = (1 - beta*w) * x_new[i] + (beta*w) * target
        # Gal
        target = x_new[2*Nx - 2]
        for i in range(2*Nx - k, 2*Nx - 1):
            w = (i - (2*Nx - k)) / max(k - 1, 1)
            x_new[i] = (1 - beta*w) * x_new[i] + (beta*w) * target

        # --- concise, throttled diagnostics ---
        if self._diag_enabled:
            now = time.time()
            clipped_frac = float(np.mean(np.abs(delta_unclipped) > limit))
            e_m, e_g = float(e[0]), float(e[1])
            upd_inf = float(np.max(np.abs(upd))) if upd.size else 0.0
            should_print = ((now - self._diag_last_ts) >= self._diag_interval) or (clipped_frac >= 0.05)
            if should_print:
                depth_cells = float(np.sqrt(max(self._q_alpha / max(self._q_sigma2, 1e-30), 0.0)))
                print(
                    f"[KLO z{self.zone}] t={sim_t:.1f}s "
                    f"e=(Man:{e_m:+.3g}, Gal:{e_g:+.3g}) "
                    f"|Δ|∞={upd_inf:.3g}g/L clip={clipped_frac:.0%} "
                    f"α={self.alpha:g} mcf={self.max_cell_step_frac:g} "
                    f"depth≈{depth_cells:.0f}cells r={self._r_meas:g}"
                )
                self._diag_last_ts = now

        # Write back to engine (single source of truth)
        self._write_engine_profile(x_new)

    # ---------- debug hook ----------
    def gain(self) -> np.ndarray:
        return self._K.copy() if self._K is not None else np.zeros((2 * self.Nx, 2))

    def notify_switch(self) -> None:
        """
        Called by the manager at every zone boundary.
        Forces a re-bind to the terminal column and recomputes K∞.
        """
        # Invalidate the cached column id so _bind_and_design() won’t try to be clever
        self._last_col_id = None
        self._bind_and_design(force=True)


# ------------------------ Manager for two outlets ------------------------

class SMBKLOManager:
    """
    Runs KLO on both outlet zones. Observer ticks every kf_dt_multiples * dt
    AFTER the engine advances (subscribe to engine callback).

    Variant 2: zone-1 (Extract) can be tamed via per-zone overrides:
        r_meas_z1, alpha_injection_z1, max_cell_step_frac_z1, deadband_gpl_z1
    """
    def __init__(
        self,
        engine: SMBTwinEngine,
        sensor: SMBOutletSensor,
        Nx: int = 100,
        kf_dt_multiples: int = 1,
        # design (global, used for both zones unless overridden for z1):
        q_sigma2: float = 1e-7, q_alpha: float = 5e-6, r_meas: float = 1e-4,
        # runtime (global):
        deadband_gpl: float = 0.05, alpha_injection: float = 1.0, max_cell_step_frac: float = 0.30,
        # riccati:
        riccati_maxit: int = 200, riccati_relax: float = 0.1, riccati_tol: float = 1e-9,
        # diagnostics:
        diag: bool = False, diag_interval_s: float = 2.0,
        # ---------- per-zone (Extract: zone 1) overrides ----------
        r_meas_z1: float | None = None,
        alpha_injection_z1: float | None = None,
        max_cell_step_frac_z1: float | None = None,
        deadband_gpl_z1: float | None = None,
    ):
        self.engine = engine
        self.sensor = sensor
        self.Nx = int(Nx)
        self._kf_dt_mult = int(kf_dt_multiples)
        self._next_sim_t: Optional[float] = None
        self.klo_ex: Optional[ColumnTailKLO] = None   # zone 1
        self.klo_ra: Optional[ColumnTailKLO] = None   # zone 3
        self._bound = False

        # base config (shared by default by both zones)
        self._cfg = dict(
            q_sigma2=q_sigma2, q_alpha=q_alpha, r_meas=r_meas,
            deadband_gpl=deadband_gpl, alpha_injection=alpha_injection, max_cell_step_frac=max_cell_step_frac,
            riccati_maxit=riccati_maxit, riccati_relax=riccati_relax, riccati_tol=riccati_tol,
            diag=diag, diag_interval_s=diag_interval_s,
        )

        # store overrides for z1 (Extract)
        self._z1_over = dict(
            r_meas=r_meas_z1,
            alpha_injection=alpha_injection_z1,
            max_cell_step_frac=max_cell_step_frac_z1,
            deadband_gpl=deadband_gpl_z1,
        )

        self.engine.subscribe(self._on_engine_update)
        # Track engine’s monotone switch counter
        self._last_switch_idx: int | None = None
        print("[KLO] manager constructed; waiting for engine station (_smb)…", flush=True)

    def _ensure_bound(self):
        if self._bound:
            return
        # accept both attributes just in case
        smb = getattr(self.engine, "_smb", None) or getattr(self.engine, "smb", None)
        if smb is None:
            return

        # zone 1 config = base + optional per-zone overrides
        cfg1 = dict(self._cfg)
        for k, v in self._z1_over.items():
            if v is not None:
                cfg1[k] = v

        # instantiate observers
        self.klo_ex = ColumnTailKLO(smb, zone=1, Nx=self.Nx, sensor=self.sensor, **cfg1)
        self.klo_ra = ColumnTailKLO(smb, zone=3, Nx=self.Nx, sensor=self.sensor, **self._cfg)

        dt = float(getattr(smb, "dt", 1.0))
        sim_t = float(getattr(smb, "timer", 0.0))
        self._next_sim_t = sim_t + self._kf_dt_mult * dt
        self._bound = True
        print(f"[KLO] initialized (every {self._kf_dt_mult}*dt)")

    def _on_engine_update(self, _res, sim_t: float):
        self._ensure_bound()
        if not self._bound or self._next_sim_t is None or sim_t < self._next_sim_t:
            return

         # Detect engine boundary and re-bind observers if needed
        try:
            idx = getattr(self.engine, "_switch_index", None)
            if idx is not None and idx != self._last_switch_idx:
                if self.klo_ex is not None:
                    self.klo_ex.notify_switch()
                if self.klo_ra is not None:
                    self.klo_ra.notify_switch()
                self._last_switch_idx = idx
        except Exception as ex:
            print(f"[KLO] switch-edge handling failed: {ex}", file=sys.stderr)

        # run z1 (if present)
        if self.klo_ex is not None:
            try:
                self.klo_ex.step_once(sim_t)
            except Exception as ex:
                print(f"[KLO] z1 error: {ex}", file=sys.stderr)

        # run z3 (if present)
        if self.klo_ra is not None:
            try:
                self.klo_ra.step_once(sim_t)
            except Exception as ex:
                print(f"[KLO] z3 error: {ex}", file=sys.stderr)

        smb = getattr(self.engine, "_smb", None) or getattr(self.engine, "smb", None)
        dt = float(getattr(smb, "dt", 1.0)) if smb is not None else 1.0
        self._next_sim_t = sim_t + self._kf_dt_mult * dt

    # -------- convenience hooks --------
    def get_tail_state(self, zone: int) -> np.ndarray:
        self._ensure_bound()
        klo = self.klo_ex if zone == 1 else self.klo_ra
        if klo is None or klo._col is None:
            return np.zeros(2 * self.Nx)
        Nx = self.Nx
        m = np.asarray(klo._col.components[0].c, float)
        g = np.asarray(klo._col.components[1].c, float)
        return np.concatenate([m[:Nx], g[:Nx]])

    def get_outlet_estimate(self, zone: int) -> Tuple[float, float]:
        self._ensure_bound()
        klo = self.klo_ex if zone == 1 else self.klo_ra
        if klo is None or klo._col is None:
            return (0.0, 0.0)
        Nx = self.Nx
        m_out = float(klo._col.components[0].c[Nx - 1])
        g_out = float(klo._col.components[1].c[Nx - 1])
        return (m_out, g_out)

    def get_gain(self, zone: int) -> np.ndarray:
        self._ensure_bound()
        klo = self.klo_ex if zone == 1 else self.klo_ra
        return klo.gain() if klo is not None else np.zeros((2 * self.Nx, 2))

    def snapshot(self) -> Dict[str, object]:
        return {
            "Nx": self.Nx,
            "z1_gain": self.get_gain(1),
            "z3_gain": self.get_gain(3),
            "z1_outlet_hat": self.get_outlet_estimate(1),
            "z3_outlet_hat": self.get_outlet_estimate(3),
        }
