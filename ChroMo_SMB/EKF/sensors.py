"""
Sensor model — 1st‑order lag for outlet analyzers (Ex/Ra × Man/Gal)
-------------------------------------------------------------------

This module implements a simple, physically‑motivated first‑order sensor model
used by the EKF as an augmented measurement dynamics:

  ż = (1/τ) (y_true − z)  ⇔  z_{k+1} = α z_k + (1−α) y_true_k,
  where α = exp(−Δt / τ) is computed per channel from its own time constant τ.

We expose:
  • SensorModel.matrices(dt)  -> (F, G)  with F = diag(α), G = I − F
  • SensorModel.predict(dt, y_true) -> z_next  (handy for tests/plots)
  • SensorModel.reset(y0)
  • Property .tau (length‑4 vector), with setters and validation

Channel order is fixed and matches the EKF:
  [Ex_Man, Ex_Gal, Ra_Man, Ra_Gal]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import logging
import numpy as np

LOGGER_NAME = "EKF"
log = logging.getLogger(LOGGER_NAME)

IDEAL_SENSOR: bool = True


# ----------------------------- Config -----------------------------
@dataclass
class SensorConfig:
    # Default per‑channel time constants [s] in EKF channel order
    tau: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.1)
    # Numerical safety limits
    min_tau: float = 1e-3  # s
    max_tau: float = 1e3   # s


# --------------------------- Sensor model -------------------------
class SensorModel:
    def __init__(self, config: SensorConfig | None = None, tau: Iterable[float] | None = None) -> None:
        cfg = config or SensorConfig()
        if tau is None:
            tau = cfg.tau
        self._min_tau = float(cfg.min_tau)
        self._max_tau = float(cfg.max_tau)
        self.tau = np.asarray(list(tau), dtype=float).reshape(4)
        self._z_last = np.zeros(4, dtype=float)  # last predicted z (for .predict helper)
        log.info("[sens] Init taus (s): %s", ", ".join(f"{t:.3g}" for t in self.tau))

    # ----------------------- Public API -----------------------
    def reset(self, y0: Iterable[float]) -> None:
        """Initialize internal z state for standalone prediction helper.
        EKFCore keeps its own z; this is primarily for debug/plotting and for
        consistent startup behavior if you choose to call .predict() externally.
        """
        y0 = np.asarray(list(y0), dtype=float).reshape(4)
        self._z_last = y0.copy()
        log.info("[sens] Reset z <- %s", ", ".join(f"{v:.3e}" for v in y0))

    def matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (F, G) for a given Δt.
           If IDEAL_SENSOR is True, use z_{k+1} = y_true_k  (F=0, G=I).
           Else, first-order lag: z_{k+1} = F z_k + G y_true_k with F=diag(exp(-Δt/τ)), G=I-F.
        """
        import numpy as _np
        dt = max(0.0, float(dt))

        if IDEAL_SENSOR:
            F = _np.zeros((4, 4), dtype=float)
            G = _np.eye(4, dtype=float)
            log.debug("[sens] IDEAL matrices dt=%.3f -> F=0, G=I", dt)
            return F, G

        tau = self._validated_tau()
        alpha = np.exp(-dt / np.maximum(tau, self._min_tau))
        alpha = np.clip(alpha, 0.0, 1.0)
        F = np.diag(alpha)
        G = np.eye(4) - F
        log.debug("[sens] matrices dt=%.3f -> alpha=%s", dt, ", ".join(f"{a:.3g}" for a in alpha))
        return F, G

    def predict(self, dt: float, y_true: Iterable[float]) -> np.ndarray:
        """Advance internal z by one step given y_true (useful for tests/plots)."""
        y_true = np.asarray(list(y_true), dtype=float).reshape(4)
        F, G = self.matrices(dt)
        z_next = F @ self._z_last + G @ y_true
        self._z_last = z_next
        log.debug("[sens] predict dt=%.3f -> z=%s", dt, ", ".join(f"{v:.3e}" for v in z_next))
        return z_next

    # ----------------------- Utilities ------------------------
    def set_tau(self, idx: int, value: float) -> None:
        assert 0 <= idx < 4, "channel index out of range"
        self.tau[idx] = float(value)
        self._enforce_tau_bounds()
        log.info("[sens] tau[%d] := %.4g s", idx, self.tau[idx])

    def set_all_tau(self, value: float) -> None:
        self.tau[:] = float(value)
        self._enforce_tau_bounds()
        log.info("[sens] tau[:] := %.4g s", self.tau[0])

    # ----------------------- Internals ------------------------
    def _validated_tau(self) -> np.ndarray:
        t = np.asarray(self.tau, dtype=float).reshape(4)
        # Replace non‑finite or non‑positive with min_tau
        bad = ~np.isfinite(t) | (t <= 0)
        if np.any(bad):
            t[bad] = self._min_tau
        # Clamp to [min, max]
        t = np.clip(t, self._min_tau, self._max_tau)
        return t

    def _enforce_tau_bounds(self) -> None:
        self.tau[:] = self._validated_tau()


# ------------------------- Simple self‑test ------------------------
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    sens = SensorModel(tau=(0.2, 0.5, 0.2, 0.5))
    sens.reset([0, 0, 0, 0])
    y = [1.0, 1.0, 1.0, 1.0]
    for k in range(5):
        z = sens.predict(1.0, y)
        print(k, z)
