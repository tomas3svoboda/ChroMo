"""
mpc_optimizer.py Mead wrapper for SMB MPC
==========================================================

Owns the SciPy `minimize(..., method='Nelder-Mead')` loop with a strict
**time budget** per SI. Plays nicely with `mpc_objective.Objective` by:

â€¢ Option A (preferred by current orchestrator):
  - Orchestrator constructs `Objective`, calls `set_seed(...)` and
    `prepare_t0()`, computes baseline if desired, then calls
    `NMRunner.solve(x0, objective, time_budget_s, cfg)`.

â€¢ Option B (convenience):
  - Orchestrator passes `seed_station`, `active_setpoints`, and
    `objective_kwargs`; `NMRunner.solve(...)` will construct the
    `Objective`, do `set_seed(...)` + `prepare_t0()` internally.

In both cases, the runner tracks **best-so-far** and returns a compact `Result`
for the GUI (iterations, elapsed, status, message, best_x, best_J, metrics).

If SciPy is missing, a tiny NM fallback is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable, Any
import time
import math
import numpy as np

try:
    import scipy.optimize as _spo
except Exception:  # pragma: no cover
    _spo = None

# Local objective type (duck-typed)
try:  # pragma: no cover
    from mpc_objective import Objective
except Exception:  # pragma: no cover
    Objective = Any  # type: ignore


# ---------------------------------------------------------------------------
# Config & Result
# ---------------------------------------------------------------------------

@dataclass
class NMConfig:
    maxiter: int = 400
    xatol: float = 1e-2
    fatol: float = 1e-2
    simplex_rel_size: float = 0.1
    warm_start_blend: float = 0.7
    improve_tol: float = 0.005

@dataclass
class Result:
    best_x: Optional[np.ndarray]
    best_J: Optional[float]
    best_metrics: Optional[Dict[str, float]]
    n_iter: int
    n_eval: int
    elapsed_s: float
    status: str
    message: str


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class NMRunner:
    """Anytime Nelderâ€“Mead with time-budget early stop and progress capture."""

    def __init__(self) -> None:
        self._best_x: Optional[np.ndarray] = None
        self._best_J: float = float("inf")
        self._best_metrics: Dict[str, float] = {}
        self._n_eval: int = 0
        self._n_iter: int = 0
        self._stopped_by_budget: bool = False
        self._last_msg: str = ""

    # --- public API ---------------------------------------------------------

    def solve(
        self,
        *,
        x0: np.ndarray,
        time_budget_s: float,
        cfg: NMConfig,
        objective: Optional[Objective] = None,
        seed_station: Optional[object] = None,
        active_setpoints: Optional[Dict[str, float]] = None,
        objective_kwargs: Optional[Dict[str, object]] = None,
        progress_cb: Optional[Callable[[Dict[str, object]], None]] = None,
        prev_best_x: Optional[np.ndarray] = None,
    ) -> Result:
        """Run an anytime Nelderâ€“Mead under a wall-clock budget.

        Either supply a prepared `objective` (Option A), or provide
        `seed_station`, `active_setpoints`, and `objective_kwargs`
        (Option B) so we can build+prepare it here.
        """
        t_start = time.time()
        self._reset()

        # Build objective if not provided (Option B)
        if objective is None:
            if seed_station is None or active_setpoints is None or objective_kwargs is None:
                raise RuntimeError("NMRunner: objective not provided; seed_station, active_setpoints, and objective_kwargs are required.")
            objective = Objective(**objective_kwargs)  # type: ignore[call-arg]
            objective.set_seed(seed_station, active_setpoints)
            objective.prepare_t0()
        obj = objective  # rename

        # Hook into per-evaluation progress for GUI fidelity
        def _eval_hook(payload: Dict[str, object]) -> None:
            try:
                x = np.asarray(payload.get("x"))
                J = float(payload.get("J"))
                metrics = dict(payload.get("metrics") or {})
                is_best = bool(payload.get("is_best", False))
                if is_best and (J < self._best_J):
                    self._best_J = J
                    self._best_x = x.copy()
                    self._best_metrics = dict(metrics)
            except Exception:
                pass
            # forward for GUI/prints
            if progress_cb is not None:
                try:
                    progress_cb({**payload,
                                 "stage": "eval",
                                 "n_eval": self._n_eval,
                                 "n_iter": self._n_iter})
                except Exception:
                    pass

        # Precompute initial simplex if bounds available
        init_simplex = None
        bounds = None
        try:
            bounds = objective_kwargs.get("bounds") if objective_kwargs is not None else None
            if bounds is None and hasattr(obj, "bounds"):
                bounds = getattr(obj, "bounds")
        except Exception:
            bounds = None
        if bounds is not None:
            try:
                lb = np.array([bounds['Q1'][0], bounds['Q2'][0], bounds['Q4'][0], bounds['SI'][0]], dtype=float)
                ub = np.array([bounds['Q1'][1], bounds['Q2'][1], bounds['Q4'][1], bounds['SI'][1]], dtype=float)
                span = np.maximum(ub - lb, 1e-9)
                step = np.maximum(cfg.simplex_rel_size * span, 1e-6)
                init_simplex = [x0]
                for i in range(len(x0)):
                    v = x0.copy()
                    v[i] = np.clip(v[i] + step[i], lb[i], ub[i])
                    init_simplex.append(v)
                init_simplex = np.array(init_simplex, dtype=float)
            except Exception:
                init_simplex = None

        # Objective wrapper with time budget
        deadline = t_start + max(0.05, float(time_budget_s))

        def fun(z: np.ndarray) -> float:
            if time.time() > deadline:
                self._stopped_by_budget = True
                raise TimeoutError("NM time budget exhausted")

            try:
                J, m = obj.evaluate(np.asarray(z, dtype=float))
                if not np.isfinite(J):
                    J = 1e12  # penalize invalid objective
            except Exception as e:
                J, m = 1e12, {"error": str(e)}  # penalize failures

            self._n_eval += 1

            # progress callback
            if progress_cb is not None:
                try:
                    progress_cb({
                        "stage": "eval",
                        "x": np.asarray(z, dtype=float),
                        "J": float(J),
                        "metrics": dict(m or {}),
                        "is_best": bool(J < self._best_J),
                        "n_eval": self._n_eval,
                        "n_iter": self._n_iter,
                        "elapsed_s": time.time() - t_start,
                        "best_J": (self._best_J if np.isfinite(self._best_J) else None),
                        "best_x": (self._best_x.copy() if self._best_x is not None else None),
                    })
                except Exception:
                    pass

            # best-so-far & early-stop
            if J < self._best_J:
                dJ = (self._best_J - J) if np.isfinite(self._best_J) else None
                self._best_J = float(J)
                self._best_x = np.asarray(z, dtype=float).copy()
                self._best_metrics = dict(m or {})
                if dJ is not None and cfg.improve_tol > 0.0 and dJ < cfg.improve_tol:
                    self._stopped_by_early = True
                    raise StopIteration("EarlyStop: improvement < tol")

            return float(J)

        # Iteration callback to capture nit and optional extra info
        def callback(xk: np.ndarray) -> None:
            self._n_iter += 1
            if progress_cb is not None:
                try:
                    progress_cb({
                        "stage": "iter",
                        "n_iter": self._n_iter,
                        "n_eval": self._n_eval,
                        "elapsed_s": time.time() - t_start,
                        "best_J": (self._best_J if np.isfinite(self._best_J) else None),
                        "best_x": (self._best_x.copy() if self._best_x is not None else None),
                    })
                except Exception:
                    pass

        # Run SciPy or fallback
        if _spo is not None:
            try:
                options = {"maxiter": int(cfg.maxiter), "xatol": float(cfg.xatol), "fatol": float(cfg.fatol), "adaptive": True}
                if init_simplex is not None:
                    options["initial_simplex"] = init_simplex
                res = _spo.minimize(fun, x0, method="Nelder-Mead", callback=callback, options=options)
                x_fin = np.asarray(res.x, dtype=float)
                J_fin = float(res.fun)
                status = "Converged" if res.success else ("Stopped" if self._stopped_by_budget else "Failed")
                msg = str(res.message)
            except StopIteration as e:
                # Early-stop triggered (Î”J < improve_tol)
                x_fin = self._best_x if self._best_x is not None else np.asarray(x0, dtype=float)
                J_fin = self._best_J if math.isfinite(self._best_J) else float("inf")
                status = "Success"
                msg = str(e)
            except TimeoutError as e:
                # Time budget exceeded mid-iteration
                x_fin = self._best_x if self._best_x is not None else np.asarray(x0, dtype=float)
                J_fin = self._best_J if math.isfinite(self._best_J) else float("inf")
                status = "TimeBoxed"
                msg = str(e)
            except Exception as e:
                x_fin = self._best_x if self._best_x is not None else np.asarray(x0, dtype=float)
                J_fin = self._best_J if math.isfinite(self._best_J) else float("inf")
                status = "Error"
                msg = f"SciPy error: {e}"
        else:
            # Fallback: tiny NM (very basic), still honors time budget via fun()
            x_fin, J_fin, status, msg = self._tiny_nm(x0, fun, deadline)

        elapsed = time.time() - t_start
        # Finalize best based on objective progress if available
        try:
            prog = obj.get_progress()  # type: ignore[attr-defined]
            if prog.get("best_x") is not None:
                self._best_x = np.asarray(prog["best_x"], dtype=float)
                self._best_J = float(prog["best_J"]) if prog.get("best_J") is not None else self._best_J
                self._best_metrics = dict(prog.get("best_metrics") or self._best_metrics)
                self._n_eval = int(prog.get("n_eval") or self._n_eval)
        except Exception:
            pass

        return Result(
            best_x=(self._best_x.copy() if self._best_x is not None else None),
            best_J=(float(self._best_J) if math.isfinite(self._best_J) else None),
            best_metrics=(dict(self._best_metrics) if self._best_metrics else None),
            n_iter=int(self._n_iter),
            n_eval=int(self._n_eval),
            elapsed_s=float(elapsed),
            status=status,
            message=msg,
        )

    # --- helpers -------------------------------------------------------------

    def _reset(self) -> None:
        self._best_x = None
        self._best_J = float("inf")
        self._best_metrics = {}
        self._n_eval = 0
        self._n_iter = 0
        self._stopped_by_budget = False
        self._stopped_by_early = False   # â† add
        self._last_msg = ""

    def _tiny_nm(self, x0: np.ndarray, fun: Callable[[np.ndarray], float], deadline: float):
        # ultra-minimal Nelderâ€“Mead; relies on `fun` to raise TimeoutError when over budget
        n = len(x0)
        simplex = [np.asarray(x0, dtype=float)]
        step = np.maximum(np.abs(x0) * 0.05, 1e-3)
        for i in range(n):
            v = simplex[0].copy(); v[i] += step[i]; simplex.append(v)
        vals = []
        for v in simplex:
            vals.append(fun(v))
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        while time.time() <= deadline and len(vals) < 10000:
            idx = np.argsort(vals); simplex = [simplex[i] for i in idx]; vals = [vals[i] for i in idx]
            xbar = np.mean(simplex[:-1], axis=0)
            xr = xbar + alpha * (xbar - simplex[-1]); Jr = fun(xr)
            if Jr < vals[0]:
                xe = xbar + gamma * (xr - xbar); Je = fun(xe)
                if Je < Jr:
                    simplex[-1], vals[-1] = xe, Je
                else:
                    simplex[-1], vals[-1] = xr, Jr
            elif Jr < vals[-2]:
                simplex[-1], vals[-1] = xr, Jr
            else:
                xc = xbar + rho * (simplex[-1] - xbar); Jc = fun(xc)
                if Jc < vals[-1]:
                    simplex[-1], vals[-1] = xc, Jc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        vals[i] = fun(simplex[i])
        best_i = int(np.argmin(vals))
        return simplex[best_i], float(vals[best_i]), ("TimeBoxed" if time.time() > deadline else "Done"), "tiny-NM"
