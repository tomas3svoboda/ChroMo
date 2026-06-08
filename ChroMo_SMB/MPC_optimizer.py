# optimizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any
from threading import Thread, Event
import time
import math
import numpy as np
import os

from scipy.optimize import minimize
from objective_function import SMBObjectiveFunction, ObjectiveWeights


# -------------------- Config & Status --------------------

@dataclass
class MPCOptimizerConfig:
    horizon_periods: int
    weights: ObjectiveWeights
    bounds: Optional[Dict[str, tuple]] = None          # {'F':(lo,hi),'D':(...),'Q4':(...)}
    maxiter: int = 120
    xatol: float = 1e-3                                # used by Nelder–Mead / Powell
    fatol: float = 1e-3
    method: str = "powell"                             # "powell" | "nelder-mead" | "grid"
    grid_points_per_dim: int = 10                      # for "grid" brute-force search
    grid_margin_frac: float = 0.02                     # trim 2% off each bound to avoid pathological edges
    brute_log_enabled: bool = True                     # log grid/brute evaluations to a file
    brute_log_dir: str = "logs"                       # directory for logs
    brute_log_prefix: str = "bruteforce_evals"         # file prefix

    # time guarding
    safety_margin_s: float = 0.5
    hard_timeout_s: Optional[float] = None

    # progress updates
    status_period_s: float = 0.25

    # early-plateau kill switch (stop if best improves < plateau_delta over plateau_window evals)
    plateau_delta: float = 0.05
    plateau_window: int = 15

    # initial simplex step sizes (only for NM)
    step_F: float = 0.5
    step_D: float = 5.0
    step_Q: float = 3.0


@dataclass
class MPCOptimizerStatus:
    running: bool = False
    message: str = "idle"
    nevals: int = 0
    best_x: Optional[List[float]] = None
    best_cost: Optional[float] = None
    best_metrics: Optional[Dict[str, float]] = None
    elapsed_s: float = 0.0


class EarlyStop(Exception):
    pass


# -------------------- Optimizer --------------------

class MPCOptimizer:
    """
    Async, time-budgeted black-box optimization around SMBObjectiveFunction.

    You pass in for each cycle:
      - smb_template: CURRENT template SMBStation (your 'memory' state). NOT modified here.
      - x0: warm-start vector [F, D, Q4]
      - config: MPCOptimizerConfig
      - get_time_remaining_s(): callable -> float seconds until the switch (for rolling MPC).
        For pre-start, pass lambda: 1e9 or similar.

    Optional:
      - state_estimator_hook(smb_template) -> smb_template':
          Runs before the objective is built (for nudging/EnKF later).
      - on_status(MPCOptimizerStatus): GUI progress callback (periodic).
      - on_done(MPCOptimizerStatus): GUI completion callback (final).
    """
    def _write_grid_log(self, rows, cfg):
        """Write grid/brute evaluations to .xls (if xlwt), else .xlsx (if openpyxl), else .csv."""
        try:
            if not rows:
                return None
            try:
                import pandas as pd
            except Exception as e:
                print(f"[MPC][warn] pandas not available for brute-force logging: {e}")
                return None
            # Ensure directory
            log_dir = getattr(cfg, "brute_log_dir", "logs") or "logs"
            os.makedirs(log_dir, exist_ok=True)
            # Choose file extension/engine
            ext = ".csv"
            engine = None
            try:
                import xlwt  # noqa: F401
                ext = ".xls"
                engine = "xlwt"
            except Exception:
                try:
                    import openpyxl  # noqa: F401
                    ext = ".xlsx"
                    engine = "openpyxl"
                except Exception:
                    pass
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"{getattr(cfg,'brute_log_prefix','bruteforce_evals')}_{ts}{ext}"
            fpath = os.path.join(log_dir, fname)
            # Build DataFrame
            # rows: [eval_idx, F, D, Q4, J, elapsed_s]
            df = pd.DataFrame(rows, columns=["eval_idx","F","D","Q4","J","elapsed_s"])
            if engine:
                df.to_excel(fpath, index=False, engine=engine)
            else:
                df.to_csv(fpath, index=False)
            print(f"[MPC][log] Brute/grid log saved to: {fpath}")
            return fpath
        except Exception as e:
            print(f"[MPC][warn] Failed to write brute/grid log: {e}")
            return None


    def __init__(
        self,
        fixed_extract: float,
        *,
        man_name: str = "Man",
        gal_name: str = "Gal",
        state_estimator_hook: Optional[Callable[[Any], Any]] = None,
        on_status: Optional[Callable[[MPCOptimizerStatus], None]] = None,
        on_done: Optional[Callable[[MPCOptimizerStatus], None]] = None,
    ):
        self.E = float(fixed_extract)
        self.man = man_name
        self.gal = gal_name
        self._state_hook = state_estimator_hook
        self._on_status = on_status
        self._on_done = on_done

        self._thr: Optional[Thread] = None
        self._cancel = Event()
        self._status = MPCOptimizerStatus()

    # ---- lifecycle ----

    def start_async(
        self,
        smb_template,
        x0: List[float],
        cfg: MPCOptimizerConfig,
        get_time_remaining_s: Callable[[], float],
    ):
        """Start a new optimization in a thread. Cancels any previous run."""
        self.cancel()
        self._cancel.clear()
        self._status = MPCOptimizerStatus(running=True, message="starting")
        self._emit_status()

        args = (smb_template, x0, cfg, get_time_remaining_s)
        self._thr = Thread(target=self._run, args=args, daemon=True)
        self._thr.start()

    def cancel(self):
        if self._thr and self._thr.is_alive():
            self._cancel.set()

    def join(self, timeout: Optional[float] = None):
        if self._thr:
            self._thr.join(timeout)

    def is_running(self) -> bool:
        return self._status.running

    def status(self) -> MPCOptimizerStatus:
        return self._status

    # ---- core ----

    def _run(self, smb_template, x0: List[float], cfg: MPCOptimizerConfig, get_time_remaining_s):
        t0 = time.time()
        try:
            # apply (optional) state correction BEFORE building objective
            template = smb_template
            if self._state_hook is not None:
                try:
                    template = self._state_hook(template)
                except Exception as e:
                    # keep going without correction
                    self._status.message = f"state_hook error ignored: {e}"

            budget_s = self._compute_budget(get_time_remaining_s, cfg)
            # print(f"[MPC][debug] remaining={get_time_remaining_s():.3f}, budget={budget_s:.3f}, hard={cfg.hard_timeout_s}")
            if budget_s <= 0:
                raise EarlyStop("No time left in this period")

            # Build objective bound to THIS template (it deep-copies each eval)
            obj = SMBObjectiveFunction(
                smb_current=template,
                horizon_periods=cfg.horizon_periods,
                fixed_extract=self.E,
                man_name=self.man,
                gal_name=self.gal,
                weights=cfg.weights,
                bounds=cfg.bounds,
            )

            best = {"x": None, "J": math.inf, "metrics": None}
            nevals = {"n": 0}
            last_emit = [0.0]
            hist: List[float] = []  # best J history for plateau detection

            def eval_guarded(x):
                # cancel?
                if self._cancel.is_set():
                    print("[MPC][debug] Cancelled before eval.")
                    raise EarlyStop("Canceled")

                # time guard
                if self._time_exceeded(t0, budget_s, cfg.hard_timeout_s):
                    print(f"[MPC][debug] Time budget exceeded after {nevals['n']} evals "
                          f"(elapsed={time.time() - t0:.2f}s, budget={budget_s:.2f}s)")
                    raise EarlyStop("Time budget exceeded")

                print(f"[MPC][debug] Eval {nevals['n'] + 1}, x={np.asarray(x)}")
                t_eval_start = time.time()
                try:
                    J = obj(x)  # heavy call
                except Exception as e:
                    print(f"[MPC][warn] Objective error for x={np.asarray(x)}: {e}. Treating as infeasible.")
                    J = math.inf
                t_eval_end = time.time()
                print(f"[MPC][debug] Eval {nevals['n'] + 1} took {t_eval_end - t_eval_start:.2f}s → J={J}")

                nevals["n"] += 1

                if J < best["J"]:
                    best["J"] = J
                    best["x"] = list(x)
                    # compute metrics once for GUI (separate call OK)
                    best["metrics"] = obj.evaluate_sequence(x)

                    # plateau tracking (disabled for grid/brute methods)
                    if (cfg.method or "").lower() not in ("grid","brute","bruteforce","grid-search"):
                        hist.append(J)
                        if len(hist) > cfg.plateau_window:
                            hist.pop(0)
                        if len(hist) == cfg.plateau_window:
                            if abs(hist[0] - min(hist)) < cfg.plateau_delta:
                                raise EarlyStop(f"Plateau: ΔJ<{cfg.plateau_delta} over {cfg.plateau_window} evals")

                # emit status periodically
                now = time.time()
                if now - last_emit[0] >= cfg.status_period_s:
                    self._status = MPCOptimizerStatus(
                        running=True,
                        message="running",
                        nevals=nevals["n"],
                        best_x=best["x"],
                        best_cost=best["J"],
                        best_metrics=best["metrics"],
                        elapsed_s=now - t0,
                    )
                    self._emit_status()
                    last_emit[0] = now

                return J

            self._status.message = "optimizing"
            self._emit_status()

            # Prepare bounds list for SciPy (Powell supports bounds directly)
            scipy_bounds = None
            if cfg.bounds:
                loF, hiF = cfg.bounds.get('F', (-np.inf, np.inf))
                loD, hiD = cfg.bounds.get('D', (-np.inf, np.inf))
                loQ, hiQ = cfg.bounds.get('Q4', (-np.inf, np.inf))
                scipy_bounds = [(loF, hiF), (loD, hiD), (loQ, hiQ)]

            method = (cfg.method or "powell").lower()
            if method in ("grid", "brute", "bruteforce", "grid-search"):
                # Brute-force uniform grid over bounds for F, D, Q4
                if not cfg.bounds:
                    raise EarlyStop("Grid search requires finite bounds for F, D, Q4.")
                loF, hiF = cfg.bounds.get('F', (float('nan'), float('nan')))
                loD, hiD = cfg.bounds.get('D', (float('nan'), float('nan')))
                loQ, hiQ = cfg.bounds.get('Q4', (float('nan'), float('nan')))
                for name, lo, hi in (('F', loF, hiF), ('D', loD, hiD), ('Q4', loQ, hiQ)):
                    if not (np.isfinite(lo) and np.isfinite(hi)):
                        raise EarlyStop(f"Grid search requires finite bounds for {name}.")
                    if hi < lo:
                        raise EarlyStop(f"Bounds for {name} are inverted: ({lo}, {hi}).")
                n = max(1, int(getattr(cfg, "grid_points_per_dim", 10)))
                F_vals = np.linspace(loF, hiF, n)
                D_vals = np.linspace(loD, hiD, n)
                Q_vals = np.linspace(loQ, hiQ, n)
                grid_rows = []
                stopped = False
                try:
                    for F in F_vals:
                        for D in D_vals:
                            for Q4 in Q_vals:
                                t0_eval = time.time()
                                try:
                                    J = eval_guarded([float(F), float(D), float(Q4)])
                                except EarlyStop as e:
                                    # record the last attempted point as infeasible, then stop
                                    grid_rows.append([nevals['n']+1, float(F), float(D), float(Q4), float('inf'), time.time()-t0_eval])
                                    stopped = True
                                    raise
                                else:
                                    # eval_guarded increments nevals['n'] after evaluation
                                    grid_rows.append([nevals['n'], float(F), float(D), float(Q4), float(J), time.time()-t0_eval])
                finally:
                    if getattr(cfg, 'brute_log_enabled', True):
                        self._write_grid_log(grid_rows, cfg)
                res = None  # force use of best-so-far
            elif method == "powell":


                # Powell is efficient in low dimensions and respects bounds
                res = minimize(
                    eval_guarded, x0, method="Powell", bounds=scipy_bounds,
                    options={"maxiter": cfg.maxiter, "xtol": cfg.xatol, "ftol": cfg.fatol, "disp": False},
                )
            else:
                # Nelder–Mead with scaled initial simplex
                init = np.asarray(x0, dtype=float)
                simplex = np.vstack([
                    init,
                    init + [cfg.step_F, 0.0, 0.0],
                    init + [0.0, cfg.step_D, 0.0],
                    init + [0.0, 0.0, cfg.step_Q],
                ])
                if scipy_bounds is not None:
                    lb = np.array([b[0] for b in scipy_bounds], float)
                    ub = np.array([b[1] for b in scipy_bounds], float)
                    simplex = np.clip(simplex, lb, ub)

                res = minimize(
                    eval_guarded, x0, method="Nelder-Mead",
                    options={
                        "maxiter": cfg.maxiter, "xatol": cfg.xatol, "fatol": cfg.fatol,
                        "initial_simplex": simplex, "disp": False
                    }
                )

            # prefer SciPy result if success; else best-so-far
            if res is not None and res.success and res.x is not None:
                x_best = list(res.x)
                J_best = float(res.fun)
                metrics = obj.evaluate_sequence(x_best)
            else:
                x_best = best["x"] if best["x"] is not None else x0
                J_best = best["J"] if best["x"] is not None else eval_guarded(x_best)
                metrics = best["metrics"] if best["metrics"] is not None else obj.evaluate_sequence(x_best)

            self._status = MPCOptimizerStatus(
                running=False,
                message="done",
                nevals=nevals["n"],
                best_x=x_best,
                best_cost=J_best,
                best_metrics=metrics,
                elapsed_s=time.time() - t0,
            )
            self._emit_status(final=True)

        except EarlyStop as e:
            self._status = MPCOptimizerStatus(
                running=False,
                message=str(e),
                nevals=self._status.nevals,
                best_x=self._status.best_x,
                best_cost=self._status.best_cost,
                best_metrics=self._status.best_metrics,
                elapsed_s=time.time() - t0,
            )
            self._emit_status(final=True)
        except Exception as e:
            self._status = MPCOptimizerStatus(
                running=False,
                message=f"error: {e}",
                nevals=self._status.nevals,
                best_x=self._status.best_x,
                best_cost=self._status.best_cost,
                best_metrics=self._status.best_metrics,
                elapsed_s=time.time() - t0,
            )
            self._emit_status(final=True)

    # ---- helpers ----

    def _compute_budget(self, get_time_remaining_s, cfg: MPCOptimizerConfig) -> float:
        try:
            remaining = max(0.0, float(get_time_remaining_s()))
        except Exception:
            remaining = 0.0
        budget = remaining - cfg.safety_margin_s
        if cfg.hard_timeout_s is not None:
            budget = min(budget, cfg.hard_timeout_s)
        return budget

    @staticmethod
    def _time_exceeded(t0, budget_s, hard_timeout_s) -> bool:
        elapsed = time.time() - t0
        if elapsed > budget_s:
            return True
        if hard_timeout_s is not None and elapsed > hard_timeout_s:
            return True
        return False

    def _emit_status(self, final: bool = False):
        if self._on_status and not final:
            try:
                self._on_status(self._status)
            except Exception:
                pass
        if self._on_done and final:
            try:
                self._on_done(self._status)
            except Exception:
                pass
