"""
mpc_orchestrator.py synchronized MPC lifecycle manager
===========================================================

Purpose
-------
Coordinates the MPC cycle around SMB switch boundaries while respecting your
OPCÂ UA interface semantics:
  ACTIVE* nodes are read only mirrors of what drives the plant.
  NEXT (backwardâ€‘compatible) nodes are writable setpoints.
  In AUTOMATIC mode the plant/server applies NEXT â†’ ACTIVE at switch time.

Dependencies
------------
 `opcua_client.SMB_OPCUAClient` (already in your repo) â€” used for snapshot IO.
 `mpc_objective.Objective` â€” must provide `set_seed(station)` and
  `evaluate(x) -> (J, metrics)` APIs and accept horizon/weights/bounds.
 `mpc_optimizer.NMRunner` â€” must provide
  `solve(x0, objective, time_budget_s, cfg) -> Result`.

Note: This module does **not** implement the Objective or the optimizer; it only
coordinates them. If those imports are missing, a clear RuntimeError is raised.
"""

from __future__ import annotations

import copy
import threading
import os, glob, pickle
from datetime import datetime
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import json

from ChroMo_SMB.SMB_PlantSim.opcua_client import SMB_OPCUAClient
from mpc_objective import Objective

try:
    from mpc_optimizer import NMRunner, NMConfig
except Exception:  # pragma: no cover
    NMRunner = None  # type: ignore
    @dataclass
    class NMConfig:  # minimalist placeholder for type hints
        maxiter: int = 400
        xatol: float = 1e-2
        fatol: float = 1e-2
        simplex_rel_size: float = 0.2
        warm_start_blend: float = 0.7

# ---------------------------------------------------------------------------
# Config & State
# ---------------------------------------------------------------------------
@dataclass
class Bounds:
    Q1: Tuple[float, float]
    Q2: Tuple[float, float]
    Q4: Tuple[float, float]
    SI: Tuple[float, float]

@dataclass
class Weights:
    w_dil_ex: float = 1.5
    w_dil_ra: float = 0.5
    w_pur_ex: float = 3.0
    w_pur_ra: float = 1.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.w_dil_ex, self.w_dil_ra, self.w_pur_ex, self.w_pur_ra)

@dataclass
class Policy:
    horizon_si: int = 8            # number of upcoming switch intervals
    guard_time_s: float = 20      # finish before switch by this guard
    improvement_epsilon: float = 0.0  # minimum Î”J (baseline âˆ’ best) to stage
    rate_limits: Optional[Dict[str, float]] = None  # optional: max Î” per SI


# --- OrchestratorStatus ---
@dataclass
class OrchestratorStatus:
    running: bool = False
    mode: str = "manual"                 # 'manual' or 'automatic' (from plant)
    is_running: bool = False             # NEW: plant run state (OPC IsRunning)
    opc_connected: bool = False
    opc_latency_ms: Optional[float] = None
    opc_last_seen_wall: Optional[float] = None
    last_error: str = ""

    time_to_switch_s: float = 0.0        # kept, now driven by OPC SwitchCountdow
    last_cycle_iter: int = 0
    last_cycle_elapsed_s: float = 0.0
    last_cycle_status: str = "Idle"   # Idle/Solving/Applied/Skipped/Failed
    last_cycle_message: str = ""

    baseline_J: Optional[float] = None
    best_J: Optional[float] = None
    best_metrics: Dict[str, float] = field(default_factory=dict)
    best_x: Optional[Tuple[float, float, float, float]] = None  # (Q1,Q2,Q4,SI)

    active_setpoints: Dict[str, float] = field(default_factory=dict)
    next_setpoints: Dict[str, float] = field(default_factory=dict)

    speed_factor: float = 1.0
    handover_dir: Optional[str] = None
    last_snapshot_index: Optional[int] = None
    last_snapshot_created_utc: Optional[str] = None

    kpi_pur_ex: Optional[float] = None
    kpi_pur_ra: Optional[float] = None
    kpi_dil_ex: Optional[float] = None
    kpi_dil_ra: Optional[float] = None

# ---------------------------------------------------------------------------
# Snapshot Watcher
# ---------------------------------------------------------------------------

class SnapshotWatcher:
    """Poll-based watcher for new state_s{k}.pkl files; calls cb(idx, path) on strictly newer index."""
    def __init__(self, folder: str, get_latest_path, parse_idx, cb, poll_s: float = 0.2):
        self.folder = folder
        self.get_latest_path = get_latest_path
        self.parse_idx = parse_idx
        self.cb = cb
        self.poll_s = max(0.1, float(poll_s))
        self._last_idx = None
        self._stop = threading.Event()
        self._thr = None

    def start(self, last_seen: int | None):
        self._last_idx = last_seen
        if self._thr and self._thr.is_alive():
            return
        self._thr = threading.Thread(target=self._loop, name="SnapshotWatcher", daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr and self._thr.is_alive():
            self._thr.join(timeout=1.0)

    def _loop(self):
        while not self._stop.is_set():
            p = self.get_latest_path()
            if p:
                idx = self.parse_idx(p)
                if idx is not None:
                    if self._last_idx is None or idx > self._last_idx:
                        self._last_idx = idx
                        try:
                            self.cb(idx, p)
                        except Exception:
                            pass
            self._stop.wait(self.poll_s)

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MPCOrchestrator:
    """SIâ€‘synchronized coordinator for MPC.

    Public methods:
      â€¢ start()/stop()
      â€¢ set_weights()/set_bounds()/set_policy()/set_nm_config()
      â€¢ subscribe(cb): register GUI/data observers; cb(state: OrchestratorStatus)
    """

    def __init__(
        self,
        *,
        opc: SMB_OPCUAClient,
        # defaults updated:
        weights: Weights = Weights(w_dil_ex=1.5, w_dil_ra=0.5, w_pur_ex=3.0, w_pur_ra=1.0),
        bounds: Bounds = Bounds((10.0, 500.0), (10.0, 500.0), (10.0, 500.0), (100.0, 5400.0)),
        policy: Policy = Policy(horizon_si=8),
        nm_cfg: NMConfig = NMConfig(maxiter=400, xatol=5e-3, fatol=5e-3, simplex_rel_size=0.2, improve_tol=0.0),
        # new (explicit) feed concentrations for dilution (%):
        feed_conc_A: float = 7.27,
        feed_conc_B: float = 3.42,
        # optional: a sane per-cycle time budget (seconds) for the optimizer:
        time_budget_s: float = 780.0,
        handover_dir: Optional[str] = None,
    ) -> None:
        if Objective is None or NMRunner is None:
            raise RuntimeError(
                "mpc_objective.Objective and mpc_optimizer.NMRunner must be available before using MPCOrchestrator"
            )

        self.opc = opc
        self._watcher = None

        self.weights = weights
        self.bounds = bounds
        self.policy = policy
        self.nm_cfg = nm_cfg
        self.feed_conc_A = float(feed_conc_A)
        self.feed_conc_B = float(feed_conc_B)
        self.time_budget_s = float(time_budget_s)

        # Observers (e.g., GUI); push OrchestratorStatus snapshots periodically
        self._observers: List[Callable[[OrchestratorStatus], None]] = []
        self._status = OrchestratorStatus()

        # Handover folder with state_s{k}.pkl from estimator
        self.handover_dir = handover_dir or os.path.join(os.getcwd(), r"C:\Users\z004d8nt\PycharmProjects\DisertationTSV\ChroMo_SMB\SMB_SE\handover")
        os.makedirs(self.handover_dir, exist_ok=True)
        self._last_snapshot_idx: Optional[int] = None
        self._status.handover_dir = self.handover_dir

        self._seed_station = None
        self._seed_meta = None

        # Internals for cycle timing & worker
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_countdown: Optional[float] = None
        self._incumbent_x0: Optional[np.ndarray] = None

        # Background notifier for observers (GUI)
        self._notifier_thr = threading.Thread(target=self._notifier_loop, daemon=True)
        self._notifier_thr.start()

        # Lightweight OPC UA heartbeat to track latency/health
        self._hb_thr = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thr.start()

        self._last_is_running: Optional[bool] = None
        self._last_idle_msg_emitted: bool = False

        self._last_opc_countdown: Optional[float] = None        # OPC countdown
        self._min_wrap_jump: float = 0.5                        # seconds (hysteresis)

        self.eval_dt: float = 1.0  # default evaluation dt [s]

    # ---------------------- Public API ----------------------

    def subscribe(self, cb: Callable[[OrchestratorStatus], None]) -> None:
        self._observers.append(cb)

    def set_weights(self, w: Weights) -> None:
        self.weights = w

    def set_bounds(self, b: Bounds) -> None:
        self.bounds = b

    def set_policy(self, p: Policy) -> None:
        self.policy = p

    def set_nm_config(self, cfg: NMConfig) -> None:
        self.nm_cfg = cfg

    def start(self) -> None:
        self._stop.clear()
        self._status.running = True
        try:
            self.opc.connect()
            self._status.opc_connected = True

            # Read one OPC snapshot for initial status
            snap = self.opc.read_snapshot()
            self._ingest_snapshot(snap)
            print(f"[MPC] OPC connected. Mode={self._status.mode}  IsRunning={self._status.is_running}  SF={self._status.speed_factor:g}")

            # Try to load the latest seed if any (may be None until s1 arrives)
            self._get_fresh_seed_or_none()

            # Start watching handover folder NOW (donâ€™t wait for a running transition)
            def _on_new_snapshot(idx, path):
                self._get_fresh_seed_or_none()
                if self._status.running and not (self._worker and self._worker.is_alive()):
                    self._on_switch_event()

            if not self._watcher:
                self._watcher = SnapshotWatcher(
                    folder=self.handover_dir,
                    get_latest_path=self._latest_snapshot_path,
                    parse_idx=self._parse_snapshot_index,
                    cb=_on_new_snapshot,
                    poll_s=0.1,
                )
                self._watcher.start(last_seen=self._last_snapshot_idx)
                print(f"[MPC] Snapshot watcher started â†’ {self.handover_dir}")

            # If plant is already running AND a seed already exists, start immediately.
            if self._status.is_running and (self._seed_station is not None):
                self._on_switch_event()
                self._status.last_cycle_status = "Idle"
                self._status.last_cycle_message = "Started immediately (running & seed present)."
            elif self._status.is_running and (self._seed_station is None):
                self._status.last_cycle_status = "Idle"
                self._status.last_cycle_message = "Waiting for first snapshot (s1+)."
            else:
                self._status.last_cycle_status = "Idle"
                self._status.last_cycle_message = "Plant STOPPED. MPC waiting."

            self._notify()

        except Exception as e:
            self._status.opc_connected = False
            self._status.last_error = f"OPC connect/start failed: {e}"
            self._notify()
            raise

    def stop(self) -> None:
        self._stop.set()
        self._status.running = False
        if self._watcher:
            try: self._watcher.stop()
            except Exception: pass
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def set_eval_dt(self, dt: float) -> None:
        """Set rollout evaluation step for the Objective (seconds)."""
        try:
            dt = float(dt)
        except Exception:
            return
        self.eval_dt = max(0.1, dt)  # safety floor
        print(f"[MPC] eval_dt set to {self.eval_dt:.3g}s")
        # (optional) expose to status for GUI display
        if hasattr(self, "_status"):
            setattr(self._status, "eval_dt", self.eval_dt)
            self._notify()

    # ---------------------- Core perâ€‘cycle routine ----------------------

    def _on_switch_event(self) -> None:
        # Avoid overlapping cycles
        if self._worker and self._worker.is_alive():
            # Let the current worker finish; we will apply its incumbent this switch
            return
        self._worker = threading.Thread(target=self._solve_cycle, daemon=True)
        self._worker.start()

    def _solve_cycle(self) -> None:
        # Refresh plant snapshot/mode
        try:
            snap = self.opc.read_snapshot()
            self._ingest_snapshot(snap)
        except Exception as e:
            self._status.last_error = f"OPC read failed: {e}"
            self._status.last_cycle_status = "Failed"
            self._status.last_cycle_message = "Snapshot read failed."
            self._notify(); return

        if not self._status.is_running:
            self._status.last_cycle_status = "Skipped"
            self._status.last_cycle_message = "Skipped: Plant STOPPED."
            self._notify(); return

        # Build ACTIVE vector for baseline
        try:
            self._status.mode = str(snap.get("Mode", "manual")).lower()
            self._status.active_setpoints = {
                "F":  float(snap.get("ActiveFeed", 0.0)),
                "Q1": float(snap.get("ActiveQ1", 0.0)),
                "Q2": float(snap.get("ActiveQ2", 0.0)),
                "Q4": float(snap.get("ActiveQ4", 0.0)),
                "SI": float(snap.get("ActiveSwitchInterval", 0.0)),
            }
        except Exception as e:
            self._status.last_error = f"OPC read failed: {e}"
            self._status.last_cycle_status = "Failed"
            return

        x_active = np.array([
            self._status.active_setpoints["Q1"],
            self._status.active_setpoints["Q2"],
            self._status.active_setpoints["Q4"],
            self._status.active_setpoints["SI"],
        ], dtype=float)

        # Speed-aware time budget: countdown (sim) / SF  âˆ’ guard âˆ’ latency
        try:
            sc_sim = float(snap.get("SwitchCountdown", 0.0))
        except Exception:
            sc_sim = 0.0
        sf = max(self._status.speed_factor, 1e-9)
        opc_lat = max(0.0, (self._status.opc_latency_ms or 0.0) / 1000.0)
        wall_budget = max(0.0, sc_sim / sf - float(self.policy.guard_time_s) - opc_lat)
        self._status.time_to_switch_s = sc_sim  # store sim countdown for GUI
        time_budget = max(0.05, wall_budget)
        print(f"[MPC] Cycle start: SC(sim)={sc_sim:.2f}s  SF={sf:g}  budgetâ‰ˆ{time_budget:.2f}s  mode={self._status.mode}")

        # Prepare Objective (seed taken from latest snapshot if loaded)
        obj = Objective(
            horizon_si=int(self.policy.horizon_si),
            weights=self.weights.as_tuple(),
            bounds={"Q1": self.bounds.Q1, "Q2": self.bounds.Q2, "Q4": self.bounds.Q4, "SI": self.bounds.SI},
            feed_conc_A=self.feed_conc_A, feed_conc_B=self.feed_conc_B,
            burn_in_si=0, score_si=None, taper=False,
            cache_enable=True, eval_dt=self.eval_dt,
        )

        # Load newest seed (may start at s1; be robust to write timing)
        seed = self._get_fresh_seed_or_none()
        if self._seed_station is None:
            # watcher may have fired milliseconds before the file/pointer is fully visible
            for _ in range(4):                      # retry up to ~0.4 s total
                time.sleep(0.1)
                self._get_fresh_seed_or_none()
                if self._seed_station is not None:
                    break

        if self._seed_station is None:
            # no seed yet â†’ just wait; watcher will trigger us again on s1
            self._status.last_cycle_status = "Idle"
            self._status.last_cycle_message = "Waiting for first snapshot (s1+)."
            self._notify()
            return

        seed_station, seed_meta = self._seed_station, self._seed_meta
        obj.set_seed(seed_station, self._status.active_setpoints)
        obj.prepare_t0()

        # Baseline J at ACTIVE (for Î”J and antiâ€‘chatter)
        t0 = time.time()
        try:
            J_base, _m_base = obj.evaluate(x_active)
            self._status.baseline_J = float(J_base)
        except Exception as e:
            self._status.last_error = f"Baseline evaluate failed: {e}"
            self._status.last_cycle_status = "Failed"
            return
        # Adjust time budget by time spent so far
        time_budget = max(0.05, time_budget - (time.time() - t0))

        # Warmâ€‘start
        x0 = x_active.copy()
        if self._incumbent_x0 is not None:
            alpha = float(getattr(self.nm_cfg, "warm_start_blend", 0.7))
            x0 = alpha * self._incumbent_x0 + (1.0 - alpha) * x0

        # Solve
        self._status.last_cycle_status = "Solving"
        runner = NMRunner()
        try:

            print_every = 10  # print every N evals (tunable)

            def _on_progress(p: Dict[str, object]) -> None:
                # Pull numbers safely
                ne = int(p.get("n_eval", 0) or 0)
                ni = int(p.get("n_iter", 0) or 0)
                elapsed = float(p.get("elapsed_s", 0.0) or 0.0)
                bestJ = p.get("best_J", None)
                bestx = p.get("best_x", None)
                is_best = bool(p.get("is_best", False))

                # Update status (GUI will refresh via notifier thread)
                self._status.last_cycle_iter = ni
                self._status.last_cycle_elapsed_s = elapsed
                if bestJ is not None:
                    try: self._status.best_J = float(bestJ)
                    except Exception: pass
                if bestx is not None:
                    try:
                        bx = np.asarray(bestx, dtype=float).ravel()
                        if bx.size == 4: self._status.best_x = (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]))
                    except Exception:
                        pass
                # optional KPIs if payload forwarded them (when is_best)
                if is_best:
                    metrics = p.get("metrics", {})
                    self._status.best_metrics = metrics
                    self._status.kpi_pur_ex = metrics.get("Pur_ex")
                    self._status.kpi_pur_ra = metrics.get("Pur_ra")
                    self._status.kpi_dil_ex = metrics.get("Dil_ex")
                    self._status.kpi_dil_ra = metrics.get("Dil_ra")

                # Low-noise prints
                if is_best:
                    try:
                        bx_txt = ", ".join(f"{v:.3g}" for v in np.asarray(self._status.best_x or [], float))
                    except Exception:
                        bx_txt = "â€”"
                    print(f"[MPC] new best @eval {ne}  J={self._status.best_J:.4g}  x=[{bx_txt}]")

                elif ne % print_every == 0 and p.get("stage") == "eval":
                    J = p.get("J")
                    try:
                        print(f"[MPC] eval {ne}  J={float(J):.4g}  best={self._status.best_J if self._status.best_J is not None else float('nan'):.4g}")
                    except Exception:
                        pass


                # Push to GUI now (keeps it feeling live)
                self._notify()

            # throttle staging to avoid hammering OPC
            self._last_stage_wall = getattr(self, "_last_stage_wall", 0.0)

            def _progress_cb(info: dict):
                # Forward GUI updates (you likely already do this)
                self._status.n_eval = info.get("n_eval")
                self._status.n_iter = info.get("n_iter")
                self._status.elapsed_s = info.get("elapsed_s")
                self._status.best_J = info.get("best_J")
                best_x = info.get("best_x")

                # Store KPI metrics for GUI when a new best appears
                if info.get("is_best"):
                    metrics = info.get("metrics", {}) or {}
                    self._status.best_metrics = metrics
                    self._status.kpi_pur_ex = metrics.get("Pur_ex")
                    self._status.kpi_pur_ra = metrics.get("Pur_ra")
                    self._status.kpi_dil_ex = metrics.get("Dil_ex")
                    self._status.kpi_dil_ra = metrics.get("Dil_ra")

                # Stage on improvement (every â‰¥0.5 s)
                if info.get("is_best") and best_x is not None:
                    now = time.time()
                    if now - self._last_stage_wall >= 0.5:
                        self._stage_next_setpoints(best_x, reason="improve")
                        self._last_stage_wall = now

                self._notify()

            res = runner.solve(
                x0=x0,
                objective=obj,
                time_budget_s=time_budget,  # â† use dynamic per-cycle budget
                cfg=self.nm_cfg,
                progress_cb=_on_progress,
            )

            # Final guard-stage of best-so-far after solve returns
            best_x = getattr(res, "best_x", None)
            if best_x is None:
                # some runners keep it internally
                best_x = getattr(runner, "_best_x", None)
            if best_x is not None:
                self._stage_next_setpoints(best_x, reason="final-guard")

        except Exception as e:
            self._status.last_error = f"Optimizer error: {e}"
            self._status.last_cycle_status = "Failed"
            return

        # --- Capture result (safe for NumPy arrays) ---
        if res.best_x is not None:
            bx_arr = np.asarray(res.best_x, dtype=float).ravel()
        else:
            bx_arr = x_active  # already np.ndarray

        # (optional) sanity on dimensionality
        if bx_arr.size != 4:
            raise ValueError(f"best_x must have 4 elements (Q1,Q2,Q4,SI); got {bx_arr.size}")

        self._incumbent_x0 = bx_arr.copy()
        self._status.best_x = tuple(map(float, bx_arr))

        self._status.best_J = float(res.best_J) if res.best_J is not None else None
        self._status.best_metrics = dict(res.best_metrics or {})
        self._status.last_cycle_iter = int(getattr(res, "n_iter", 0))
        self._status.last_cycle_elapsed_s = float(getattr(res, "elapsed_s", 0.0))
        self._status.last_cycle_status = getattr(res, "status", "Done")
        self._status.last_cycle_message = str(getattr(res, "message", ""))

        # Decide whether to stage
        mode = self._status.mode
        improvement = None
        try:
            if self._status.best_J is not None and self._status.baseline_J is not None:
                improvement = self._status.baseline_J - self._status.best_J  # positive is good
        except Exception:
            improvement = None

        should_stage = (
            mode == "automatic"
            and self._status.best_x is not None
            and (improvement is None or improvement >= float(self.policy.improvement_epsilon))
        )

        if should_stage:
            Q1, Q2, Q4, SI = self._status.best_x
            values = {"Q1": float(Q1), "Q2": float(Q2), "Q4": float(Q4), "SwitchInterval": float(SI)}
            try:
                self.opc.write(values)
                self._status.next_setpoints = {"Q1": values["Q1"], "Q2": values["Q2"], "Q4": values["Q4"], "SI": values["SwitchInterval"]}
                self._status.last_cycle_status = "Applied"
                self._status.last_cycle_message = "Staged NEXT"
            except Exception as e:
                self._status.last_error = f"OPC write failed: {e}"
                self._status.last_cycle_status = "Failed"
        else:
            self._status.last_cycle_status = "Skipped" if mode == "automatic" else "ReadOnly"
            if mode != "automatic":
                self._status.last_cycle_message = "Plant in MANUAL mode â€” no staging."
            elif improvement is not None and improvement < float(self.policy.improvement_epsilon):
                self._status.last_cycle_message = f"Î”J={improvement:.3g} < Îµ â†’ no stage"
            else:
                self._status.last_cycle_message = "No valid incumbent â†’ no stage"

    # ---------------------- Observer notifier ----------------------

    def _notifier_loop(self):
        while not self._stop.is_set():
            if self._observers:
                snap = OrchestratorStatus(**asdict(self._status))
                for cb in list(self._observers):
                    try: cb(snap)
                    except: pass
            self._stop.wait(0.25)

    # --- helper: push status to observers safely ---
    def _notify(self) -> None:
        snap = copy.deepcopy(self._status)
        for cb in list(self._observers):
            try:
                cb(snap)
            except Exception:
                pass

    def _heartbeat_loop(self) -> None:
        """Probe OPC to estimate latency and connection health every ~1 s."""
        while not self._stop.is_set():
            try:
                t0 = time.time()
                snap = self.opc.read_snapshot()
                t1 = time.time()

                self._status.opc_connected = True
                self._status.opc_latency_ms = (t1 - t0) * 1000.0
                self._status.opc_last_seen_wall = t1
                self._ingest_snapshot(snap)

                try:
                    opc_cd = float(snap.get("SwitchCountdown", 0.0))
                except Exception:
                    opc_cd = 0.0
                self._last_opc_countdown = opc_cd

                # Handle transitions of IsRunning
                is_run = self._status.is_running
                if self._last_is_running is None or is_run != self._last_is_running:
                    self._last_is_running = is_run
                    if not is_run:
                        self._status.last_cycle_status = "Idle"
                        self._status.last_cycle_message = "Plant is STOPPED (IsRunning=false). MPC will not optimize/apply."
                    else:
                        # Plant just started running
                        self._status.last_cycle_status = "Idle"
                        self._status.last_cycle_message = "Plant is RUNNING. Starting immediately."
                        # Ensure we have s0 (fail fast if missing)
                        self._require_seed_or_crash(is_running=True)
                        # Start watching the handover folder for subsequent cycles
                        def _on_new_snapshot(idx, path):
                            # Debounce: only react if MPC is running
                            if not self._status.running:
                                return
                            # Load the seed (updates _seed_station/_seed_meta and last index)
                            self._get_fresh_seed_or_none()
                            # Kick an optimization cycle (donâ€™t stack workers)
                            if not (self._worker and self._worker.is_alive()):
                                self._on_switch_event()

                        # CREATE & START the watcher *now* (plant is already running)
                        if not self._watcher:
                            self._watcher = SnapshotWatcher(
                                folder=self.handover_dir,
                                get_latest_path=self._latest_snapshot_path,
                                parse_idx=self._parse_snapshot_index,
                                cb=_on_new_snapshot,
                                poll_s=0.1,
                            )
                            self._watcher.start(last_seen=self._last_snapshot_idx)
                            print(f"[MPC] Snapshot watcher started â†’ {self.handover_dir}")
                        # Kick first cycle immediately
                        if not (self._worker and self._worker.is_alive()):
                            self._on_switch_event()

                self._notify()

            except Exception as e:
                self._status.opc_connected = False
                self._status.last_error = f"OPC heartbeat failed: {e}"
                self._notify()

            self._stop.wait(1.0)

    # ---------------------- Snapshot handover helpers ----------------------
    def _latest_snapshot_path(self) -> Optional[str]:
        """
        Prefer the engine's atomic pointer JSON; fall back to glob if absent.
        Returns an absolute path to state_s{N}.pkl or None if not found.
        """
        # Pointer: e.g., {"switch_index": N, "file": "state_sN.pkl", "created_utc": "..."}
        ptr = os.path.join(self.handover_dir, "state_slatest.json")
        try:
            with open(ptr, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fn = meta.get("file")
            if isinstance(fn, str) and fn.endswith(".pkl"):
                p = os.path.join(self.handover_dir, fn)
                # In practice this exists because engine writes pointer AFTER the pickle,
                # but keep a defensive check:
                if os.path.exists(p):
                    return p
        except Exception:
            pass  # pointer missing or unreadable â†’ fall back

        # Fallback: your original glob logic
        paths = glob.glob(os.path.join(self.handover_dir, "state_s*.pkl"))
        best = None; best_idx = -1
        for p in paths:
            idx = self._parse_snapshot_index(p)
            if idx is not None and idx > best_idx:
                best_idx = idx; best = p
        return best

    def _parse_snapshot_index(self, path: str) -> Optional[int]:
        try:
            base = os.path.basename(path)
            n = base.replace("state_s", "").replace(".pkl", "")
            return int(n)
        except Exception:
            return None

    def _load_snapshot(self, path: str) -> Tuple[object, dict]:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        st = payload.get("station", None)
        meta = payload.get("meta", {})
        if st is None:
            raise RuntimeError(f"Snapshot missing station: {path}")
        return st, meta

    def _require_seed_or_crash(self, is_running: bool):
        p = self._latest_snapshot_path()
        if not is_running:
            raise RuntimeError("Plant is STOPPED; MPC should not start cycles.")
        if not p:
            raise RuntimeError("No snapshot files present; expected state_s0.pkl. Aborting.")
        idx = self._parse_snapshot_index(p)
        if idx is None or idx < 0:
            raise RuntimeError(f"Invalid snapshot filename: {p}")
        if idx == 0:
            print(f"[MPC] Found seed snapshot s0 â†’ {p}")
        else:
            print(f"[MPC] Warning: first snapshot is s{idx} (not s0). Proceeding with latest.", flush=True)
        st, meta = self._load_snapshot(p)
        # cache
        self._seed_station, self._seed_meta = st, meta
        self._last_snapshot_idx = idx
        self._status.last_snapshot_index = idx
        self._status.last_snapshot_created_utc = meta.get("created_utc")
        return st, meta

    def _get_fresh_seed_or_none(self):
        p = self._latest_snapshot_path()
        if not p:
            return None
        idx = self._parse_snapshot_index(p)
        if idx is None:
            return None
        if self._last_snapshot_idx is None or idx > self._last_snapshot_idx:
            st, meta = self._load_snapshot(p)
            # cache
            self._seed_station, self._seed_meta = st, meta
            self._last_snapshot_idx = idx
            self._status.last_snapshot_index = idx
            self._status.last_snapshot_created_utc = meta.get("created_utc")
            print(f"[MPC] Loaded snapshot s{idx} ({os.path.basename(p)})  created={self._status.last_snapshot_created_utc}")
            return st, meta
        return None

    def _ingest_snapshot(self, snap: Dict[str, Any]) -> None:
        try:
            self._status.mode = str(snap.get("Mode", "manual")).lower()
        except Exception:
            self._status.mode = "manual"
        try:
            self._status.is_running = bool(snap.get("IsRunning", False))
        except Exception:
            self._status.is_running = False
        try:
            self._status.time_to_switch_s = float(snap.get("SwitchCountdown", 0.0))
        except Exception:
            pass
        self._status.active_setpoints = {
            "F":  float(snap.get("ActiveFeed", 0.0)),
            "Q1": float(snap.get("ActiveQ1", 0.0)),
            "Q2": float(snap.get("ActiveQ2", 0.0)),
            "Q4": float(snap.get("ActiveQ4", 0.0)),
            "SI": float(snap.get("ActiveSwitchInterval", 0.0)),
        }
        try:
            self._status.speed_factor = float(snap.get("SpeedFactor", 1.0))
        except Exception:
            self._status.speed_factor = 1.0

    # --- OPC staging helper (single source of truth) ---
    def _stage_next_setpoints(self, x, *, reason: str) -> bool:
        """
        Stage NEXT setpoints to OPC UA and echo-verify.
        x = (Q1, Q2, Q4, SI) in that order. Feed is kept at current ACTIVE.
        Returns True on echo match.
        """
        if x is None:
            return False
        try:
            q1, q2, q4, si = map(float, x)
        except Exception:
            return False

        # Only stage when plant is in automatic (or relax if you want MANUAL staging)
        mode = (self._status.mode or "").lower()
        if "auto" not in mode:
            print(f"[MPC] Skipping stage (mode={self._status.mode})", flush=True)
            return False

        try:
            snap = self.opc.read_snapshot()
            f_active = float(snap.get("ActiveFeed", snap.get("Feed", 21.0)))

            # Write NEXT buffer (keys must match your opcua_client)
            self.opc.write({
                "Feed": f_active,
                "Q1": q1,
                "Q2": q2,
                "Q4": q4,
                "SwitchInterval": si,
            })

            # Echo-read to verify
            rb = self.opc.read_snapshot()
            ok = (
                abs(float(rb.get("Q1", q1)) - q1) < 1e-6 and
                abs(float(rb.get("Q2", q2)) - q2) < 1e-6 and
                abs(float(rb.get("Q4", q4)) - q4) < 1e-6 and
                abs(float(rb.get("SwitchInterval", si)) - si) < 1e-6
            )
            if ok:
                msg = f"Staged NEXT ({reason}): Q1={q1:.1f}  Q2={q2:.1f}  Q4={q4:.1f}  SI={si:.1f}"
                self._status.last_cycle_message = msg
                self._status.last_write_wall = time.time()
                self._status.best_applied = False
                self._notify()
                print("[MPC] stagedâ†’echo OK:", msg, flush=True)
                return True
            else:
                print("[MPC][WARN] stagedâ†’echo mismatch; server may clamp or mapping mismatch", flush=True)
                return False

        except Exception as e:
            print(f"[MPC][ERR] staging failed: {e}", flush=True)
            return False

# ---------------------------------------------------------------------------
# Minimal CLI runner (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import logging
    from ChroMo_SMB.SMB_PlantSim.opcua_client import SMB_OPCUAClient
    from mpc_gui import MPCGuiApp

    opc = SMB_OPCUAClient(endpoint="opc.tcp://127.0.0.1:4840")
    orch = MPCOrchestrator(opc=opc, handover_dir=os.path.join(os.getcwd(), r"C:\Users\z004d8nt\PycharmProjects\DisertationTSV\ChroMo_SMB\SMB_SE\handover"))
    try:
        orch.start()
        from mpc_gui import MPCGuiApp
        MPCGuiApp(orch).run()
    finally:
        orch.stop()
