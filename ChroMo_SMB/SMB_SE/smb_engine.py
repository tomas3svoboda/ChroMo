"""
SMB Digital Twin Engine (headless)
==================================

Purpose
-------
Run a deterministic, low-noise digital twin of a simulated moving bed (SMB)
plant that is exposed over OPC UA. The engine:
  • Reads the plant snapshot (IsRunning, Active* setpoints, SimulationTime,…)
  • Keeps an internal SMBStation in lockstep
  • Applies updated setpoints either at zone-switch boundaries (default)
    or immediately, per operator policy
  • Notifies subscribers after each simulation advance

Design choices
--------------
• **Simulation time source:** If the OPC UA snapshot contains `SimulationTime`,
  that is the ground truth. Otherwise the engine falls back to wall‑clock ×
  SpeedFactor (pinned when the run starts).
• **Setpoint application:** The twin applies plant Active* immediately on each tick.
  On a detected switch, it re-reads a post-switch snapshot, then applies before stepping.
• **Terminology:** Variables are named in process terms (Feed, Eluent, Extract,
  Recycle), and zone flows are derived from a simple flow balance.

Public API
----------
class SMBTwinEngine:
  subscribe(callback)        # callback(state_dict, sim_time)
  start(); stop()

CLI (developer convenience): see `python smb_engine.py -h`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
import os, tempfile, pickle, datetime
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Adjust these imports to your project layout if needed
from ChroMo_SMB.SMB_PlantSim.opcua_client import SMB_OPCUAClient
from ChroMo_SMB.EKF.SMB.SMBStation import SMBStation
from ChroMo_SMB.EKF.SMB.LinColumn import LinColumn
from ChroMo_SMB.EKF.SMB.Tube import Tube


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentSpec:
    name: str
    feed_concentration: float
    henry_constant: float
    delta: float
    Di: float


# ---------------------------------------------------------------------------
# Helpers (process‑centric, kept tiny and readable)
# ---------------------------------------------------------------------------

# OLD:
# def _compute_zone_flows(feed: float, eluent: float, extract: float, recycle: float) -> Dict[str, float]:

# NEW:
def _compute_zone_flows_F_Q(feed: float, q1: float, q2: float, q4: float) -> Dict[str, float]:
    """
    Compute dependent streams and per-zone flows from (F, Q1, Q2, Q4).
      Zone I   = Q1
      Zone II  = Q2
      Zone III = QIII = Q2 + F
      Zone IV  = Q4
      D = Q1 - Q4
      E = Q1 - Q2
      R = QIII - Q4 = Q2 + F - Q4
    Balance: F + D == E + R
    """
    v1 = q1
    v2 = q2
    v3 = q2 + feed
    v4 = q4
    D = q1 - q4
    E = q1 - q2
    R = v3 - q4
    if abs((feed + D) - (E + R)) > 1e-6:
        raise ValueError("Flow balance error")
    return {
        "feed": feed, "q1": q1, "q2": q2, "q4": q4,
        "eluent": D, "extract": E, "raffinate": R,
        "zone1": v1, "zone2": v2, "zone3": v3, "zone4": v4
    }



def _build_station(
    *,
    dt: float,
    Nx: int,
    switch_interval: float,
    col_length: float,
    col_diameter: float,
    porosity: float,
    dead_volume: float,
    components: List[ComponentSpec],
    zone_flows: List[float],
) -> SMBStation:
    """Create a fresh SMBStation with 4 zones (Column + Tube),
    attach components in order, set numerics & flows, and initialize numerics.
    """
    smb = SMBStation()
    for zone in range(1, 5):
        smb.addColZone(zone, LinColumn(col_length, col_diameter, porosity), Tube(dead_volume))
    for zone_idx, fr in enumerate(zone_flows, start=1):
        smb.setFlowRateZone(zone_idx, fr)

    smb.setSwitchInterval(switch_interval)
    smb.setdt(dt)
    smb.setNx(Nx)

    for c in components:
        smb.createComponentAB(c.name, c.feed_concentration, c.henry_constant, c.delta, c.Di)

    smb.initCols()
    return smb


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SMBTwinEngine:
    """Headless SMB digital twin that mirrors an OPC UA plant.

    • Reads plant snapshots via `SMB_OPCUAClient.read_snapshot()`
    • Keeps an internal `SMBStation` synchronized with the plant
    • Applies updated setpoints either at boundaries (default) or immediately

    Subscribe with a function `fn(state, sim_time)`; it will be called after
    each advance (`SMBStation.step(n)`), where `state` is the station's step
    return object and `sim_time` is the station time in seconds.
    """

    # ---------------- Construction ----------------
    def __init__(
        self,
        *,
        opc_endpoint: str = "opc.tcp://127.0.0.1:4840",
        obj_name: str = "SMB_Simulation",
        dt: float = 1.0,
        Nx: int = 100,
        switch_interval_default: float = 780.0,
        col_length: float = 310.0,
        col_diameter: float = 10.0,
        porosity: float = 0.376,
        dead_volume: float = 0.2,
        components: Optional[List[Dict[str, float]]] = None,
        max_steps_per_update: int = 1,
        poll_period_s: float = 0.1,
        opc_client: Optional[SMB_OPCUAClient] = None,
        manage_client: bool = True,
        # Tolerances for staging changes
        flow_tol: float = 1e-6,
        si_tol: float = 1e-6,
        speed_tol: float = 1e-6,
        force_apply_timeout_s: float = 10.0,
        min_switch_gap_s: float = 10.0,
    ) -> None:

        self._wakeup = threading.Event()
        self._subscribed_to_client = False
        self.min_switch_gap_s = float(min_switch_gap_s)
        self._edge_latch_sc: Optional[float] = None
        self._last_switch_index: Optional[int] = None

        # OPC client ownership
        self._manage_client = bool(manage_client)
        self.cli = opc_client if opc_client is not None else SMB_OPCUAClient(
            endpoint=opc_endpoint, obj_browse_name=obj_name
        )

        # Numerics / geometry
        self.dt = float(dt)
        self.Nx = int(Nx)
        self.switch_interval_default = float(switch_interval_default)
        self.col_length = float(col_length)
        self.col_diameter = float(col_diameter)
        self.porosity = float(porosity)
        self.dead_volume = float(dead_volume)

        # Component list (order matters)
        comp_raw = components or [
            {"name": "Man", "feed_concentration": 7.27, "henry_constant": 0.616, "delta": 31.43, "Di": 0.0007},
            {"name": "Gal", "feed_concentration": 3.42, "henry_constant": 0.265, "delta": 31.43, "Di": 0.0007},
        ]
        self.components: List[ComponentSpec] = [
            ComponentSpec(**c) for c in comp_raw
        ]

        # Cadence
        self.max_steps_per_update = int(max_steps_per_update)
        self.poll_period_s = float(poll_period_s)

        # Update policy
        self.flow_tol = float(flow_tol)
        self.si_tol = float(si_tol)
        self.speed_tol = float(speed_tol)
        self.force_apply_timeout_s = float(force_apply_timeout_s)

        # Observers & threading
        self._listeners: List[Callable[[Dict[int, List[np.ndarray]], float], None]] = []
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._stop = threading.Event()

        # Runtime state
        self._smb: Optional[SMBStation] = None
        self._sim_start_wall: Optional[float] = None  # wall time pinned to sim t=0 at given speed
        self._running_prev: Optional[bool] = None

        # Boundary re-entry protection
        self._boundary_in_progress: bool = False
        self._last_boundary_sim_t: Optional[float] = None
        self._ignore_boundaries_until: Optional[float] = None

        # Concurrency
        self._lock = threading.RLock()

        # Snapshot handover config
        self.snapshot_dir: Optional[str] = os.path.join(os.getcwd(), "handover")
        self.snapshot_prefix: str = "state_s"
        # Create folder if enabled
        if self.snapshot_dir:
            os.makedirs(self.snapshot_dir, exist_ok=True)

        # Monotone switch counter for handover
        self._switch_index: int = -1  # will emit s0 on rising edge

        # Last applied and pending updates
        self._applied: Dict[str, float] = {}   # keys: F,D,E,Q4,SI,SPEED

    # ---------------- Public API ----------------
    def subscribe(self, fn: Callable[[Dict[int, List[np.ndarray]], float], None]) -> None:
        """Register a callback that receives (step_result, sim_time) after each advance."""
        self._listeners.append(fn)
        name = getattr(fn, "__qualname__", repr(fn))
        print(f"[SMB Engine] Listener added: {name} (total={len(self._listeners)})", flush=True)

    def start(self) -> None:
        """Begin the background loop (non‑blocking)."""
        self._stop.clear()
        if not self._thr.is_alive():
            self._thr = threading.Thread(target=self._loop, daemon=True)
            self._thr.start()

    def stop(self) -> None:
        """Stop the background loop and wait briefly for it to exit."""
        self._stop.set()
        if self._thr.is_alive():
            self._thr.join(timeout=2.0)

    def get_station_clone(self) -> SMBStation:
        """Thread-safe deep copy of the current SMBStation (for MPC seeding)."""
        with self._lock:
            if self._smb is None:
                raise RuntimeError("Engine not initialized yet")
            return deepcopy(self._smb)

    # ---------- Snapshot handover (hardened) ----------

    def _snapshot_path_for(self, idx: int) -> str:
        assert self.snapshot_dir, "snapshot_dir is disabled"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        fname = f"{self.snapshot_prefix}{idx}.pkl"  # deep, atomic, versioned
        return os.path.join(self.snapshot_dir, fname)

    def _atomic_write_bytes(self, path: str, data: bytes) -> None:
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        # write to temp file, flush, then atomic rename (same filesystem)
        with tempfile.NamedTemporaryFile("wb", dir=d, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, path)
        # extra safety: fsync the directory on POSIX so the rename itself is durable
        if os.name == "posix":
            try:
                dirfd = os.open(d, os.O_DIRECTORY | os.O_RDONLY)
                try:
                    os.fsync(dirfd)
                finally:
                    os.close(dirfd)
            except Exception:
                # best-effort; safe to ignore on platforms/filesystems that don't support it
                pass

    def _emit_snapshot(self, *, reason: str) -> None:
        """Create deep-copy snapshot and write atomically to handover folder,
        then publish an atomic pointer JSON (and optional alias) for MPC pickup."""
        if not self.snapshot_dir:
            return

        try:
            with self._lock:
                # Consistent view under lock
                station_copy = deepcopy(self._smb)

                sim_time   = float(getattr(self._smb, "timer", 0.0))
                applied_si = float(self._applied.get("SI", self.switch_interval_default))
                speed      = float(self._applied.get("SPEED", 1.0))
                applied    = {
                    "F":  float(self._applied.get("F",  0.0)),
                    "Q1": float(self._applied.get("Q1", 0.0)),
                    "Q2": float(self._applied.get("Q2", 0.0)),
                    "Q4": float(self._applied.get("Q4", 0.0)),
                    "SI": applied_si,
                }

                plant = {}
                snap = getattr(self, "_latest_snap", None) or {}
                for k, tag in (("F","ActiveFeed"),("Q1","ActiveQ1"),
                               ("Q2","ActiveQ2"),("Q4","ActiveQ4"),
                               ("SI","ActiveSwitchInterval")):
                    try:
                        plant[k] = float(snap[tag])
                    except Exception:
                        plant[k] = None

                switch_index = int(self._switch_index)

            meta = {
                "reason": reason,
                "switch_index": switch_index,
                "sim_time_s": sim_time,
                "dt_s": float(self.dt),
                "si_s": applied_si,
                "speed_factor": speed,
                "engine_applied": applied,
                "plant_active": plant,
                "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "schema": "smb_engine:handover:2",
            }

            payload = {"station": station_copy, "meta": meta}

            # Serialize outside the lock
            data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

            # 1) Write numbered snapshot atomically
            path = self._snapshot_path_for(switch_index)
            self._atomic_write_bytes(path, data)
            print(f"[SMB Engine] Snapshot s{switch_index} saved → {path}  ({reason})", flush=True)

            # 2) Publish atomic pointer JSON (MPC reads this file)
            pointer = {
                "switch_index": switch_index,
                "file": os.path.basename(path),
                "created_utc": meta["created_utc"],
            }
            ptr_bytes = json.dumps(pointer, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ptr_path  = os.path.join(self.snapshot_dir, f"{self.snapshot_prefix}latest.json")
            self._atomic_write_bytes(ptr_path, ptr_bytes)

            # 3) Optional: also maintain a fixed alias .pkl (off by default)
            if getattr(self, "write_snapshot_alias", False):
                alias_path = os.path.join(self.snapshot_dir, f"{self.snapshot_prefix}current.pkl")
                self._atomic_write_bytes(alias_path, data)

        except Exception as ex:
            # Never crash the engine because of snapshotting
            print(f"[SMB Engine] Snapshot error: {ex!r}", file=sys.stderr)


    # ---------------- Internals ----------------
    def _read_snapshot_safe(self) -> Dict[str, float]:
        try:
            return self.cli.read_snapshot()
        except Exception as ex:
            print(f"[SMB Engine] Warning: read_snapshot failed: {ex}", file=sys.stderr)
            return {}

    @staticmethod
    def _pick(snap: Dict[str, float], *names: str, default: float = 0.0) -> float:
        for n in names:
            v = snap.get(n)
            if v is not None:
                return float(v)
        return float(default)

    def _read_operating_points(self, snap: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
        """
        Return (F, Q1, Q2, Q4, SI, S) from snapshot.
        Authoritative: plant Active* (what is running now).
        Fallback to writable only if Active* is absent at startup.
        """
        si_def = self.switch_interval_default

        def pick_active_or_fallback(act, fb, dflt):
            v = snap.get(act, None)
            if v is None:
                v = snap.get(fb, dflt)  # init-only fallback
            return max(0.0, float(v))

        F  = pick_active_or_fallback("ActiveFeed",            "Feed",            21.0)
        Q1 = pick_active_or_fallback("ActiveQ1",              "Q1",              180.0)
        Q2 = pick_active_or_fallback("ActiveQ2",              "Q2",              93.0)
        Q4 = pick_active_or_fallback("ActiveQ4",              "Q4",              45.0)
        SI = pick_active_or_fallback("ActiveSwitchInterval",  "SwitchInterval",  si_def)
        S  = max(0.0, float(snap.get("SpeedFactor", 1.0)))

        return F, Q1, Q2, Q4, SI, S

    def _sync_to_plant(self, snap: Dict[str, float]) -> bool:
        """
        Ensure SMBStation exactly matches plant Active* (apply immediately in this tick).
        Returns True if an apply was performed.
        """
        try:
            F  = float(snap["ActiveFeed"])
            Q1 = float(snap["ActiveQ1"])
            Q2 = float(snap["ActiveQ2"])
            Q4 = float(snap["ActiveQ4"])
            SI = float(snap["ActiveSwitchInterval"])
        except Exception:
            return False  # Active* incomplete this tick

        chg = {}
        def diff(k, v, tol):
            a = self._applied.get(k)
            if a is None or abs(a - v) > tol:
                chg[k] = v

        diff("F",  F,  self.flow_tol)
        diff("Q1", Q1, self.flow_tol)
        diff("Q2", Q2, self.flow_tol)
        diff("Q4", Q4, self.flow_tol)
        diff("SI", SI, self.si_tol)

        if not chg:
            return False

        # Apply immediately (before any step for this tick)
        self._apply_setpoints(F, Q1, Q2, Q4, SI)
        self._applied.update({"F": F, "Q1": Q1, "Q2": Q2, "Q4": Q4, "SI": SI})

        # One-shot verify on next tick (belt & suspenders)
        self._verify_next = True
        return True


    def _detect_changes(self, F: float, Q1: float, Q2: float, Q4: float, SI: float, S: float) -> Dict[str, float]:
        """Return dict of changed setpoints keyed by 'F','Q1','Q2','Q4','SI','SPEED'."""
        chg: Dict[str, float] = {}
        def diff(key: str, val: float, tol: float) -> None:
            old = self._applied.get(key)
            if old is None or abs(val - old) > tol:
                chg[key] = val
        diff("F", F, self.flow_tol)
        diff("Q1", Q1, self.flow_tol)
        diff("Q2", Q2, self.flow_tol)
        diff("Q4", Q4, self.flow_tol)
        diff("SI", SI, self.si_tol)
        diff("SPEED", S, self.speed_tol)
        return chg

    def _apply_setpoints(self, F: float, Q1: float, Q2: float, Q4: float, SI: float) -> None:
        """
        Atomically push (F, Q1, Q2, Q4, SI) into SMBStation.
        - Validates & balances flows via _compute_zone_flows_F_Q (raises on mismatch).
        - Sets all four zone flows and the switch interval under one lock.
        - Verifies by reading SMBStation's internal values back (flowRates/interval).
        NOTE: LinColumn numerics are rebuilt on the next step(); KLO runs after that step.
        """
        # --- validate inputs ---
        if not self._smb:
            print("[SMB Engine] Warning: apply_setpoints called before station exists", flush=True)
            return
        if not all(np.isfinite(x) for x in (F, Q1, Q2, Q4, SI)):
            raise ValueError("Non-finite setpoint detected")

        # Clamp tiny negatives; protect against nonpositive SI
        eps = 0.0
        F  = max(F,  eps); Q1 = max(Q1, eps); Q2 = max(Q2, eps); Q4 = max(Q4, eps)
        SI = max(SI, max(self.dt, 1e-9))

        # --- compute balanced zone flows (raises on imbalance) ---
        flows = _compute_zone_flows_F_Q(F, Q1, Q2, Q4)

        # --- atomic apply under the engine lock ---
        with self._lock:
            smb = self._smb
            smb.setFlowRateZone(1, flows["zone1"])
            smb.setFlowRateZone(2, flows["zone2"])
            smb.setFlowRateZone(3, flows["zone3"])
            smb.setFlowRateZone(4, flows["zone4"])
            # propagate Q1..Q4 into tubes/columns immediately
            smb.apply_zone_flows_now()
            smb.setSwitchInterval(SI)

            # read-back verify (what the station actually holds)
            fr1 = float(smb.flowRates[1]); fr2 = float(smb.flowRates[2])
            fr3 = float(smb.flowRates[3]); fr4 = float(smb.flowRates[4])
            si  = float(getattr(smb, "interval", SI))

        # tolerances
        f_tol  = getattr(self, "flow_tol", 1e-6)
        si_tol = getattr(self, "si_tol",   1e-6)

        # compare and warn if anything drifted (due to rounding etc.)
        mism = {}
        if abs(fr1 - flows["zone1"]) > f_tol: mism["zone1"] = (flows["zone1"], fr1)
        if abs(fr2 - flows["zone2"]) > f_tol: mism["zone2"] = (flows["zone2"], fr2)
        if abs(fr3 - flows["zone3"]) > f_tol: mism["zone3"] = (flows["zone3"], fr3)
        if abs(fr4 - flows["zone4"]) > f_tol: mism["zone4"] = (flows["zone4"], fr4)
        if abs(si  - SI)              > si_tol: mism["SI"]    = (SI, si)

        if mism:
            print(f"[SMB Engine] Notice: station quantized/clamped setpoints: {mism}", flush=True)

        print(
            f"[SMB Engine] Applied setpoints  F={F:.6g}, Q1={Q1:.6g}, Q2={Q2:.6g}, Q4={Q4:.6g}, SI={SI:.6g}",
            flush=True
        )

    # ---------------- Main loop ----------------

    def _get_target_sim_time(self, snap: Dict[str, float]) -> float:
        """Prefer plant‑reported SimulationTime; otherwise fall back to wall‑time × SpeedFactor."""
        simtime = snap.get("SimulationTime")
        if simtime is not None:
            try:
                return float(simtime)
            except Exception:
                pass
        speed = max(0.0, self._applied.get("SPEED", 1.0))
        wall_elapsed = time.time() - (self._sim_start_wall or time.time())
        return wall_elapsed * speed

    def _is_plant_boundary(self, snap: Dict[str, float]) -> bool:
        """True once when SwitchCountdown jumps up (plant performed a switch)."""
        try:
            sc = float(snap.get("SwitchCountdown", 0.0))
        except Exception:
            return False
        prev = getattr(self, "_last_countdown", None)
        self._last_countdown = sc
        if prev is None:
            return False
        # countdown decreases during SI; at the instant of a switch it jumps up ~SI
        return sc > (prev + 0.5 * max(self.dt, 1e-9))

    def _plant_switch_index(self, snap):
        v = snap.get("SwitchIndex", None)
        if v is None: return None
        try:
            return int(float(v))
        except Exception:
            return None

    def _active_delta_vs_applied(self, snap: Dict[str, float]) -> Dict[str, float]:
        out = {}
        try:
            pairs = [("F", float(snap["ActiveFeed"]), self.flow_tol),
                     ("Q1", float(snap["ActiveQ1"]), self.flow_tol),
                     ("Q2", float(snap["ActiveQ2"]), self.flow_tol),
                     ("Q4", float(snap["ActiveQ4"]), self.flow_tol),
                     ("SI", float(snap["ActiveSwitchInterval"]), self.si_tol)]
        except Exception:
            return out
        for k, v, tol in pairs:
            a = self._applied.get(k)
            if a is None or abs(a - v) > tol:
                out[k] = v
        return out

    def _loop(self) -> None:
        import time, math

        self._verify_next = False
        self._last_countdown = None
        self._boundary_in_progress = False
        self._running_prev = self._running_prev
        self._pending_emit_reason = None  # emit snapshot after first step post-boundary

        while not self._stop.is_set():
            tick_wall = time.time()

            # 1) Read latest plant snapshot
            snap = self._read_snapshot_safe() or {}
            self._latest_snap = snap
            is_running = bool(snap.get("IsRunning", False))
            sf = float(snap.get("SpeedFactor", 1.0))
            sim_t_now = float(snap.get("SimulationTime", 0.0))

            # 1a) Optional wakeup subscription
            if not self._subscribed_to_client:
                try:
                    self.cli.subscribe_active_changes(lambda: self._wakeup.set())
                    self._subscribed_to_client = True
                except Exception:
                    self._subscribed_to_client = False

            # 2) First build / start → build station & phase-align
            if self._smb is None or (self._running_prev is False and is_running is True):
                F, Q1, Q2, Q4, SI, _ = self._read_operating_points(snap)
                flows = _compute_zone_flows_F_Q(F, Q1, Q2, Q4)
                zone_flows = [flows["zone1"], flows["zone2"], flows["zone3"], flows["zone4"]]
                with self._lock:
                    self._smb = _build_station(
                        dt=self.dt, Nx=self.Nx, switch_interval=SI,
                        col_length=self.col_length, col_diameter=self.col_diameter,
                        porosity=self.porosity, dead_volume=self.dead_volume,
                        components=self.components, zone_flows=zone_flows
                    )
                    try:
                        sc0 = float(snap.get("SwitchCountdown", SI))
                    except Exception:
                        sc0 = SI
                    self._smb.countdown = max(sc0, max(self.dt, 1e-9))

                self._applied.update({"F": F, "Q1": Q1, "Q2": Q2, "Q4": Q4, "SI": SI, "SPEED": sf})
                sim_now = float(snap.get("SimulationTime", 0.0))
                self._sim_start_wall = tick_wall - sim_now / max(sf, 1e-9)
                self._running_prev = is_running

                # prime last switch index (no edge on first frame)
                self._last_switch_index = None
                try:
                    v = snap.get("SwitchIndex", None)
                    if v is not None:
                        self._last_switch_index = int(float(v))
                except Exception:
                    self._last_switch_index = None

            # 3) SpeedFactor re-pin
            if self._applied.get("SPEED") != sf:
                sim_now = float(getattr(self._smb, "timer", 0.0)) if self._smb else 0.0
                self._sim_start_wall = tick_wall - sim_now / max(sf, 1e-9)
                self._applied["SPEED"] = sf
                print(f"[SMB Engine] SpeedFactor -> {sf:.6g} (re-pinned)", flush=True)

            # 3a) Phase-lock countdown to plant each tick (prevents early twin switch)
            if self._smb is not None and is_running:
                try:
                    si_act = float(snap.get("ActiveSwitchInterval", self._applied.get("SI", self.switch_interval_default)))
                    sc = float(snap.get("SwitchCountdown"))
                    # accept only sane values: within (dt, 2*SI]
                    if np.isfinite(sc) and sc >= self.dt and sc <= 2.0 * max(si_act, self.dt):
                        with self._lock:
                            # hard phase lock to plant’s current position in interval
                            self._smb.countdown = sc
                except Exception:
                    pass

            # 3b) Mid-interval Active* deltas → apply immediately (esp. SI changes)
            if self._smb is not None and is_running:
                try:
                    if self._sync_to_plant(snap):   # will apply only if deltas vs _applied exceed tolerances
                        # ensure a snapshot is emitted after the first post-apply step
                        if not self._pending_emit_reason:
                            self._pending_emit_reason = "mid-apply"
                        # also schedule a one-shot verify on the next tick (belt & suspenders)
                        self._verify_next = True
                except Exception:
                    pass

            # 4) Detect boundary (prefer SwitchIndex edge)
            boundary_now = False
            if is_running:
                idx = None
                try:
                    v = snap.get("SwitchIndex", None)
                    if v is not None:
                        idx = int(float(v))
                except Exception:
                    idx = None
                if idx is not None:
                    if self._last_switch_index is None or idx < self._last_switch_index:
                        self._last_switch_index = idx
                    elif idx > self._last_switch_index:
                        if self._ignore_boundaries_until is None or sim_t_now >= self._ignore_boundaries_until:
                            boundary_now = True
                            self._last_switch_index = idx
                else:
                    candidate = self._is_plant_boundary(snap)
                    if candidate and self._ignore_boundaries_until is not None and sim_t_now < self._ignore_boundaries_until:
                        candidate = False
                    boundary_now = candidate

            # 4b) Handle boundary (with re-entry latch)
            if boundary_now and not self._boundary_in_progress and self._smb is not None:
                self._boundary_in_progress = True
                try:
                    post = self._read_snapshot_safe() or snap
                    self._latest_snap = post
                    sim_t_post = float(post.get("SimulationTime", sim_t_now))

                    # --- Optionally wait a few ms for fresh Active* (coherence retry) ---
                    delta = self._active_delta_vs_applied(post)
                    if not delta:
                        for _ in range(3):
                            time.sleep(self.poll_period_s * 0.25)
                            post2 = self._read_snapshot_safe() or {}
                            d2 = self._active_delta_vs_applied(post2)
                            if d2:
                                post = post2
                                delta = d2
                                self._latest_snap = post
                                break

                    # --- (1) SWITCH FIRST (to match PlantSim semantics) ---
                    with self._lock:
                        self._smb.force_switch_now()

                    # --- (2) SANITIZE plant post-switch countdown (read/repair) ---
                    si_act = float(post.get("ActiveSwitchInterval", self._applied.get("SI", self.switch_interval_default)))
                    try:
                        sc_post = float(post.get("SwitchCountdown"))
                    except Exception:
                        sc_post = float('nan')

                    if (not math.isfinite(sc_post)) or (sc_post <= 0.5 * self.dt) or (sc_post > 2.0 * si_act):
                        for _ in range(2):
                            time.sleep(self.poll_period_s * 0.25)
                            post2 = self._read_snapshot_safe() or {}
                            try:
                                sc2 = float(post2.get("SwitchCountdown", 0.0))
                            except Exception:
                                sc2 = 0.0
                            if (sc2 > self.dt) and (sc2 <= 2.0 * si_act):
                                sc_post = sc2
                                post = post2
                                self._latest_snap = post
                                break
                        else:
                            sc_post = si_act
                    if (not math.isfinite(sc_post)) or (sc_post <= self.dt):
                        sc_post = max(si_act, 2.0 * self.dt)

                    # --- (3) APPLY Active* AFTER switching (flows+SI) ---
                    if delta:
                        self._sync_to_plant(post)          # this will call _apply_setpoints(...) which includes apply_zone_flows_now()

                    # --- (4) OVERWRITE countdown with plant value (so we don't keep SI) ---
                    with self._lock:
                        self._smb.countdown = float(sc_post)

                    self._last_countdown = float(sc_post)

                    # --- (5) bookkeeping and deferred snapshot emission ---
                    plant_idx = post.get("SwitchIndex", None)
                    if plant_idx is not None:
                        try:
                            self._switch_index = int(float(plant_idx))
                        except Exception:
                            self._switch_index += 1
                    else:
                        self._switch_index += 1

                    self._last_boundary_sim_t = sim_t_post
                    self._ignore_boundaries_until = sim_t_post + float(getattr(self, "min_switch_gap_s", 0.0) or 0.0)

                    # Defer snapshot until after first step (matrices rebuilt & state advanced)
                    self._pending_emit_reason = "boundary"

                    # Do not step in this same tick
                    self._running_prev = is_running
                    self._wakeup.wait(timeout=self.poll_period_s)
                    self._wakeup.clear()
                    continue
                finally:
                    self._boundary_in_progress = False

            # 5) One-shot verify (only if we actually applied something previously)
            if self._verify_next:
                try:
                    mism = {}
                    for k, tag in (("F","ActiveFeed"),("Q1","ActiveQ1"),("Q2","ActiveQ2"),
                                   ("Q4","ActiveQ4"),("SI","ActiveSwitchInterval")):
                        if tag not in self._latest_snap:
                            continue
                        act = float(self._latest_snap[tag])
                        tol = self.flow_tol if k in ("F","Q1","Q2","Q4") else self.si_tol
                        if k not in self._applied or abs(self._applied[k] - act) > tol:
                            mism[k] = act
                    if mism:
                        print(f"[SMB Engine] Post-apply mismatch → forcing exact Active*: {mism}", flush=True)
                        self._apply_setpoints(
                            mism.get("F",  self._applied.get("F",  21.0)),
                            mism.get("Q1", self._applied.get("Q1", 180.0)),
                            mism.get("Q2", self._applied.get("Q2", 93.0)),
                            mism.get("Q4", self._applied.get("Q4",  45.0)),
                            mism.get("SI", self._applied.get("SI", self.switch_interval_default)),
                        )
                        for k, v in mism.items():
                            self._applied[k] = v
                except Exception:
                    pass
                self._verify_next = False

            # 6) Advance to plant time using INTEGER steps (phase-safe)
            if self._smb is not None and is_running:
                target_t = float(self._latest_snap["SimulationTime"]) if "SimulationTime" in self._latest_snap else \
                           max(0.0, (tick_wall - (self._sim_start_wall or tick_wall)) * self._applied.get("SPEED", 1.0))

                timer = float(self._smb.timer)
                ahead = target_t - timer
                if ahead > 0.0:
                    steps_needed = int(math.floor(ahead / self.dt + 1e-12))
                    if steps_needed > 0:
                        phase_left = float(getattr(self._smb, "countdown", self.dt))
                        max_no_cross = max(0, int(math.floor((phase_left - 1e-12) / self.dt)))
                        n = min(steps_needed, int(self.max_steps_per_update), max_no_cross)
                        if n > 0:
                            step_res = self._smb.step(n)
                            sim_t = float(self._smb.timer)
                            for cb in list(self._listeners):
                                try:
                                    cb(step_res, sim_t)
                                except Exception as ex:
                                    import sys, traceback
                                    name = getattr(cb, "__qualname__", repr(cb))
                                    print(f"[SMB Engine] Listener error in {name}: {ex}", file=sys.stderr, flush=True)
                                    traceback.print_exc()
                            # Emit deferred snapshot now that numerics/state are updated
                            if self._pending_emit_reason:
                                self._emit_snapshot(reason=self._pending_emit_reason)
                                self._pending_emit_reason = None

            # 7) Remember flags & sleep/poll
            self._running_prev = is_running
            self._wakeup.wait(timeout=self.poll_period_s)
            self._wakeup.clear()

# ---------------------------------------------------------------------------
# CLI wrapper (developer convenience)
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SMB Digital Twin Engine (OPC UA client, dynamic updates)")
    p.add_argument("--endpoint", default="opc.tcp://127.0.0.1:4840")
    p.add_argument("--obj", default="SMB_Simulation")
    p.add_argument("--dt", type=float, default=1)
    p.add_argument("--Nx", type=int, default=100)
    p.add_argument("--si", type=float, default=780.0)
    p.add_argument("--L", type=float, default=310.0)
    p.add_argument("--D", type=float, default=10.0)
    p.add_argument("--eps", type=float, default=0.376)
    p.add_argument("--dv", type=float, default=0.2)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--poll", type=float, default=0.25)
    p.add_argument("--force-apply-timeout-s", type=float, default=10.0,
                   help="If no boundary is detected within this time, apply anyway")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    eng = SMBTwinEngine(
        opc_endpoint=args.endpoint,
        obj_name=args.obj,
        dt=args.dt,
        Nx=args.Nx,
        switch_interval_default=args.si,
        col_length=args.L,
        col_diameter=args.D,
        porosity=args.eps,
        dead_volume=args.dv,
        max_steps_per_update=args.max_steps,
        poll_period_s=args.poll,
        force_apply_timeout_s=args.force_apply_timeout_s,
    )
    eng.start()
    print("[SMB Engine] running headless. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()
        print("[SMB Engine] stopped.")


if __name__ == "__main__":
    main()
