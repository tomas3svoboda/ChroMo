"""
Digital Twin (lean version)
===========================

Purpose
-------
Bring the **SMBStation** up in a *known, fully initialized* state the moment
EKF starts — independent of the OPC UA plant state. The twin is created with:

- 4 zones, each with **LinColumn + Tube**
- N components (your list, usually 2: **Man**, **Gal**)
- Discretization **dt, Nx**, switch interval, and **zone flow totals**

Once running, you can optionally *apply* OPC snapshots to update flows and
interval (read‑only sync).

This file is intentionally compact and readable for non‑programmers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging
import math
import numpy as np

try:
    # Preferred (package) imports
    from SMB.SMBStation import SMBStation
    from SMB.LinColumn import LinColumn
    from SMB.Tube import Tube
except Exception:  # pragma: no cover
    # Flat layout fallback
    from SMBStation import SMBStation  # type: ignore
    from LinColumn import LinColumn    # type: ignore
    from Tube import Tube              # type: ignore

LOGGER_NAME = "EKF"
log = logging.getLogger(LOGGER_NAME)


# ------------------------------------------------------------------
# Defaults (you can change these in one place)
# ------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, float | int] = {
    "dt": 0.05,
    "Nx": 100,
    "switch_interval": 1894,
    "col_length": 320,
    "col_diameter": 10,
    "porosity": 0.376,
    "dead_volume": 0.5,
}

DEFAULT_COMPONENTS: List[Dict[str, float | str]] = [
    {"name": "Man", "feed_concentration": 9.0, "henry_constant": 4.55, "delta": 54.0, "Di": 0.0007},
    {"name": "Gal", "feed_concentration": 6.0, "henry_constant": 2.77, "delta": 84.0, "Di": 0.0007},
]

# Stream flows (any consistent units). These map to zone totals below.
DEFAULT_FLOWS: Dict[str, float] = {
    "feed": 20.0,
    "eluent": 170.0,
    "extract": 32.0,   # not used in zone totals, kept for completeness
    "recycle": 2.0,
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def derive_zone_flows(streams: Dict[str, float]) -> List[float]:
    """Map stream flows → per‑zone totals [Z1, Z2, Z3, Z4].

    Assumed SMB topology:
      Z1 = recycle + eluent
      Z2 = recycle
      Z3 = recycle + feed
      Z4 = recycle
    """
    rec = float(streams.get("recycle", 0.0))
    fee = float(streams.get("feed", 0.0))
    elu = float(streams.get("eluent", 0.0))
    return [rec + elu, rec, rec + fee, rec]


# ------------------------------------------------------------------
# One simple builder used everywhere
# ------------------------------------------------------------------

def init_smb(params: Dict[str, float | int], components: List[Dict[str, float | str]], zone_flows: Iterable[float]) -> SMBStation:
    """Create a fully initialized SMBStation.

    - 4 zones, each with (LinColumn + Tube)
    - zone_flows: iterable of 4 totals for zones 1..4
    - components: list of dicts with keys: name, feed_concentration, henry_constant, delta, Di
    """
    smb = SMBStation()

    # Geometry per zone (1..4)
    for zone in range(4):
        smb.addColZone(
            zone + 1,
            LinColumn(params["col_length"], params["col_diameter"], params["porosity"]),
            Tube(params["dead_volume"])  # transport delay / dead volume
        )

    # Zone totals
    for i, fr in enumerate(zone_flows, 1):
        smb.setFlowRateZone(i, float(fr))

    # Timing & discretization
    smb.setSwitchInterval(float(params["switch_interval"]))
    smb.setdt(float(params["dt"]))
    smb.setNx(int(params["Nx"]))

    # Components
    for comp in components:
        smb.createComponentAB(
            comp["name"],
            comp["feed_concentration"],
            comp["henry_constant"],
            comp["delta"],
            comp["Di"],
        )

    # Factorize / allocate numerics
    smb.initCols()
    return smb


# ------------------------------------------------------------------
# A tiny twin wrapper for EKF use
# ------------------------------------------------------------------
@dataclass
class TwinConfig:
    """Twin runtime knobs the EKF might read; keep this tiny."""
    dt: float = float(DEFAULT_PARAMS["dt"])  # for convenience
    Nx: int = int(DEFAULT_PARAMS["Nx"])      # — not strictly needed here


@dataclass
class AdvanceInfo:
    steps: int
    t_start: float
    t_end: float
    reason: str
    def __str__(self) -> str:
        return f"steps={self.steps}, Δt={self.t_end - self.t_start:.3f}s, reason={self.reason}"

@dataclass(frozen=True)
class _ObjSlice:
    zone: int                 # 1..4
    kind: str                 # "tube" or "col"
    obj_id: int               # id(obj) for stable identity across switches
    comp_slices: Tuple[slice, slice]  # [Man, Gal] slices in the big vector

@dataclass
class FullStateLayout:
    """Mapping of the full SMB state vector."""
    dim: int
    objects: List[_ObjSlice]   # order of objects in the flattened vector


class DigitalTwin:
    """Minimal, readable wrapper around SMBStation for the EKF.

    On construction we build a fresh, initialized SMB using defaults unless you
    pass custom params/components/flows.
    """
    def __init__(
        self,
        params: Optional[Dict[str, float | int]] = None,
        components: Optional[List[Dict[str, float | str]]] = None,
        flows: Optional[Dict[str, float]] = None,
    ) -> None:
        self.cfg = TwinConfig()
        self.params = dict(DEFAULT_PARAMS if params is None else params)
        self.components = list(DEFAULT_COMPONENTS if components is None else components)
        self.stream_flows = dict(DEFAULT_FLOWS if flows is None else flows)

        self.smb: SMBStation = init_smb(self.params, self.components, derive_zone_flows(self.stream_flows))
        log.info(
            "[twin] Ready: dt=%.3fs, Nx=%d, interval=%.3fs, flows=%s, comps=%d",
            float(getattr(self.smb, "dt", self.params["dt"])),
            int(self.smb.settings.get("Nx", self.params["Nx"])),
            float(getattr(self.smb, "interval", self.params["switch_interval"])),
            dict(self.smb.flowRates),
            len(self.smb.components),
        )
        self._init_tubes()
        self._dt = float(self.params["dt"])          # hard-lock dt for the twin
        self.smb.setdt(self._dt)                     # enforce dt on the SMB core too
        self._ticks = int(round(float(getattr(self.smb, "timer", 0.0)) / self._dt))

    # ---------------- Basic metadata ----------------
    def get_time(self) -> float:
        return float(getattr(self.smb, "timer", 0.0))

    # ---------------- Read‑only OPC sync ----------------
    def apply_opc_snapshot(self, snap):
        """Keep streams & interval in sync every poll; lock phase (countdown) once."""
        if isinstance(snap, dict):
            getv = lambda k, d=None: snap.get(k, d)
        else:
            getv = lambda k, d=None: getattr(snap, k, d)

        # --- 1) Streams → per-zone totals (every poll) ---
        streams = {
            "feed":    float(getv("FeedFlow",    self.stream_flows.get("feed", 0.0))),
            "eluent":  float(getv("EluentFlow",  self.stream_flows.get("eluent", 0.0))),
            "recycle": float(getv("RecycleFlow", self.stream_flows.get("recycle", 0.0))),
        }
        self.stream_flows.update(streams)
        zflows = derive_zone_flows(self.stream_flows)
        for i, fr in enumerate(zflows, 1):
            self.smb.setFlowRateZone(i, float(fr))
        # keep tube α consistent with new flows (preserve concentrations)
        for z in (1, 2, 3, 4):
            qz = float(self.smb.flowRates.get(z, 0.0))
            for obj in self.smb.zones[z]:
                try:
                    from SMB.Tube import Tube
                except Exception:
                    from Tube import Tube
                if isinstance(obj, Tube) and hasattr(obj, "reconfigure"):
                    obj.reconfigure(qz, self.smb.dt)

        # --- 2) Switch interval (every poll; do NOT touch countdown) ---
        si = float(getv("SwitchInterval", self.params["switch_interval"]))
        if not math.isclose(si, float(getattr(self.smb, "interval", -1.0)), rel_tol=0.0, abs_tol=1e-12):
            self.smb.setSwitchInterval(si)

        # --- 3) Phase (countdown) — lock once; ignore tiny OPC jitter thereafter ---
        sc_raw = getv("SwitchCountdown", None)
        if sc_raw is None:
            return
        sc_opc = float(sc_raw)

        if not hasattr(self, "_phase_locked"):
            # First RUNNING snapshot: take OPC phase *as is*
            self.smb.countdown = sc_opc
            self.smb.switchingEnabled = si > 0
            self._phase_locked = {"sc0": sc_opc, "t0": float(getv("SimulationTime", 0.0))}
            log.info("[twin] Phase locked: countdown=%.3f (ABS), interval=%.3f", sc_opc, si)
            return

        # After lock: do NOT chase OPC jitter. Only respond to *true resets*.
        # Treat as reset if OPC jumps by a *large* amount or wraps near 0/interval.
        dt = float(self._dt)
        sc_twin = float(getattr(self.smb, "countdown", sc_opc))
        interval = float(getattr(self.smb, "interval", si))
        # compute minimal wrapped difference in [-interval/2, +interval/2]
        diff = sc_opc - sc_twin
        if diff >  0.5 * interval: diff -= interval
        if diff < -0.5 * interval: diff += interval

        LARGE_JUMP = max(10.0 * dt, 0.5)   # e.g., >0.5 s or >10*dt
        NEAR_WRAP  = min(5.0 * dt, 0.25)   # within 5*dt of wrap

        if abs(diff) >= LARGE_JUMP or sc_opc <= NEAR_WRAP or sc_opc >= (interval - NEAR_WRAP):
            self.smb.countdown = sc_opc
            self.smb.switchingEnabled = si > 0
            log.warning("[twin] Phase reset detected: forcing countdown from %.3f → %.3f (interval=%.3f)",
                        sc_twin, sc_opc, interval)
        # else: ignore — let the twin free-run; it advances by exact dt steps

    # ---------------- Deterministic advance ----------------
    def step_until(self, target_time: float, *, max_steps: int | None = None) -> "AdvanceInfo":
        """
        Advance the twin by an *exact* integer number of dt steps so that
        time(twin) == round(SimulationTime/dt)*dt.
        - No float accumulation.
        - No 'best effort' caps unless max_steps is explicitly given.
        """
        # --- constants & current state
        dt = float(self._dt)                          # single source of truth
        now = float(getattr(self.smb, "timer", 0.0))
        start_ticks = self._ticks                     # integer
        start_time = start_ticks * dt

        # --- compute exact target ticks from plant SimulationTime
        target_ticks = int(round(float(target_time) / dt))
        need = max(0, target_ticks - start_ticks)
        if need == 0:
            # If the SMB timer drifted from our tick time, snap it back
            if abs(now - start_time) > 1e-12:
                setattr(self.smb, "timer", start_time)
            return AdvanceInfo(steps=0, t_start=start_time, t_end=start_time, reason="already_synced")

        # --- optional cap (used only if the caller asks for it)
        to_run = int(need if (max_steps is None) else min(need, int(max_steps)))

        # --- run exactly 'to_run' steps
        for _ in range(to_run):
            # one deterministic SMB step of size dt
            self.smb.step(1)
            self._ticks += 1

        end_time = self._ticks * dt

        # --- snap SMB timer to the exact tick to avoid float drift
        # (important because internal arithmetic may accumulate 1e-16 type errors)
        setattr(self.smb, "timer", end_time)

        # --- if we were capped, say so; otherwise we are fully caught up
        reason = "capped" if to_run < need else "caught_up"
        return AdvanceInfo(steps=to_run, t_start=start_time, t_end=end_time, reason=reason)

    # ---------------- EKF profile I/O (2‑component layout) ----------------

    def get_switch_phase(self) -> dict:
        """Lightweight phase snapshot for debug logs."""
        S = self.smb
        return {
            "countdown": float(getattr(S, "countdown", -1.0)),
            "interval":  float(getattr(S, "interval",  -1.0)),
            "switchState": int(getattr(S, "switchState", 0)),
        }

    # ---------------- Deep copy for MPC ----------------
    def deepcopy_for_mpc(self) -> SMBStation:
        return self.smb.deepCopy()

    # ---------------- Internals ----------------
    def _last_col(self, zone: int) -> LinColumn:
        objs = self.smb.zones[int(zone)]
        if not objs:
            raise RuntimeError(f"Zone {zone} has no objects; initialization failed.")
        # find the last LinColumn in the zone (robust to [Tube, Col] vs [Col, Tube])
        for obj in reversed(objs):
            if isinstance(obj, LinColumn):
                return obj  # type: ignore
        raise TypeError(f"Zone {zone} has no LinColumn at all.")


    def _init_tubes(self) -> None:
        dt = float(getattr(self.smb, "dt", self.params["dt"]))
        for z in (1, 2, 3, 4):
            q = float(self.smb.flowRates.get(z, 0.0))
            if q <= 0:
                continue
            for obj in self.smb.zones[z]:
                if isinstance(obj, Tube):
                    obj.init(q, dt)

    # ---------------- Full-state EKF support (4 tubes + 4 columns, 2 comps) ----------------
    def _iter_zone_objects(self):
        """Yield (zone, kind, obj) in the exact order used by SMBStation.zones."""
        for z in (1, 2, 3, 4):
            for obj in self.smb.zones[z]:
                if isinstance(obj, Tube):
                    kind = "tube"
                elif isinstance(obj, LinColumn):
                    kind = "col"
                else:
                    raise TypeError(f"Unexpected object type in zone {z}: {type(obj).__name__}")
                yield z, kind, obj

    def build_full_state_layout(self) -> FullStateLayout:
        """Compute the flattened layout for the *entire* SMB (tubes + columns)."""
        off = 0
        entries: List[_ObjSlice] = []
        for z, kind, obj in self._iter_zone_objects():
            c0 = np.asarray(obj.components[0].c, float).reshape(-1)
            c1 = np.asarray(obj.components[1].c, float).reshape(-1)
            n0 = c0.size; n1 = c1.size
            sl0 = slice(off, off + n0); off += n0
            sl1 = slice(off, off + n1); off += n1
            entries.append(_ObjSlice(zone=z, kind=kind, obj_id=id(obj),
                                     comp_slices=(sl0, sl1)))
        return FullStateLayout(dim=off, objects=entries)

    def get_profiles_full(self, layout: Optional[FullStateLayout] = None) -> np.ndarray:
        L = layout or self.build_full_state_layout()
        x = np.empty(L.dim, dtype=float)
        for entry in L.objects:
            # find the object again in current topology by identity
            obj = None
            for _, _, o in self._iter_zone_objects():
                if id(o) == entry.obj_id:
                    obj = o; break
            if obj is None:
                raise RuntimeError("Object identity not found while flattening (topology changed unexpectedly).")
            x[entry.comp_slices[0]] = np.asarray(obj.components[0].c, float).reshape(-1)
            x[entry.comp_slices[1]] = np.asarray(obj.components[1].c, float).reshape(-1)
        return x

    def set_profiles_full(self, x: np.ndarray, layout: Optional[FullStateLayout] = None) -> None:
        """Restore the *entire* SMB state from a flattened vector (nonnegative clamp)."""
        L = layout or self.build_full_state_layout()
        x = np.asarray(x, float).reshape(-1)
        if x.size != L.dim:
            raise ValueError(f"Full state length mismatch: expected {L.dim}, got {x.size}")

        for entry in L.objects:
            # locate current object by id (zone may have changed due to rotate)
            target = None
            for _, _, o in self._iter_zone_objects():
                if id(o) == entry.obj_id:
                    target = o; break
            if target is None:
                raise RuntimeError("Object identity not found while restoring (topology changed unexpectedly).")
            s0, s1 = entry.comp_slices
            target.components[0].c[:] = np.maximum(x[s0], 0.0)
            target.components[1].c[:] = np.maximum(x[s1], 0.0)

    def outlet_indices_full(self, layout: Optional[FullStateLayout] = None) -> Tuple[int, int, int, int]:
        """Return indices (ExMan, ExGal, RaMan, RaGal) selecting the *last cell* of
        the last column in Z1 and Z3 within the full state vector."""
        L = layout or self.build_full_state_layout()

        # find the last LinColumn in zone 1 and 3
        z1_last = None; z3_last = None
        for e in L.objects:
            if e.kind == "col":
                if e.zone == 1:
                    z1_last = e  # entries are in zone order; last assignment wins
                elif e.zone == 3:
                    z3_last = e
        if z1_last is None or z3_last is None:
            raise RuntimeError("Could not locate last columns for zones 1 and/or 3.")

        s1_m, s1_g = z1_last.comp_slices
        s3_m, s3_g = z3_last.comp_slices
        i_ex_man = s1_m.stop - 1
        i_ex_gal = s1_g.stop - 1
        i_ra_man = s3_m.stop - 1
        i_ra_gal = s3_g.stop - 1
        return i_ex_man, i_ex_gal, i_ra_man, i_ra_gal

    def permutation_from_layouts(
        self, prev: FullStateLayout, curr: Optional[FullStateLayout] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a permutation that maps *previous* full vector into the *current* topology.
        Returns (src_idx, dst_idx) such that x_curr[dst_idx] = x_prev[src_idx].

        Assumes tube/column internal sizes did not change (true if dt,Nx fixed and
        tubes were not re-initialized).
        """
        C = curr or self.build_full_state_layout()
        # index objects by identity in each layout
        prev_map = {e.obj_id: e for e in prev.objects}
        curr_map = {e.obj_id: e for e in C.objects}

        src_idx = []
        dst_idx = []
        for obj_id, e_prev in prev_map.items():
            e_curr = curr_map.get(obj_id)
            if e_curr is None:
                raise RuntimeError("Object identity missing in current layout (unexpected reallocation).")
            # Man component block
            ps0, pc0 = e_prev.comp_slices[0], e_curr.comp_slices[0]
            if (ps0.stop - ps0.start) != (pc0.stop - pc0.start):
                raise RuntimeError("Component length changed between layouts; cannot permute safely.")
            src_idx.extend(range(ps0.start, ps0.stop))
            dst_idx.extend(range(pc0.start, pc0.stop))
            # Gal component block
            ps1, pc1 = e_prev.comp_slices[1], e_curr.comp_slices[1]
            if (ps1.stop - ps1.start) != (pc1.stop - pc1.start):
                raise RuntimeError("Component length changed between layouts; cannot permute safely.")
            src_idx.extend(range(ps1.start, ps1.stop))
            dst_idx.extend(range(pc1.start, pc1.stop))

        return np.asarray(src_idx, dtype=int), np.asarray(dst_idx, dtype=int)

    def full_state_dim(self) -> int:
        return self.build_full_state_layout().dim

    def flow_snapshot(self) -> dict:
        """Return current stream and per-zone flows + basic switch phase for debugging."""
        # streams (what we last applied from OPC or defaults)
        streams = dict(self.stream_flows)
        # zone totals (what SMBStation actually uses internally)
        zones = {z: float(self.smb.flowRates.get(z, 0.0)) for z in (1, 2, 3, 4)}
        return {
            "streams": streams,                       # keys: feed, eluent, recycle, extract (extract not used for zones)
            "zones": zones,                           # keys: 1..4
            "interval": float(getattr(self.smb, "interval", -1.0)),
            "countdown": float(getattr(self.smb, "countdown", -1.0)),
            "switchState": int(getattr(self.smb, "switchState", 0)),
            "dt": float(getattr(self.smb, "dt", self.params.get("dt", 0.0))),
        }

    # --- NEW: integer-tick utilities ---
    def plant_tick_from_time(self, sim_time: float) -> int:
        """Map plant SimulationTime to an integer tick on our dt grid."""
        return int(round(float(sim_time) / float(self._dt)))

    def get_tick(self) -> int:
        """Current twin integer tick."""
        return int(self._ticks)

    def step_to_tick(self, target_tick: int, *, max_steps: int | None = None) -> "AdvanceInfo":
        """
        Advance by an *exact* number of steps so that get_tick() == target_tick.
        Snaps the SMB timer to tick*dt after advancing to remove float drift.
        """
        dt = float(self._dt)
        start_ticks = int(self._ticks)
        need = max(0, int(target_tick) - start_ticks)
        if need == 0:
            t = start_ticks * dt
            # snap timer exactly to grid if needed
            if abs(float(getattr(self.smb, "timer", 0.0)) - t) > 1e-12:
                setattr(self.smb, "timer", t)
            return AdvanceInfo(steps=0, t_start=t, t_end=t, reason="already_synced")

        to_run = need if (max_steps is None) else min(need, int(max_steps))
        for _ in range(to_run):
            self.smb.step(1)
            self._ticks += 1

        end_time = self._ticks * dt
        setattr(self.smb, "timer", end_time)
        return AdvanceInfo(steps=to_run, t_start=(start_ticks * dt), t_end=end_time,
                           reason=("caught_up" if to_run == need else "capped"))





# Convenience factory used by tests or bootstrap
def make_default_twin() -> DigitalTwin:
    return DigitalTwin(DEFAULT_PARAMS, DEFAULT_COMPONENTS, DEFAULT_FLOWS)
