"""
EKF Manager — supervisor/orchestrator for the SMB state estimator
-----------------------------------------------------------------

Key change in this version:
• EKF tick schedule is driven by PLANT SimulationTime (OPC).
• Wall-time sleeping is paced using OPC SpeedFactor.
• Twin is advanced to the latest SimulationTime each loop.

Full-state support:
• When ManagerConfig.full_state is True, the manager calls CN adapter
  methods build_A_sequence_full(...) and build_selector_S_full(...).
• Switch-edge blackout/reseeding logic is removed; it is no longer needed
  because the full-state A_seq includes permutation matrices Π at switches.
"""
from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, asdict
from typing import Any, Optional

try:
    import numpy as np  # noqa: F401
except Exception:  # pragma: no cover
    np = None  # type: ignore


# -----------------------------------------------------------------------------#
# Logging
# -----------------------------------------------------------------------------#
LOGGER_NAME = "EKF"
log = logging.getLogger(LOGGER_NAME)

def _setup_default_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        log.setLevel(level)
        log.propagate = True
        return
    if log.handlers:
        log.propagate = False
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][EKF] %(message)s"))
    log.addHandler(h)
    log.setLevel(level)
    log.propagate = False


# -----------------------------------------------------------------------------#
# OPC snapshot
# -----------------------------------------------------------------------------#
@dataclass
class OpcSnapshot:
    # status & timing
    IsRunning: bool
    SimulationTime: float
    SwitchCountdown: float
    ElapsedWallTime: float
    SpeedFactor: float

    # operating params
    FeedFlow: float
    ExtractFlow: float
    RecycleFlow: float
    EluentFlow: float
    SwitchInterval: float

    # measurements
    ExtractConcentration_Man: float
    ExtractConcentration_Gal: float
    RaffinateConcentration_Man: float
    RaffinateConcentration_Gal: float

    # auxiliary
    ExtractOutletFlow: float
    RaffinateOutletFlow: float

    @staticmethod
    def from_mapping(m: dict[str, Any]) -> "OpcSnapshot":
        return OpcSnapshot(
            IsRunning=bool(m["IsRunning"]),
            SimulationTime=float(m["SimulationTime"]),
            SwitchCountdown=float(m["SwitchCountdown"]),
            ElapsedWallTime=float(m["ElapsedWallTime"]),
            SpeedFactor=float(m["SpeedFactor"]),
            FeedFlow=float(m["FeedFlow"]),
            ExtractFlow=float(m["ExtractFlow"]),
            RecycleFlow=float(m["RecycleFlow"]),
            EluentFlow=float(m["EluentFlow"]),
            SwitchInterval=float(m["SwitchInterval"]),
            ExtractConcentration_Man=float(m["ExtractConcentration_Man"]),
            ExtractConcentration_Gal=float(m["ExtractConcentration_Gal"]),
            RaffinateConcentration_Man=float(m["RaffinateConcentration_Man"]),
            RaffinateConcentration_Gal=float(m["RaffinateConcentration_Gal"]),
            ExtractOutletFlow=float(m["ExtractOutletFlow"]),
            RaffinateOutletFlow=float(m["RaffinateOutletFlow"]),
        )

    def y_meas(self) -> list[float]:
        return [
            self.ExtractConcentration_Man,
            self.ExtractConcentration_Gal,
            self.RaffinateConcentration_Man,
            self.RaffinateConcentration_Gal,
        ]


# -----------------------------------------------------------------------------#
# Manager config
# -----------------------------------------------------------------------------#
@dataclass
class ManagerConfig:
    dt_model: float = 0.05        # SMBStation internal dt
    opc_poll: float = 0.5         # min wall-time between polls
    ekf_period: float = 15.0      # EKF period in SIMULATION TIME seconds

    # NEW: choose full-state wiring (4 columns) vs legacy (2-column)
    full_state: bool = True

    # legacy knobs retained (mostly irrelevant with full_state=True)
    switch_apply_on_next: bool = True
    catchup_cap_steps: Optional[int] = None
    tick_on_start: bool = True
    idle_log_period: float = 30.0

    log_level: int = logging.INFO
    csv_path: Optional[str] = None


# -----------------------------------------------------------------------------#
# EKF Manager (SimulationTime–driven)
# -----------------------------------------------------------------------------#
class EKFManager(threading.Thread):
    def __init__(self, opc_client: Any, twin: Any, ekf_core: Any,
                 cn_adapter: Any, sensor_model: Any, config: ManagerConfig | None = None) -> None:
        super().__init__(daemon=True)
        self.opc = opc_client
        self.twin = twin
        self.ekf = ekf_core
        self.cn = cn_adapter
        self.sensors = sensor_model
        self.cfg = config or ManagerConfig()

        _setup_default_logging(self.cfg.log_level)

        self._stop = threading.Event()
        self._loop_counter = 0
        self._tick_counter = 0
        self._last_running: bool | None = None
        self._last_snapshot: Optional[OpcSnapshot] = None
        self._last_diag: dict[str, Any] = {}
        self._idle_next_log = 0.0

        # TICK ANCHOR IN SIMULATION TIME (not twin time)
        self._last_tick_sim_time: Optional[float] = None

        # keep countdown for logging/diagnostics
        self._last_sc: float | None = None
        self._last_flow_sig: Optional[tuple] = None
        self._last_outlet_log_t: float = 0.0
        self._outlet_log_period_s: float = 5.0  # throttle interval

        # optional CSV
        self._csv = None
        if self.cfg.csv_path:
            try:
                self._csv = open(self.cfg.csv_path, "w", buffering=1, newline="")
                print("wall_time,opc_sim_time,twin_time,speed_factor,tick,dt_sim,residual,nis,"
                      "innov_ExMan,innov_ExGal,innov_RaMan,innov_RaGal,"
                      "yhat_ExMan,yhat_ExGal,yhat_RaMan,yhat_RaGal,"
                      "ymeas_ExMan,ymeas_ExGal,ymeas_RaMan,ymeas_RaGal",
                      file=self._csv)
            except Exception:
                log.exception("[csv] Failed to open csv_path=%s", self.cfg.csv_path)
                self._csv = None

    # ----- control -----
    def stop(self) -> None:
        self._stop.set()

    def get_state_for_mpc(self) -> dict[str, Any]:
        station_copy = self.twin.deepcopy_for_mpc()
        diag = {
            "tick": self._tick_counter,
            "loop": self._loop_counter,
            "last_diag": self._last_diag,
            "last_snapshot": asdict(self._last_snapshot) if self._last_snapshot else None,
            "twin_time": self.twin.get_time(),
            "switch_phase": self.twin.get_switch_phase(),
        }
        return {"station": station_copy, "diag": diag}

    # ----- main loop -----
    def run(self) -> None:
        log.info("Manager started (dt_model=%.3fs, opc_poll=%.2fs, ekf_period=%.2fs, full_state=%s)",
                 self.cfg.dt_model, self.cfg.opc_poll, self.cfg.ekf_period, self.cfg.full_state)
        self._last_tick_sim_time = None

        while not self._stop.is_set():
            loop_t0 = time.time()
            try:
                snap = self._read_opc()

                if not snap.IsRunning:
                    self._maybe_idle_log()
                else:
                    self._idle_next_log = time.time() + self.cfg.idle_log_period

                self._handle_running_transition(snap)
                self._sync_and_advance_twin(snap)
                self._maybe_run_ekf_tick(snap)

            except Exception as exc:  # pragma: no cover
                log.exception("[loop] Unhandled error: %s", exc)

            self._loop_counter += 1
            time.sleep(self._compute_sleep_wall(snap) if self._last_tick_sim_time is not None else self.cfg.opc_poll)

        log.info("Manager stopped after %d loops, %d ticks", self._loop_counter, self._tick_counter)
        try:
            if self._csv:
                self._csv.close()
        except Exception:
            pass

    # ----- internals -----
    def _read_opc(self) -> OpcSnapshot:
        raw = self.opc.read()
        snap = raw if isinstance(raw, OpcSnapshot) else OpcSnapshot.from_mapping(raw)
        self._last_snapshot = snap
        log.debug("[opc] %s", asdict(snap))
        return snap

    def _handle_running_transition(self, snap: OpcSnapshot) -> None:
        if self._last_running is None:
            self._last_running = snap.IsRunning
            log.info("[sync] Initial IsRunning=%s", snap.IsRunning)
            log.info("[sync] Plant snapshot: t=%.2f s, countdown=%.2f s, interval=%.2f s",
                     snap.SimulationTime, snap.SwitchCountdown, snap.SwitchInterval)
            if not snap.IsRunning:
                log.info("[idle] Plant is not running. EKF will wait until IsRunning=True.")
            if snap.IsRunning:
                self._on_startup(snap)
            return

        if snap.IsRunning and not self._last_running:
            log.info("[sync] Plant transitioned to RUNNING. Aligning twin and sensors…")
            self._on_startup(snap)
        elif (not snap.IsRunning) and self._last_running:
            log.warning("[sync] Plant transitioned to STOPPED. Twin will idle; EKF paused.")
        self._last_running = snap.IsRunning

    def _on_startup(self, snap: OpcSnapshot) -> None:
        # 1) Apply params + phase align
        self.twin.apply_opc_snapshot(snap)

        # Catch twin up exactly by ticks
        plant_tick = self.twin.plant_tick_from_time(snap.SimulationTime)
        adv = self.twin.step_to_tick(plant_tick, max_steps=self.cfg.catchup_cap_steps)
        log.info("[sync] Twin caught up to plant tick=%d (%s)", plant_tick, adv)

        # 3) Sensor cold start
        try:
            self.sensors.reset(self._extract_y_meas_vector(snap))
        except Exception:
            log.exception("[sync] Sensor reset failed")

        # --- NEW: force EKF dimension reset ---
        try:
            self.ekf._N = None
            self.ekf._P = None
            self.ekf._z = None
            log.info("[sync] EKF core reset (dimension will re-init on next tick)")
        except Exception:
            pass

        # 4) EKF tick anchor
        t_sim = float(snap.SimulationTime)
        if self.cfg.tick_on_start and self.cfg.ekf_period > 0:
            self._last_tick_sim_time = max(0.0, t_sim - self.cfg.ekf_period)
        else:
            self._last_tick_sim_time = t_sim

    @staticmethod
    def _opc_streams_from_snapshot(snap) -> dict:
        return {
            "feed":    float(getattr(snap, "FeedFlow", 0.0)),
            "eluent":  float(getattr(snap, "EluentFlow", 0.0)),
            "recycle": float(getattr(snap, "RecycleFlow", 0.0)),
            "extract": float(getattr(snap, "ExtractFlow", 0.0)),
        }

    @staticmethod
    def _zones_from_streams(streams: dict) -> dict:
        # SMB topology: Z1=recycle+eluent, Z2=recycle, Z3=recycle+feed, Z4=recycle
        rec = float(streams.get("recycle", 0.0))
        fee = float(streams.get("feed", 0.0))
        elu = float(streams.get("eluent", 0.0))
        return {1: rec + elu, 2: rec, 3: rec + fee, 4: rec}


    def _sync_and_advance_twin(self, snap: OpcSnapshot) -> None:
        if not snap.IsRunning:
            return
        # keep params in sync (flows, interval)
        self.twin.apply_opc_snapshot(snap)

        # --- Flow alignment log (only when something changes) ---
        opc_streams = self._opc_streams_from_snapshot(snap)
        opc_zones   = self._zones_from_streams(opc_streams)
        twin_flow   = self.twin.flow_snapshot()

        sig = (
            round(opc_streams["feed"], 6), round(opc_streams["eluent"], 6),
            round(opc_streams["recycle"], 6), round(float(getattr(snap, "SwitchInterval", 0.0)), 6),
            round(twin_flow["streams"]["feed"], 6), round(twin_flow["streams"]["eluent"], 6),
            round(twin_flow["streams"]["recycle"], 6), round(twin_flow["interval"], 6),
        )
        if sig != self._last_flow_sig:
            self._last_flow_sig = sig
            log.info(
                "[flows] OPC streams={feed=%.3f, eluent=%.3f, recycle=%.3f} "
                "OPC zones={Z1=%.3f,Z2=%.3f,Z3=%.3f,Z4=%.3f} | "
                "TWIN streams={feed=%.3f, eluent=%.3f, recycle=%.3f} "
                "TWIN zones={Z1=%.3f,Z2=%.3f,Z3=%.3f,Z4=%.3f} "
                "interval(opc=%.3f,twin=%.3f) countdown=%.2f switchState=%d",
                opc_streams["feed"], opc_streams["eluent"], opc_streams["recycle"],
                opc_zones[1], opc_zones[2], opc_zones[3], opc_zones[4],
                twin_flow["streams"]["feed"], twin_flow["streams"]["eluent"], twin_flow["streams"]["recycle"],
                twin_flow["zones"][1], twin_flow["zones"][2], twin_flow["zones"][3], twin_flow["zones"][4],
                float(getattr(snap, "SwitchInterval", 0.0)), twin_flow["interval"],
                twin_flow["countdown"], twin_flow["switchState"],
            )

        # --- Advance twin to the plant SimulationTime ONCE per loop ---
        adv = self._twin_step_until(snap.SimulationTime)   # your helper -> twin.step_to_tick(...)

        # --- Always show a clear sync line with exact steps run ---
        try:
            ptick = self.twin.plant_tick_from_time(snap.SimulationTime)
            ttick = self.twin.get_tick()
            dt    = float(self.twin._dt)
            p_t   = ptick * dt
            t_t   = ttick * dt
            ran   = int(getattr(adv, "steps", -1))
            if ptick == ttick:
                log.info("[sync] aligned tick=%d t=%.3fs (ran=%d)", ttick, t_t, ran)
            else:
                log.info("[sync] MISALIGNED plant_tick=%d (t=%.3fs) twin_tick=%d (t=%.3fs) Δ=%+d (ran=%d)",
                         ptick, p_t, ttick, t_t, ptick - ttick, ran)
        except Exception:
            pass

        # read-only phase delta (wrapped), for visibility only
        try:
            sc_opc   = float(snap.SwitchCountdown)
            sc_twin  = float(getattr(self.twin.smb, "countdown", sc_opc))
            interval = float(getattr(self.twin.smb, "interval", snap.SwitchInterval))
            d = sc_opc - sc_twin
            if d >  0.5 * interval: d -= interval
            if d < -0.5 * interval: d += interval
            log.debug("[sync] phaseΔ=%.3fs (opc %.3f vs twin %.3f, interval=%.3f)", d, sc_opc, sc_twin, interval)
        except Exception:
            pass

    def _maybe_run_ekf_tick(self, snap: OpcSnapshot) -> None:
        if not snap.IsRunning or self._last_tick_sim_time is None:
            try:
                self._last_sc = float(snap.SwitchCountdown)
            except Exception:
                pass
            return

        t_sim_now = float(snap.SimulationTime)
        dt_sim = t_sim_now - self._last_tick_sim_time
        if dt_sim + 1e-9 < self.cfg.ekf_period:
            try:
                self._last_sc = float(snap.SwitchCountdown)
            except Exception:
                pass
            return

        # ---------- Build CN transition sequence and selector (full-state only) ----------
        t_start = self._last_tick_sim_time
        t_end = self.twin.get_time()  # should equal t_sim_now after sync

        try:
            A_seq = self.cn.build_A_sequence_full(self.twin, t_start, t_end)
            S     = self.cn.build_selector_S_full(self.twin)
        except Exception:
            log.exception("[tick] CN adapter failed to build A_seq/S — skipping this tick")
            self._last_tick_sim_time = t_sim_now
            self._last_sc = float(getattr(snap, "SwitchCountdown", 0.0))
            return

        y_meas = self._extract_y_meas_vector(snap)

        # ---------- EKF step ----------
        try:
            diag = self.ekf.tick(
                A_seq=A_seq, S=S, y_meas=y_meas,
                sensor_model=self.sensors, twin=self.twin,
                dt_model=self.cfg.dt_model, t_start=t_start, t_end=t_end
            )
        except Exception:
            log.exception("[tick] EKF core error — keeping previous state")
            diag = {"status": "ekf_error"}

        # ---------- bookkeeping & verbose log ----------
        self._tick_counter += 1
        self._last_tick_sim_time = t_sim_now
        self._last_sc = float(getattr(snap, "SwitchCountdown", 0.0))
        self._last_diag = {"tick": self._tick_counter, "elapsed_sim": dt_sim, **(diag or {})}

        def _vec(key):
            v = self._last_diag.get(key)
            if v is None:
                return [float("nan")] * 4
            try:
                return [float(x) for x in v]
            except Exception:
                return [float("nan")] * 4

        innov = _vec("innov")
        y_pred = _vec("y_pred")
        y_meas_vec = _vec("y_meas")
        residual = float(self._last_diag.get("residual", float("nan")))
        nis = float(self._last_diag.get("nis", float("nan")))
        sf = float(getattr(snap, "SpeedFactor", 1.0))
        sc = float(getattr(snap, "SwitchCountdown", 0.0))
        si = float(getattr(snap, "SwitchInterval", 0.0))

        twin_info = {
            "countdown": float(getattr(self.twin.smb, "countdown", 0.0)),
            "interval":  float(getattr(self.twin.smb, "interval", 0.0)),
            "switchState": int(getattr(self.twin.smb, "switchState", 0)),
        }
        opc_info = {"opc_countdown": sc, "opc_interval": si}

        names = ["ExMan", "ExGal", "RaMan", "RaGal"]
        innov_str = "[" + ", ".join(f"{n}={v:.3e}" for n, v in zip(names, innov)) + "]"

        log.info(
            "[EKF] [tick %d] Δt=%.2fs, residual=%.3e, NIS=%.3e, sf=%.3f, "
            "switch_twin=%s, switch_opc=%s, innov=%s",
            self._tick_counter, dt_sim, residual, nis, sf, twin_info, opc_info, innov_str
        )

        if self._csv:
            try:
                t_wall = time.time()
                twin_t = self.twin.get_time()
                def f4(v): return [float(x) for x in (v if isinstance(v, (list, tuple)) else [float("nan")]*4)]
                inv = f4(innov); yp = f4(y_pred); ym = f4(y_meas_vec)
                print(f"{t_wall:.3f},{t_sim_now:.6f},{twin_t:.6f},{sf:.3f},"
                      f"{self._tick_counter},{dt_sim:.3f},{residual:.6e},{nis:.6e},"
                      f"{inv[0]:.6e},{inv[1]:.6e},{inv[2]:.6e},{inv[3]:.6e},"
                      f"{yp[0]:.6e},{yp[1]:.6e},{yp[2]:.6e},{yp[3]:.6e},"
                      f"{ym[0]:.6e},{ym[1]:.6e},{ym[2]:.6e},{ym[3]:.6e}",
                      file=self._csv)
            except Exception:
                log.exception("[csv] write failed")


    def _compute_sleep_wall(self, snap: OpcSnapshot) -> float:
        """
        Compute how long to sleep in WALL TIME:
        we want to wake up often enough to catch the next SIM tick,
        but not hammer the OPC server.
        """
        min_poll = float(self.cfg.opc_poll)
        try:
            sf = max(1e-6, float(snap.SpeedFactor))
            t_sim_now = float(snap.SimulationTime)
            t_next_tick = float(self._last_tick_sim_time or t_sim_now) + self.cfg.ekf_period
            sim_remaining = max(0.0, t_next_tick - t_sim_now)
            wall_until_tick = sim_remaining / sf   # convert using SpeedFactor
            return max(min_poll, min(wall_until_tick * 0.5, 1.5 * min_poll))
        except Exception:
            return min_poll

    def _twin_step_until(self, target_sim_time: float):
        """Advance the twin to the plant's integer tick derived from SimulationTime."""
        plant_tick = self.twin.plant_tick_from_time(target_sim_time)
        # remove caps unless you explicitly set one in cfg
        return self.twin.step_to_tick(plant_tick, max_steps=self.cfg.catchup_cap_steps)


    def _maybe_idle_log(self) -> None:
        now = time.time()
        if now >= self._idle_next_log:
            log.info("[idle] Plant not running (IsRunning=False). Waiting idle…")
            self._idle_next_log = now + self.cfg.idle_log_period


    @staticmethod
    def _extract_y_meas_vector(snap: OpcSnapshot) -> list[float]:
        return snap.y_meas()


# -----------------------------------------------------------------------------#
if __name__ == "__main__":  # pragma: no cover
    _setup_default_logging(logging.DEBUG)
    log.info("EKF Manager module loaded.")
