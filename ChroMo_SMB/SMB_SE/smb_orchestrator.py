"""
smb_orchestrator.py
-------------------
High-level runner that wires together:
  - OPC UA client (reads plant state & setpoints)
  - Digital twin engine (simulates in lockstep with plant)
  - Optional live visualization
  - CSV/console logger

Designed for chemical engineers: names, units, and prints are plain-English.

Default behavior:
- Refuses to attach if the plant is already running (to avoid mid-run attach).
- Waits until the server flips IsRunning=True, then starts the twin + viz + logging.

You can override to attach mid-run with --attach-if-running.

Units (defaults):
  dt = 0.25 s         (simulation step)
  Nx = 100            (spatial nodes per column)
  si = 1894 steps     (switch interval, in time steps)
  L  = 320 mm         (column length)
  D  = 10 mm          (column inner diameter)
  eps= 0.376          (bed porosity)
  dv = 0.5 (vol units consistent with engine; tube dead volume)

"""

from __future__ import annotations

import argparse
import sys
import os
import threading
from datetime import datetime
import time
import signal
from typing import Optional, Dict, Any, Iterable

# Matplotlib is optional; plotting can be disabled with --no-viz
import matplotlib
import matplotlib.pyplot as plt

from ChroMo_SMB.SMB_PlantSim.opcua_client import SMB_OPCUAClient
from smb_engine import SMBTwinEngine
from smb_viz import attach_viz
from smb_logging import SMBTwinLogger, build_filename
from smb_sensor import SMBOutletSensor, OutletTagMap
from smb_klo import SMBKLOManager

REQUIRED_RO: set[str] = {
    "IsRunning", "SimulationTime", "ElapsedWallTime", "SwitchCountdown", "SpeedFactor",
    "ActiveFeed", "ActiveQ1", "ActiveQ2", "ActiveQ4", "ActiveSwitchInterval",
    "ExtractConcentration_Man", "ExtractConcentration_Gal",
    "RaffinateConcentration_Man", "RaffinateConcentration_Gal",
}
OPTIONAL_RW: set[str] = {"Feed", "Q1", "Q2", "Q4", "SwitchInterval"}

# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="smb_orchestrator",
        description="Run the SMB digital twin against the OPC UA plant (with optional viz & logging).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # OPC UA connection
    p.add_argument("--endpoint", default="opc.tcp://127.0.0.1:4840",
                   help="OPC UA endpoint")
    p.add_argument("--object", dest="obj", default="SMB_Simulation",
                   help="Server browse name for the SMB object")

    # Numerics & geometry
    p.add_argument("--dt", type=float, default=1.0, help="Simulation time step [s]")
    p.add_argument("--Nx", type=int,   default=100,  help="Spatial grid points per column")
    p.add_argument("--si", type=float, default=780.0,
                   help="Switch interval [time steps, i.e., multiples of dt]")
    p.add_argument("--L",  type=float, default=310.0, help="Column length [mm]")
    p.add_argument("--D",  type=float, default=10.0,  help="Column inner diameter [mm]")
    p.add_argument("--eps", type=float, default=0.376, help="Bed porosity [-]")
    p.add_argument("--dv",  type=float, default=0.2,   help="Tube dead volume [engine units]")

    # Engine cadence
    p.add_argument("--max-steps", type=int, default=1,
                   help="Max simulation steps per engine update (keeps UI responsive)")
    p.add_argument("--poll", type=float, default=0.25,
                   help="OPC polling period [s]")
    p.add_argument("--apply-mode", default="boundary", choices=["boundary", "immediate"],
                   help="When to apply setpoint changes: at column-switch boundaries (boundary) or instantly (immediate)")

    # Viz & logging
    p.add_argument("--no-viz", action="store_true", help="Disable the live plot window")
    p.add_argument("--redraw-ms", type=int, default=300, help="Live plot refresh period [ms]")
    p.add_argument("--max-ts-points", type=int, default=800,
                   help="Max time-series points kept in the live plot")

    p.add_argument("--log-period", type=float, default=5.0,
                   help="CSV log interval [s]")
    p.add_argument("--print-period", type=float, default=30.0,
                   help="Terminal diff print interval [s]")
    p.add_argument("--csv-prefix", default="smb_log",
                   help="CSV filename prefix")

    # Snapshot handover
    p.add_argument("--handover-dir", default=os.path.join(os.getcwd(), "handover"),
                   help="Folder where the engine will write deep-copy snapshots (state_s{k}.pkl)")

    # Attach behavior
    p.add_argument("--attach-if-running", action="store_true",
                   help="Allow attach mid-run if plant IsRunning=True")

    return p.parse_args()


# ----------------------------
# Utilities
# ----------------------------
def _ensure_gui_backend() -> None:
    """
    If the current backend is headless (Agg), try to switch to TkAgg
    so engineers get a window by default. If that fails, we keep Agg.
    """
    try:
        if "agg" in matplotlib.get_backend().lower():
            matplotlib.use("TkAgg")
    except Exception:
        pass


def _graceful_shutdown(logger: Optional[SMBTwinLogger],
                       eng: Optional[SMBTwinEngine],
                       cli: Optional[SMB_OPCUAClient]) -> None:
    # Stop logger first so it stops reading the client/engine
    try:
        if logger is not None:
            logger.stop()
    except Exception:
        pass
    # Then stop engine
    try:
        if eng is not None:
            eng.stop()
    except Exception:
        pass
    # Finally disconnect OPC
    try:
        if cli is not None:
            cli.disconnect()
    except Exception:
        pass

def _require_keys(snap: Dict[str, Any], keys: Iterable[str]) -> list[str]:
    return [k for k in keys if k not in snap]

def _validate_contract(cli: SMB_OPCUAClient) -> None:
    snap = cli.read_snapshot()
    missing_ro = _require_keys(snap, REQUIRED_RO)
    missing_rw = _require_keys(snap, OPTIONAL_RW)
    if missing_ro:
        print("[orchestrator] OPC UA contract check failed:\n  Missing read-only keys: "
              + ", ".join(sorted(missing_ro)), file=sys.stderr)
        sys.exit(3)
    if missing_rw:
        print("[orchestrator] Note: missing optional writable keys: "
              + ", ".join(sorted(missing_rw)))

class PlantMonitor:
    def __init__(self, cli: SMB_OPCUAClient, poll_s: float):
        self.cli = cli
        self.poll_s = max(0.2, float(poll_s))
        self._t = None
        self._stop = threading.Event()
        self._started = False
        self._prev_running = None
        self._prev_mode = None
        self._prev_sf = None
        self._prev_sc_shown = None
        self.heartbeat_s = 1e9
        self._last_print = 0.0
        self._last_line = None
        self._last_line_ts = 0.0

    def start(self):
        if self._started:
            return
        self._started = True
        self._t = threading.Thread(target=self._loop, name="PlantMonitor", daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        self._print_once("[monitor] Watching OPC UA: running/state/countdown…")
        while not self._stop.is_set():
            try:
                snap = self.cli.read_snapshot()
                running = bool(snap.get("IsRunning", False))
                mode = snap.get("Mode", "UNKNOWN")
                sc = float(snap.get("SwitchCountdown", 0.0))
                sf = max(float(snap.get("SpeedFactor", 1.0)), 1e-9)
                budget_wall = max(0.0, sc / sf)
                F  = snap.get("ActiveFeed"); Q1 = snap.get("ActiveQ1"); Q2 = snap.get("ActiveQ2")
                Q4 = snap.get("ActiveQ4");  SI = snap.get("ActiveSwitchInterval")

                now_ts = time.time()
                sc_round = int(sc)
                state_changed = (running != self._prev_running) or (mode != self._prev_mode) or (sf != self._prev_sf)
                heartbeat = (now_ts - self._last_print) >= self.heartbeat_s
                sc_changed = (sc_round != self._prev_sc_shown)

                if state_changed:
                    state = "RUNNING" if running else "STOPPED"
                    self._print_once(f"[monitor] Plant={state}  Mode={mode}  SpeedFactor={sf:g}")
                    if running and self._prev_running is False:
                        self._print_once("[monitor] Rising edge detected → engine will emit snapshot s0 (start).")
                    self._last_print = now_ts

                if heartbeat and sc_changed:
                    now = datetime.utcnow().strftime("%H:%M:%S")
                    self._print_once(
                        f"[monitor] {now}  SC(sim)={sc_round:d}s  SF={sf:.3g}  "
                        f"est_budget(wall)≈{budget_wall:.2f}s  "
                        f"Active(F,Q1,Q2,Q4,SI)=({F},{Q1},{Q2},{Q4},{SI})"
                    )
                    self._last_print = now_ts
                    self._prev_sc_shown = sc_round

                self._prev_running = running
                self._prev_mode = mode
                self._prev_sf = sf

            except Exception as e:
                self._print_once(f"[monitor] Warning: snapshot read failed: {e}")
            self._stop.wait(self.poll_s)

    def _print_once(self, msg: str):
        now = time.time()
        if msg == self._last_line and (now - self._last_line_ts) < self.heartbeat_s:
            return
        print(msg, flush=True)
        self._last_line = msg
        self._last_line_ts = now

# ----------------------------
# Main
# ----------------------------
def main() -> None:

    mon: Optional[PlantMonitor] = None
    args = parse_args()

    if not args.no_viz:
        _ensure_gui_backend()

    # --- Clean handover folder on orchestrator start ---
    if args.handover_dir:
        try:
            os.makedirs(args.handover_dir, exist_ok=True)
            removed = 0
            for f in os.listdir(args.handover_dir):
                if f.startswith("state_s") and f.endswith(".pkl"):
                    os.remove(os.path.join(args.handover_dir, f))
                    removed += 1
            if removed > 0:
                print(f"[orchestrator] Cleared {removed} old snapshot(s) from {args.handover_dir}")
            else:
                print(f"[orchestrator] Handover folder is clean: {args.handover_dir}")
        except Exception as e:
            print(f"[orchestrator] Warning: could not clear handover folder: {e}")

    # Build filenames & announce
    csv_name = build_filename(prefix=args.csv_prefix)
    print(f"[orchestrator] OPC endpoint: {args.endpoint}")
    print(f"[orchestrator] SMB object  : {args.obj}")
    print(f"[orchestrator] Logging to  : {csv_name}")
    if args.no_viz:
        print("[orchestrator] Live plot   : OFF")
    else:
        print("[orchestrator] Live plot   : ON")

    # One shared OPC UA client
    cli = SMB_OPCUAClient(endpoint=args.endpoint, obj_browse_name=args.obj)
    logger: Optional[SMBTwinLogger] = None
    eng: Optional[SMBTwinEngine] = None

    # Let Ctrl+C exit cleanly
    def _sigint_handler(_sig, _frm):
        print("\n[orchestrator] Ctrl+C received — stopping…")
        _graceful_shutdown(logger, eng, cli)
        sys.exit(130)
    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        cli.connect()
        _validate_contract(cli)
        snap = cli.read_snapshot()
        is_running = bool(snap.get("IsRunning", False))
        if is_running and not args.attach_if_running:
            print("Plant is already running. Start orchestrator BEFORE the plant, "
                  "or pass --attach-if-running to attach mid-run (snapshots will start from the next boundary)..")
            sys.exit(2)
        elif not is_running:
            print("Waiting for plant to start (IsRunning=True)…")

        # --- Engine ---
        eng = SMBTwinEngine(
            opc_client=cli,            # reuse this client
            manage_client=False,       # engine will not close it
            opc_endpoint=args.endpoint,
            obj_name=args.obj,
            dt=args.dt, Nx=args.Nx,
            switch_interval_default=args.si,
            col_length=args.L, col_diameter=args.D,
            porosity=args.eps, dead_volume=args.dv,
            max_steps_per_update=args.max_steps,
            poll_period_s=args.poll,
        )

        # Configure engine handover folder for snapshots s0, s1, …
        eng.snapshot_dir = args.handover_dir
        os.makedirs(eng.snapshot_dir, exist_ok=True)
        print(f"[orchestrator] Handover folder: {eng.snapshot_dir}")

        # --- Sensor (ideal) ---
        sensor = SMBOutletSensor(
            opc_client=cli,
            tags=OutletTagMap(  # adjust if your tag names differ
                extract_man="ExtractConcentration_Man",
                extract_gal="ExtractConcentration_Gal",
                raffinate_man="RaffinateConcentration_Man",
                raffinate_gal="RaffinateConcentration_Gal",
            ),
            detlim_gpl=0.01,     # detection limit only; no lag/dead-time in this milestone
        )

        klo_mgr = SMBKLOManager(
            engine=eng, sensor=sensor, Nx=args.Nx, kf_dt_multiples=1,

            # — návrh (hloubka + síla) —
            q_sigma2=5.0e-5,
            q_alpha =5.0e-1,     # ↑ na poměr 10 000 → depth ≈ 100 buněk

            # — měření —
            r_meas =5.0e-7,      # víc věřit senzoru v zóně 3 (globální); z1 přepíšeme níže

            # — zápis (hladkost) —
            alpha_injection   =0.70,
            max_cell_step_frac=0.40,
            deadband_gpl      =0.005,

            # jemnější EXTRACT (zóna 1)
            r_meas_z1=1.0e-4,
            alpha_injection_z1=0.45,
            max_cell_step_frac_z1=0.30,
            deadband_gpl_z1=0.01,

            diag=True, diag_interval_s=2.0
        )

        # --- Plant monitor (console visibility) ---
        if mon is None:
            mon = PlantMonitor(cli=cli, poll_s=args.poll)
            mon.start()

        # --- Live Viz (optional) ---
        if not args.no_viz:
            fig = attach_viz(
                eng,
                Nx=args.Nx,
                L_mm=args.L,
                si_steps=args.si,
                dt_s=args.dt,
                redraw_ms=args.redraw_ms,
                max_ts_points=args.max_ts_points,
            )
        else:
            fig = None

        # --- Logger ---
        logger = SMBTwinLogger(
            opc_client=cli,
            engine=eng,
            csv_path=csv_name,
            log_period_s=args.log_period,
            print_period_s=args.print_period
        )
        logger.start()

        # --- Start engine & (maybe) open window ---
        eng.start()
        print("[orchestrator] Engine started.")
        if fig is not None:
            print("[orchestrator] Opening live plot window…")
            plt.show()  # blocks until window closed
        else:
            # Headless mode: idle loop to keep engine + logger alive
            print("[orchestrator] Headless mode — press Ctrl+C to stop.")
            while True:
                time.sleep(1.0)

    finally:
        try:
            if mon is not None:
                mon.stop()
        except Exception:
            pass
        _graceful_shutdown(logger, eng, cli)
        print("[orchestrator] Stopped.")


if __name__ == "__main__":
    main()
