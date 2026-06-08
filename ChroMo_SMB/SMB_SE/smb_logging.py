"""
CSV logger for the SMB digital twin.

What it does
------------
• Logs, at a fixed period, the **plant** and **twin** outlet concentrations
  (Extract & Raffinate; Man & Gal) + their differences (twin − plant).
• Also logs operating setpoints (SpeedFactor, Feed/Eluent/Extract/Recycle,
  SwitchInterval) and the twin countdown for context.
• Prints a short diff summary to the terminal at a slower cadence.

Assumptions
-----------
• Component 0 = "Man", Component 1 = "Gal".
• Plant outlet tag names are discovered once at startup (best‑effort); override
  `OutletTagResolver.PRIORITY` for definitive names in your server.
"""

from __future__ import annotations
import csv, re, sys, time, threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np

from ChroMo_SMB.SMB_PlantSim.opcua_client import SMB_OPCUAClient
from smb_engine import SMBTwinEngine

# Fixed component labels
COMP0_NAME = "Man"
COMP1_NAME = "Gal"


def _as_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _pick(snap: Dict[str, float], *names: str, default: float = 0.0) -> float:
    for n in names:
        if n in snap:
            return _as_float(snap[n], default)
    return float(default)


class OutletTagResolver:
    """Resolve plant outlet concentration tag names for Man/Gal, Extract/Raffinate."""
    PRIORITY = {
        "ex_man": ["ExtractConcentration_Man", "Extract_Man", "ExtractMan", "ExtractA", "ExtractA_Conc"],
        "ex_gal": ["ExtractConcentration_Gal", "Extract_Gal", "ExtractGal", "ExtractB", "ExtractB_Conc"],
        "ra_man": ["RaffinateConcentration_Man", "Raffinate_Man", "RaffinateMan", "RaffinateA", "RaffinateA_Conc"],
        "ra_gal": ["RaffinateConcentration_Gal", "Raffinate_Gal", "RaffinateGal", "RaffinateB", "RaffinateB_Conc"],
    }
    PATTERNS = {
        "ex_man": [r"^Extract.*(Man|A)\b", r"^EX.*(Man|A)\b"],
        "ex_gal": [r"^Extract.*(Gal|B)\b", r"^EX.*(Gal|B)\b"],
        "ra_man": [r"^Raffinate.*(Man|A)\b", r"^RA.*(Man|A)\b"],
        "ra_gal": [r"^Raffinate.*(Gal|B)\b", r"^RA.*(Gal|B)\b"],
    }

    def __init__(self) -> None:
        self.map: Dict[str, Optional[str]] = {k: None for k in self.PRIORITY.keys()}

    def resolve(self, snap: Dict[str, float]) -> None:
        keys = set(snap.keys())
        # exact priority
        for k, cand_list in self.PRIORITY.items():
            for c in cand_list:
                if c in keys:
                    self.map[k] = c
                    break
        # regex fallback
        for k, patt_list in self.PATTERNS.items():
            if self.map[k] is not None:
                continue
            for patt in patt_list:
                rgx = re.compile(patt, re.IGNORECASE)
                for s in keys:
                    if rgx.match(s):
                        self.map[k] = s
                        break
                if self.map[k] is not None:
                    break

    def selected(self) -> Dict[str, Optional[str]]:
        return dict(self.map)

    def read_extract(self, snap: Dict[str, float]) -> Tuple[float, float]:
        ex_man = _as_float(snap.get(self.map["ex_man"], 0.0), 0.0) if self.map["ex_man"] else 0.0
        ex_gal = _as_float(snap.get(self.map["ex_gal"], 0.0), 0.0) if self.map["ex_gal"] else 0.0
        return ex_man, ex_gal

    def read_raffinate(self, snap: Dict[str, float]) -> Tuple[float, float]:
        ra_man = _as_float(snap.get(self.map["ra_man"], 0.0), 0.0) if self.map["ra_man"] else 0.0
        ra_gal = _as_float(snap.get(self.map["ra_gal"], 0.0), 0.0) if self.map["ra_gal"] else 0.0
        return ra_man, ra_gal


def _twin_outlets_from_step(step_res: Dict[int, List[np.ndarray]]) -> Tuple[float, float, float, float]:
    """Return twin Extract/Raffinate for comp0=Man, comp1=Gal.
    step_res[zone][-1][comp][-1] — last object’s outlet, last grid cell.
    """
    def safe(zone: int, comp: int) -> float:
        try:
            return float(step_res[zone][-1][comp][-1])
        except Exception:
            return 0.0

    ex_man = safe(1, 0); ex_gal = safe(1, 1)
    ra_man = safe(3, 0); ra_gal = safe(3, 1)
    return ex_man, ex_gal, ra_man, ra_gal


class SMBTwinLogger:
    """Periodic CSV/console logger for the SMB twin vs plant."""

    def __init__(
        self,
        opc_client: SMB_OPCUAClient,
        engine: SMBTwinEngine,
        csv_path: str,
        log_period_s: float = 5.0,
        print_period_s: float = 20.0,
    ) -> None:
        self.cli = opc_client
        self.engine = engine
        self.csv_path = csv_path
        self.log_period_s = float(log_period_s)
        self.print_period_s = float(print_period_s)

        # twin caches (filled by engine thread)
        self._sim_t = 0.0
        self._ex_man = 0.0; self._ex_gal = 0.0
        self._ra_man = 0.0; self._ra_gal = 0.0

        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

        now = time.time()
        self._next_log_t = now + self.log_period_s
        self._next_print_t = now + self.print_period_s

        # CSV header
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            "timestamp_iso", "sim_time_s",
            f"plant_extract_{COMP0_NAME}", f"plant_extract_{COMP1_NAME}",
            f"plant_raffinate_{COMP0_NAME}", f"plant_raffinate_{COMP1_NAME}",
            f"twin_extract_{COMP0_NAME}",  f"twin_extract_{COMP1_NAME}",
            f"twin_raffinate_{COMP0_NAME}",f"twin_raffinate_{COMP1_NAME}",
            f"diff_extract_{COMP0_NAME}",  f"diff_extract_{COMP1_NAME}",
            f"diff_raffinate_{COMP0_NAME}",f"diff_raffinate_{COMP1_NAME}",
            "speed_factor", "feed", "q1", "q2", "q4",
            "switch_interval_s", "twin_countdown_s",
        ])
        self._csv_file.flush()

        # subscribe to engine
        self.engine.subscribe(self._on_engine_update)

        # resolve plant tag names once
        first = self._read_snapshot()
        self.resolver = OutletTagResolver()
        self.resolver.resolve(first)
        sel = self.resolver.selected()
        print("[smb_logging] Using plant outlet tags:",
              f"Extract {COMP0_NAME}={sel['ex_man'] or 'NOT FOUND'}, {COMP1_NAME}={sel['ex_gal'] or 'NOT FOUND'}; "
              f"Raffinate {COMP0_NAME}={sel['ra_man'] or 'NOT FOUND'}, {COMP1_NAME}={sel['ra_gal'] or 'NOT FOUND'}")
        if any(v is None for v in sel.values()):
            print("[smb_logging] WARNING: Some outlet tags were not found. "
                  "Add your exact names to OutletTagResolver.PRIORITY.", file=sys.stderr)

    # Engine thread callback — update cache only
    def _on_engine_update(self, step_res: Dict[int, List[np.ndarray]], sim_t: float) -> None:
        ex_man, ex_gal, ra_man, ra_gal = _twin_outlets_from_step(step_res)
        self._ex_man, self._ex_gal = ex_man, ex_gal
        self._ra_man, self._ra_gal = ra_man, ra_gal
        self._sim_t = float(sim_t)

    def _read_snapshot(self) -> Dict[str, float]:
        try:
            return self.cli.read_snapshot()
        except Exception as ex:
            print(f"[smb_logging] Warning: read_snapshot failed: {ex}", file=sys.stderr)
            return {}

    def _read_engine_countdown(self) -> float:
        try:
            smb = getattr(self.engine, "_smb", None)
            return float(getattr(smb, "countdown")) if smb is not None else 0.0
        except Exception:
            return 0.0

    def _log_once(self) -> None:
        snap = self._read_snapshot()

        # Plant (using resolved names)
        p_ex_man, p_ex_gal = self.resolver.read_extract(snap)
        p_ra_man, p_ra_gal = self.resolver.read_raffinate(snap)

        # Twin (from cache)
        t_ex_man, t_ex_gal = self._ex_man, self._ex_gal
        t_ra_man, t_ra_gal = self._ra_man, self._ra_gal

        # Diffs (twin − plant)
        d_ex_man = t_ex_man - p_ex_man
        d_ex_gal = t_ex_gal - p_ex_gal
        d_ra_man = t_ra_man - p_ra_man
        d_ra_gal = t_ra_gal - p_ra_gal

        # Operating points — lean schema only
        speed = _pick(snap, "SpeedFactor", default=1.0)
        F  = _pick(snap, "ActiveFeed", "Feed", default=0.0)
        Q1 = _pick(snap, "ActiveQ1", "Q1", default=0.0)
        Q2 = _pick(snap, "ActiveQ2", "Q2", default=0.0)
        Q4 = _pick(snap, "ActiveQ4", "Q4", default=0.0)
        SI = _pick(snap, "ActiveSwitchInterval", "SwitchInterval", default=0.0)
        cd = self._read_engine_countdown()

        ts = datetime.now().isoformat(timespec="seconds")
        self._csv.writerow([
            ts, f"{self._sim_t:.6f}",
            f"{p_ex_man:.6g}", f"{p_ex_gal:.6g}", f"{p_ra_man:.6g}", f"{p_ra_gal:.6g}",
            f"{t_ex_man:.6g}", f"{t_ex_gal:.6g}", f"{t_ra_man:.6g}", f"{t_ra_gal:.6g}",
            f"{d_ex_man:.6g}", f"{d_ex_gal:.6g}", f"{d_ra_man:.6g}", f"{d_ra_gal:.6g}",
            f"{speed:.6g}", f"{F:.6g}", f"{Q1:.6g}", f"{Q2:.6g}", f"{Q4:.6g}",
            f"{SI:.6g}", f"{cd:.6g}",
        ])
        self._csv_file.flush()

        # Keep last diffs for printing so CLI and CSV agree
        self._last_print = (d_ex_man, d_ex_gal, d_ra_man, d_ra_gal)


    def _print_once(self) -> None:
        if not hasattr(self, "_last_print"):
            self._log_once()
        d_ex_man, d_ex_gal, d_ra_man, d_ra_gal = getattr(self, "_last_print", (0.0, 0.0, 0.0, 0.0))
        print(
            f"[smb_logging][Δ outlets @ {datetime.now().strftime('%H:%M:%S')}]  "
            f"Extract {COMP0_NAME}={d_ex_man:+.4g}, {COMP1_NAME}={d_ex_gal:+.4g} | "
            f"Raffinate {COMP0_NAME}={d_ra_man:+.4g}, {COMP1_NAME}={d_ra_gal:+.4g}"
        )

    def start(self) -> None:
        if hasattr(self, "_thr") and self._thr and self._thr.is_alive():
            return
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, name="SMBLogger", daemon=True)
        self._thr.start()
        print(f"[smb_logging] Logging to: {self.csv_path}")

    def stop(self) -> None:
        if hasattr(self, "_stop"):
            self._stop.set()
        if hasattr(self, "_thr") and self._thr and self._thr.is_alive():
            self._thr.join(timeout=2.0)
        try:
            self._csv_file.close()
        except Exception:
            pass
        print("[smb_logging] stopped.")

    def _loop(self) -> None:
        next_log = time.time() + self.log_period_s
        next_print = time.time() + self.print_period_s
        while not self._stop.is_set():
            now = time.time()
            if now >= next_log:
                next_log += self.log_period_s
                self._log_once()
            if now >= next_print:
                next_print += self.print_period_s
                self._print_once()
            time.sleep(0.1)


def build_filename(prefix: str = "smb_log") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
