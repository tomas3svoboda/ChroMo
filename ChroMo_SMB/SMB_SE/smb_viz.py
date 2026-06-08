"""
Thread-safe visualization for the SMB digital twin.

Design goals (for chemical engineers)
-------------------------------------
• Show spatial concentration profiles across the 4 columns (Man, Gal).
• Show Extract and Raffinate outlet histories over the last few switch
  intervals (in seconds), updating smoothly without blocking the engine.
• Keep plotting in the main/UI thread; the engine thread only pushes data.

Notes
-----
• The history window is defined in seconds as: history_si × (si_steps × dt_s).
• No external deps beyond NumPy and Matplotlib.
"""

from __future__ import annotations
from typing import Dict, List, Deque, Tuple
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

# Compact, readable defaults
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.0,
})


class SMBViz:
    """Main visualization object. Engine calls `ingest`; UI timer draws periodically.

    Parameters
    ----------
    Nx : int
        Spatial grid per column (for plotting profiles: 4×Nx points).
    L_mm : float
        Column length [mm]. The combined profile spans 4×L.
    si_steps : float
        Switch interval in **time steps** (as configured in engine/plant).
    dt_s : float
        Simulation step [s]. Used to convert si_steps → seconds for history window.
    history_si : int
        Number of switching intervals to display in the outlet histories.
    redraw_ms : int
        UI refresh period [ms]. Larger → lighter CPU but less frequent updates.
    max_ts_points : int
        Max points kept in each time-series (decimated for performance).
    """

    def __init__(
        self,
        *,
        Nx: int = 100,
        L_mm: float = 320.0,
        si_steps: float = 1894.0,
        dt_s: float = 0.05,
        history_si: int = 10,
        redraw_ms: int = 750,
        max_ts_points: int = 800,
    ) -> None:
        # ------------ config & time conversions (set BEFORE using in titles) ------------
        self.Nx = int(Nx)
        self.L_mm = float(L_mm)
        self._si_steps = float(si_steps)
        self._dt_s = float(dt_s)
        self._hist_si = float(history_si)
        # seconds per single SI, and total history window in seconds
        self._si_sec = self._si_steps * self._dt_s
        self._win_sec = self._hist_si * self._si_sec

        self._lock = threading.Lock()
        self._sim_t = 0.0
        self._max_ts_points = int(max_ts_points)

        # Buffers (engine thread writes under lock)
        self._p0 = np.zeros(4 * self.Nx)  # Man profile
        self._p1 = np.zeros(4 * self.Nx)  # Gal profile
        self._hex0: Deque[Tuple[float, float]] = deque(maxlen=30000)
        self._hex1: Deque[Tuple[float, float]] = deque(maxlen=30000)
        self._hra0: Deque[Tuple[float, float]] = deque(maxlen=30000)
        self._hra1: Deque[Tuple[float, float]] = deque(maxlen=30000)

        # Figure & axes
        self.fig = plt.figure(figsize=(9.6, 5.4), dpi=110)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
        self.ax_prof = self.fig.add_subplot(gs[:, 0])
        self.ax_ex = self.fig.add_subplot(gs[0, 1])
        self.ax_ra = self.fig.add_subplot(gs[1, 1])
        try:
            self.fig.canvas.manager.set_window_title("SMB Digital Twin – Visualization")
        except Exception:
            pass

        # Spatial profile artists
        x_axis = np.linspace(0.0, 4.0 * self.L_mm, 4 * self.Nx)
        (self.line_p0,) = self.ax_prof.plot(x_axis, self._p0, label="Man")
        (self.line_p1,) = self.ax_prof.plot(x_axis, self._p1, label="Gal")
        for b in (1, 2, 3):
            self.ax_prof.axvline(b * self.L_mm, linestyle="--", linewidth=0.8)
        self.ax_prof.set_xlabel("Continuous Column Length [mm]")
        self.ax_prof.set_ylabel("Concentration [g/L]")
        self.ax_prof.legend(loc="upper right", framealpha=0.75)
        self.ax_prof.margins(y=0.1)

        # Outlet history artists (Extract)
        (self.line_ex0,) = self.ax_ex.plot([], [], label="Man")
        (self.line_ex1,) = self.ax_ex.plot([], [], label="Gal")
        self.ax_ex.set_title(f"Extract outlet (last ~{int(self._hist_si)} SI)")
        self.ax_ex.set_xlabel("Time [s]")
        self.ax_ex.set_ylabel("c [g/L]")
        self.ax_ex.legend(loc="upper right", framealpha=0.75)
        self.ax_ex.margins(y=0.1)

        # Outlet history artists (Raffinate)
        (self.line_ra0,) = self.ax_ra.plot([], [], label="Man")
        (self.line_ra1,) = self.ax_ra.plot([], [], label="Gal")
        self.ax_ra.set_title(f"Raffinate outlet (last ~{int(self._hist_si)} SI)")
        self.ax_ra.set_xlabel("Time [s]")
        self.ax_ra.set_ylabel("c [g/L]")
        self.ax_ra.legend(loc="upper right", framealpha=0.75)
        self.ax_ra.margins(y=0.1)

        self.fig.tight_layout()

        # Main-thread timer for redraw
        self._timer = self.fig.canvas.new_timer(interval=int(redraw_ms))
        self._timer.add_callback(self._on_timer)
        self._timer.start()

    # ---------- ENGINE THREAD ONLY ----------
    def ingest(self, step_res: Dict[int, List[np.ndarray]], sim_t: float) -> None:
        """Push one engine update. No GUI operations here.

        step_res: dict[zone] -> [col_array, tube_array] with shape (2, Nx)
        """
        Nx = self.Nx

        def concat_profile(comp_idx: int) -> np.ndarray:
            profs = []
            for zone in range(1, 5):
                objs = step_res.get(zone, [])
                # Prefer the *last* object in the zone chain (Tube), fall back to Column
                obj_idx = 1 if len(objs) > 1 else 0
                try:
                    arr = np.asarray(objs[obj_idx])
                    if arr.ndim == 2:
                        profs.append(arr[comp_idx, :])
                    elif arr.ndim == 1:
                        profs.append(arr)
                    else:
                        profs.append(np.zeros(Nx))
                except Exception:
                    profs.append(np.zeros(Nx))
            return np.hstack(profs) if profs else np.zeros(4 * Nx)

        # Outlets from zone 1 (Extract) and zone 3 (Raffinate), components 0/1
        def safe_outlet(zone: int, comp: int) -> float:
            try:
                return float(step_res[zone][-1][comp][-1])
            except Exception:
                return 0.0

        ex0 = safe_outlet(1, 0)
        ex1 = safe_outlet(1, 1)
        ra0 = safe_outlet(3, 0)
        ra1 = safe_outlet(3, 1)

        p0 = concat_profile(0)
        p1 = concat_profile(1)

        with self._lock:
            self._p0 = p0
            self._p1 = p1
            self._sim_t = float(sim_t)
            self._hex0.append((sim_t, ex0))
            self._hex1.append((sim_t, ex1))
            self._hra0.append((sim_t, ra0))
            self._hra1.append((sim_t, ra1))

    # ---------- UI THREAD ONLY ----------
    def _on_timer(self) -> None:
        with self._lock:
            sim_t = self._sim_t
            p0, p1 = self._p0, self._p1
            hex0 = list(self._hex0)
            hex1 = list(self._hex1)
            hra0 = list(self._hra0)
            hra1 = list(self._hra1)

        # Spatial profile update + autoscale Y
        self.line_p0.set_ydata(p0)
        self.line_p1.set_ydata(p1)
        self.ax_prof.relim()
        self.ax_prof.autoscale_view(scalex=False, scaley=True)

        # Histories (last ~history_si SI) with decimation
        def window_and_decimate(buf):
            if not buf:
                return [], []
            t, v = zip(*buf)
            tmin = max(0.0, sim_t - self._win_sec)
            tw = np.asarray(t)
            vw = np.asarray(v)
            sel = tw >= tmin
            tw = tw[sel]
            vw = vw[sel]
            n = len(tw)
            if n > self._max_ts_points:
                idx = np.linspace(0, n - 1, self._max_ts_points).astype(int)
                tw = tw[idx]
                vw = vw[idx]
            return tw, vw

        t_ex0, v_ex0 = window_and_decimate(hex0)
        t_ex1, v_ex1 = window_and_decimate(hex1)
        t_ra0, v_ra0 = window_and_decimate(hra0)
        t_ra1, v_ra1 = window_and_decimate(hra1)

        self.line_ex0.set_data(t_ex0, v_ex0)
        self.line_ex1.set_data(t_ex1, v_ex1)
        self.ax_ex.set_xlim(max(0.0, sim_t - self._win_sec), max(sim_t, 1.0))
        self.ax_ex.relim()
        self.ax_ex.autoscale_view(scalex=False, scaley=True)

        self.line_ra0.set_data(t_ra0, v_ra0)
        self.line_ra1.set_data(t_ra1, v_ra1)
        self.ax_ra.set_xlim(max(0.0, sim_t - self._win_sec), max(sim_t, 1.0))
        self.ax_ra.relim()
        self.ax_ra.autoscale_view(scalex=False, scaley=True)

        self.fig.canvas.draw_idle()


def attach_viz(
    eng,
    *,
    Nx: int = 100,
    L_mm: float = 320.0,
    si_steps: float = 1894.0,
    dt_s: float = 0.05,
    history_si: float = 10.0,
    redraw_ms: int = 750,
    max_ts_points: int = 800,
):
    viz = SMBViz(
        Nx=Nx,
        L_mm=L_mm,
        si_steps=si_steps,
        dt_s=dt_s,
        history_si=int(history_si),
        redraw_ms=redraw_ms,
        max_ts_points=max_ts_points,
    )
    eng.subscribe(viz.ingest)
    return viz.fig
