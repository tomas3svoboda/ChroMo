import os
import threading
import time
import tkinter as tk
from tkinter import ttk
from ChroMo_SMB.SMB_PlantSim.opcua_server import SMBOPCUAServer
#from SMB.SMBStation import SMBStation
#from SMB.LinColumn_working import LinColumn
#from SMB.Tube import Tube
from SMB.SMBStation import SMBStation
from SMB.NonlinColumn import NonLinColumn
from SMB.Tube import Tube

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# --- Matplotlib sane defaults for Tk GUI ---
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.2  # thinner than default (2.0)
})

import numpy as np
from matplotlib.gridspec import GridSpec
from collections import deque

# Bilance function: computes all flows given feed, eluent, extract, recycle (Zone IV)
# Bilance function: computes all flows given feed F and zone flows Q1, Q2, Q4
def smb_all_flows_F_Q(F, Q1, Q2, Q4):
    V_I   = Q1
    V_II  = Q2
    V_III = Q2 + F           # QIII
    V_IV  = Q4
    D = Q1 - Q4              # eluent
    E = Q1 - Q2              # extract
    R = V_III - Q4           # raffinate
    # sanity: F + D == E + R
    assert abs((F + D) - (E + R)) < 1e-9, "Balance error!"
    return {
        "feed": F, "q1": Q1, "q2": Q2, "q4": Q4,
        "eluent": D, "extract": E, "raffinate": R,
        "zone1": V_I, "zone2": V_II, "zone3": V_III, "zone4": V_IV
    }

class SimulationThread(threading.Thread):
    def __init__(self, smb, params, components, update_callback,
                 speed_factor_var, sim_start_wall, ui_post=None, throttle_sec=0.10):
        super().__init__(daemon=True)
        self.smb = smb
        self.params = params
        self.components = components
        self.update_callback = update_callback
        self._stop_event = threading.Event()
        self.speed_factor_var = speed_factor_var
        self.sim_start_wall = sim_start_wall
        # NEW: how to post work to the UI thread (root.after)
        self.ui_post = ui_post
        # NEW: minimum wall-time between UI redraws
        self.throttle_sec = throttle_sec

    def get_steps_per_update(self, speed_factor):
        # keep your mapping, or bump a little if you like
        if speed_factor <= 2:
            return 20
        elif speed_factor <= 5:
            return 40
        elif speed_factor <= 10:
            return 160
        elif speed_factor <= 20:
            return 200
        elif speed_factor <= 40:
            return 300
        else:
            return 300

    def run(self):
        last_draw = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                speed = float(self.speed_factor_var.get())
                self.steps_per_update = self.get_steps_per_update(speed)

                # target sim time from wall clock
                wall_elapsed = time.time() - self.sim_start_wall[0]
                target_sim_time = wall_elapsed * speed

                sim_time_now = self.smb.timer
                sim_time_needed = target_sim_time - sim_time_now

                if sim_time_needed <= 0:
                    time.sleep(0.02)
                    continue

                dt = self.params['dt']
                # use regular division; int() will floor
                nsteps = int(min(sim_time_needed / dt, self.steps_per_update))
                if nsteps <= 0:
                    # we’re already at (or within one dt of) the target → let wall time advance
                    time.sleep(0.02)
                    continue

                res = self.smb.step(nsteps)

                # Only ask the UI to redraw at most every throttle_sec
                now = time.perf_counter()
                if (now - last_draw) >= self.throttle_sec:
                    if self.ui_post is not None:
                        # Post to Tk main thread
                        self.ui_post(self.update_callback, res)
                    else:
                        # Fallback (not recommended for Tk): direct call
                        self.update_callback(res)
                    last_draw = now

        except Exception as e:
            print("SimulationThread Exception:", e)

    def stop(self):
        self._stop_event.set()


# --- FILE LOGGING HELPERS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # always the script folder
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

log_file_handle = [None]
last_file_log_simtime = [None]

def open_new_logfile(prefix="smb_log"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}.txt"
    fpath = os.path.join(log_dir, fname)
    # line buffered so each '\n' flushes; still call flush on headers
    f = open(fpath, "w", encoding="utf-8", buffering=1)
    return f, fpath


def close_logfile():
    try:
        if log_file_handle[0] is not None:
            log_file_handle[0].flush()
            log_file_handle[0].close()
            log_file_handle[0] = None
    except Exception:
        pass


# --- OUTLETS HISTORY BUFFER (last 5 switching periods) ---
# Use deques for efficient pruning. We store (t, val) pairs in sim-time seconds.
history = {
    "extract": {"Man": deque(), "Gal": deque()},
    "raffinate": {"Man": deque(), "Gal": deque()},
}


# Helper to trim history to a given time window
def trim_history(window_s, t_now):
    for side in ("extract", "raffinate"):
        for comp in ("Man", "Gal"):
            dq = history[side][comp]
            while dq and dq[0][0] < t_now - window_s:
                dq.popleft()


def init_smb(params, components, zone_flows):
    smb = SMBStation()
    for zone in range(4):
        smb.addColZone(zone + 1,
                       NonLinColumn(params['col_length'], params['col_diameter'], params['porosity']),
                       Tube(params['dead_volume']))
    for i, fr in enumerate(zone_flows, 1):
        smb.setFlowRateZone(i, fr)

    smb.setSwitchInterval(params['switch_interval'])
    smb.setdt(params['dt'])
    smb.setNx(params['Nx'])

    for comp in components:
        smb.createNonLinComponentAB(
            comp["name"],
            comp["feed_concentration"],
            comp.get("henry_constant", -1),
            comp["delta"], comp["Di"],
            comp["K_L"], comp["q_m"]
        )

    # >>> NEW: select isotherm mode <<<
    smb.set_isotherm_mode('competitive' if params.get('iso_mode', 'noncomp') == 'competitive' else 'noncomp')

    smb.initCols()
    return smb



def main():

    # --- Default system parameters ---
    params = {
        'dt': 1,
        'Nx': 30,
        'switch_interval': 1437,
        'col_length': 310,
        'col_diameter': 10,
        'porosity': 0.376,
        'dead_volume': 0.2,
        'iso_mode': 'competitive'   # or 'noncomp'
    }
    '''components = [
        {"name": "Man", "feed_concentration": 7.27, "henry_constant": -1,
         "delta": 15.41, "Di": 0.0007, "K_L": 0.74, "q_m": 13.44},
        {"name": "Gal", "feed_concentration": 3.42, "henry_constant": -1,
         "delta": 15.41, "Di": 0.0007, "K_L": 0.27, "q_m": 18.13},
    ]'''
    components = [
        {"name": "Man", "feed_concentration": 7.27, "henry_constant": -1,
         "delta": 15.415, "Di": 0.0007, "K_L": 0.649, "q_m": 13.444},
        {"name": "Gal", "feed_concentration": 3.42, "henry_constant": -1,
         "delta": 15.415, "Di": 0.0007, "K_L": 0.235, "q_m": 18.134},
    ]

    # Default operating values
    flows = {
        "feed": 21.0,
        "q1": 37.2,
        "q2": 16.2,
        "q4": 16.2
    }

    # OPC UA inplementation
    shared_state = {
        'mode': 'manual',

        # ACTIVE mirrors (RO on server)
        'active_feed': flows['feed'],
        'active_q1': flows['q1'],
        'active_q2': flows['q2'],
        'active_q4': flows['q4'],
        'active_switch_interval': params['switch_interval'],

        # NEXT setpoints (RW on server)
        'feed': flows["feed"],
        'q1': flows["q1"],
        'q2': flows["q2"],
        'q4': flows["q4"],
        'switch_interval': params['switch_interval'],

        # Measurements and telemetry (RO on server)
        'extract_concentration_man': 0.0,
        'extract_concentration_gal': 0.0,
        'raffinate_concentration_man': 0.0,
        'raffinate_concentration_gal': 0.0,
        'simulation_time': 0.0,
        'is_running': False,
        'switch_countdown': 0.0,
        'elapsed_wall_time': 0.0,
        'speed_factor': 1.0,

        # Switch conting
        'switch_index': 0,
        'last_switch_simtime': 0.0,
    }

    wall_time_start = [None]   # Use list for mutability
    elapsed_wall_accum = [0.0]  # Accumulates wall time over multiple runs
    sim_start_wall = [None]     # Used foe elapsed wall time and simulation time synchronization logic

    last_switch_countdown = [None]  # tracks previous countdown to detect switch edge in auto mode
    switch_edge_armed = [True]  # allows exactly one increment per boundary

    computed = smb_all_flows_F_Q(flows["feed"], flows["q1"], flows["q2"], flows["q4"])
    zone_flows = [computed['zone1'], computed['zone2'], computed['zone3'], computed['zone4']]

    smb = init_smb(params, components, zone_flows)
    sim_state = {"running": False}
    sim_thread = [None]

    # Start OPC UA server
    opcua_server = SMBOPCUAServer(shared_state)
    opcua_server.start()

    # --- Windows HiDPI fix (run before tk.Tk()) ---
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
    except Exception:
        pass

    # --- Tkinter UI ---
    root = tk.Tk()
    # Normalize Tk's internal font scaling (1.0 == 96 DPI baseline)
    try:
        root.tk.call('tk', 'scaling', 1.0)
    except Exception:
        pass
    root.title("SMB Interactive Simulator")

    ctrl = ttk.Frame(root, padding=10)
    ctrl.pack(side='left', fill='y')

    # --- Inputs ---
    # Add radio buttons somewhere in the control frame:
    tk.Label(ctrl, text="Feed (F) [mL/h]").pack(anchor='w')
    feed_var = tk.DoubleVar(value=flows["feed"])
    feed_scale = tk.Scale(ctrl, from_=1, to=500, variable=feed_var, orient='horizontal', resolution=1)
    feed_scale.pack(fill='x')

    tk.Label(ctrl, text="Zone I (Q1) [mL/h]").pack(anchor='w')
    q1_var = tk.DoubleVar(value=flows["q1"])
    q1_scale = tk.Scale(ctrl, from_=1, to=500, variable=q1_var, orient='horizontal', resolution=1)
    q1_scale.pack(fill='x')

    tk.Label(ctrl, text="Zone II (Q2) [mL/h]").pack(anchor='w')
    q2_var = tk.DoubleVar(value=flows["q2"])
    q2_scale = tk.Scale(ctrl, from_=1, to=500, variable=q2_var, orient='horizontal', resolution=1)
    q2_scale.pack(fill='x')

    tk.Label(ctrl, text="Zone IV (Q4) [mL/h]").pack(anchor='w')
    q4_var = tk.DoubleVar(value=flows["q4"])
    q4_scale = tk.Scale(ctrl, from_=1, to=500, variable=q4_var, orient='horizontal', resolution=1)
    q4_scale.pack(fill='x')

    tk.Label(ctrl, text="Switch interval [s]").pack(anchor='w')
    switch_var = tk.DoubleVar(value=params['switch_interval'])
    switch_scale = tk.Scale(ctrl, from_=100, to=10800, variable=switch_var, orient='horizontal', resolution=1)
    switch_scale.pack(fill='x')

    # --- Matplotlib figure with outlets history ---
    fig = plt.figure(figsize=(9, 4.8), dpi=110)
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1], figure=fig)
    ax    = fig.add_subplot(gs[:, 0])   # main spatial profile
    ax_ex = fig.add_subplot(gs[0, 1])   # extract history
    ax_ra = fig.add_subplot(gs[1, 1])   # raffinate history
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side='right', fill='both', expand=True)
    x_axis = np.linspace(0, 4*params['col_length'], 4*params['Nx'])

    # --- Dependent flows text box ---
    apply_btn = ttk.Button(ctrl, text="Apply", command=lambda: apply_params())
    apply_btn.pack(pady=(10,0), fill='x')

    flows_display = tk.Text(ctrl, width=34, height=6, font=("Consolas", 9))
    flows_display.pack(fill='x', pady=(8,0))
    flows_display.insert('end', "Calculated dependent flows will appear here.\n")

    def update_dependent_flows():
        try:
            F  = float(feed_var.get())
            Q1 = float(q1_var.get())
            Q2 = float(q2_var.get())
            Q4 = float(q4_var.get())
            c = smb_all_flows_F_Q(F, Q1, Q2, Q4)

            # UI only – do not mirror to shared_state (these are not exposed on the server anymore)
            flows_display.delete('1.0', 'end')
            flows_display.insert('end',
                f" Eluent (D): {c['eluent']:.3f} mL/h\n"
                f" Extract (E): {c['extract']:.3f} mL/h\n"
                f" Raffinate :  {c['raffinate']:.3f} mL/h\n"
                f" Zone I  :    {c['zone1']:.3f} mL/h (Q1)\n"
                f" Zone II :    {c['zone2']:.3f} mL/h (Q2)\n"
                f" Zone III:    {c['zone3']:.3f} mL/h (Q3)\n"
                f" Zone IV :    {c['zone4']:.3f} mL/h (Q4)\n"
            )
        except Exception as e:
            flows_display.delete('1.0', 'end')
            flows_display.insert('end', f"Invalid flows: {e}")


    for var in [feed_var, q1_var, q2_var, q4_var]:
        var.trace_add('write', lambda *args: update_dependent_flows())


    update_dependent_flows()

    log = tk.Text(ctrl, width=34, height=10)
    log.pack(fill='x', pady=(10,0))

    # --- Labels for time and mode ---
    mode_var = tk.StringVar(value="manual")
    ttk.Radiobutton(ctrl, text="Manual", variable=mode_var, value="manual", command=lambda: on_mode_change()).pack(anchor='w')
    ttk.Radiobutton(ctrl, text="Automatic", variable=mode_var, value="automatic", command=lambda: on_mode_change()).pack(anchor='w')

    tk.Label(ctrl, text="Simulation speed").pack(anchor='w', pady=(5,0))
    speed_factor = tk.DoubleVar(value=1.0)
    speed_combo = ttk.Combobox(ctrl, textvariable=speed_factor, values=[1,2,5,10,20,40], state="readonly")
    speed_combo.pack(fill='x', pady=2)

    btn_frame = ttk.Frame(ctrl)
    btn_frame.pack(pady=(20,0), fill='x')
    start_btn = ttk.Button(btn_frame, text="Start", command=lambda: start_simulation())
    stop_btn = ttk.Button(btn_frame, text="Stop",  command=lambda: stop_simulation(), state='disabled')
    start_btn.pack(side='left', expand=True, fill='x')
    stop_btn.pack(side='left', expand=True, fill='x')

    reset_btn = ttk.Button(ctrl, text="Reset Simulation", command=lambda: reset_simulation())
    reset_btn.pack(pady=(10,0), fill='x')

    time_label = ttk.Label(ctrl, text="Simulation time: 0.0 s")
    time_label.pack(pady=(10,0), fill='x')
    wall_label = ttk.Label(ctrl, text="Elapsed wall time: 0.0 s")
    wall_label.pack(pady=(0,0), fill='x')
    switch_countdown_label = ttk.Label(ctrl, text="Time to next switch: 0.0 s")
    switch_countdown_label.pack(pady=(0,0), fill='x')

    # --- Plotting update function ---
    def show_step(res):
        # 1) Main spatial profile
        ax.clear()
        for idx, comp in enumerate(components):
            profiles = []
            for zone in range(4):
                objs = res[zone+1]                # list of objects in the zone
                obj_idx = 1 if len(objs) > 1 else 0  # prefer the 2nd object if present, else 1st
                arr = np.asarray(objs[obj_idx])   # shape expected: (n_components, Nx) OR (Nx,)
                if arr.ndim == 2:
                    profiles.append(arr[idx, :])  # take this component
                elif arr.ndim == 1:
                    profiles.append(arr)
            full_profile = np.hstack(profiles)
            ax.plot(x_axis, full_profile, label=comp["name"])

        for boundary in range(1, 4):
            ax.axvline(boundary * params['col_length'], color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel("Continuous Column Length [mm]")
        ax.set_ylabel("Concentration")
        ax.legend()
        ax.grid(False)

        # 2) Read outlets and update shared_state + history buffers
        try:
            # Extract outlet: zone 1, last column, Man (0) and Gal (1)
            extract_man = float(res[1][-1][0][-1])
            extract_gal = float(res[1][-1][1][-1])
            # Raffinate outlet: zone 3, last column, Man (0) and Gal (1)
            raffinate_man = float(res[3][-1][0][-1])
            raffinate_gal = float(res[3][-1][1][-1])
        except Exception:
            extract_man = extract_gal = raffinate_man = raffinate_gal = 0.0

        shared_state['extract_concentration_man'] = extract_man
        shared_state['extract_concentration_gal'] = extract_gal
        shared_state['raffinate_concentration_man'] = raffinate_man
        shared_state['raffinate_concentration_gal'] = raffinate_gal

        # Simulation time + labels
        time_label.config(text=f"Simulation time: {smb.timer:.1f} s")
        shared_state['simulation_time'] = smb.timer

        if wall_time_start[0] is not None:
            wall_label.config(text=f"Elapsed wall time: {get_elapsed_wall_time():.1f} s")
        else:
            wall_label.config(text="Elapsed wall time: 0.0 s")

        switch_countdown_label.config(text=f"Time to next switch: {smb.countdown:.1f} s")
        shared_state['switch_countdown'] = float(smb.countdown)
        shared_state['elapsed_wall_time'] = float(get_elapsed_wall_time())

        # 3) Update history (keep only last 5 switch periods based on ACTIVE switch interval)
        window_s = 5.0 * float(shared_state.get('active_switch_interval', params['switch_interval']))
        t_now = float(smb.timer)
        history['extract']['Man'].append((t_now, extract_man))
        history['extract']['Gal'].append((t_now, extract_gal))
        history['raffinate']['Man'].append((t_now, raffinate_man))
        history['raffinate']['Gal'].append((t_now, raffinate_gal))
        trim_history(window_s, t_now)

        # 4) Small subplots with time evolution
        def plot_history(ax_sub, dq_man, dq_gal, title):
            ax_sub.clear()
            if dq_man:
                t_m, v_m = zip(*dq_man)
                ax_sub.plot(t_m, v_m, label='Man')
            if dq_gal:
                t_g, v_g = zip(*dq_gal)
                ax_sub.plot(t_g, v_g, label='Gal')
            ax_sub.set_xlim(max(0.0, t_now - window_s), max(t_now, 1.0))
            ax_sub.set_xlabel('t [s]')
            ax_sub.set_ylabel('c [g/L]')
            ax_sub.set_title(title)
            ax_sub.tick_params(labelsize=8)
            ax_sub.grid(True, alpha=0.3)
            ax_sub.legend(loc='upper right', fontsize=8)

        plot_history(ax_ex, history['extract']['Man'], history['extract']['Gal'], 'Extract outlet (last 5 SI)')
        plot_history(ax_ra, history['raffinate']['Man'], history['raffinate']['Gal'], 'Raffinate outlet (last 5 SI)')

        # 5) Auto-apply OPC UA setpoints exactly at switch (rising countdown edge)
        # --- Debounced boundary detection ---
        prev = last_switch_countdown[0]
        curr = float(smb.countdown)

        # classify movement with a tiny hysteresis
        EPS = max(1e-6, 0.1*params['dt'])
        increasing = (prev is not None and curr > prev + EPS)
        decreasing = (prev is not None and curr < prev - EPS)

        # store for next call
        last_switch_countdown[0] = curr

        # fire exactly once per real boundary
        switched = increasing and switch_edge_armed[0]

        if switched:
            if shared_state.get('mode') == 'automatic':
                # >>> NEW: get the latest client writes atomically <<<
                try:
                    opcua_server.pull_next_setpoints_into_shared_state()
                except Exception:
                    pass

                # now read staged setpoints and apply at boundary
                F  = float(shared_state.get('feed'))
                Q1 = float(shared_state.get('q1'))
                Q2 = float(shared_state.get('q2'))
                Q4 = float(shared_state.get('q4'))
                si = float(shared_state.get('switch_interval'))

                c = smb_all_flows_F_Q(F, Q1, Q2, Q4)
                for zone, flow in enumerate([c['zone1'], c['zone2'], c['zone3'], c['zone4']], 1):
                    smb.setFlowRateZone(zone, flow)
                smb.apply_zone_flows_now()
                smb.setSwitchInterval(si)

                # mirror into ACTIVE tags
                shared_state['active_feed'] = F
                shared_state['active_q1']   = Q1
                shared_state['active_q2']   = Q2
                shared_state['active_q4']   = Q4
                shared_state['active_switch_interval'] = si

                now_str = time.strftime("%Y-%m-%d %H:%M:%S")
                log.insert('end', f"[AUTO] {now_str} Applied OPC UA setpoints at switch: "
                                  f"F={F:.2f} Q1={Q1:.2f} Q2={Q2:.2f} Q4={Q4:.2f}  SI={si:.1f}\n")
                log.see('end')
                if log_file_handle[0] is not None:
                    log_file_handle[0].write(f"[AUTO] {now_str} apply-at-switch F={F:.2f} Q1={Q1:.2f} Q2={Q2:.2f} Q4={Q4:.2f} SI={si:.1f}\n")

            else:
                # MANUAL mode: only apply deferred SI (flows already applied on button press)
                si_target = float(shared_state['switch_interval'])
                si_active = float(shared_state.get('active_switch_interval', getattr(smb, 'interval', si_target)))
                if abs(si_target - si_active) > 1e-9:
                    smb.setSwitchInterval(si_target)
                    shared_state['active_switch_interval'] = si_target
                    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
                    log.insert('end', f"[MANUAL] {now_str} Applied deferred switch interval at boundary: SI={si_target:.1f}\n")
                    log.see('end')
                    if log_file_handle[0] is not None:
                        log_file_handle[0].write(f"[MANUAL] {now_str} apply-SI-at-switch SI={si_target:.6g}\n")

            # publish the boundary identity exactly once
            shared_state['last_switch_simtime'] = float(smb.timer)
            shared_state['switch_index'] = int(shared_state.get('switch_index', 0)) + 1

            # disarm until we see the countdown decreasing again
            switch_edge_armed[0] = False

        # re-arm once countdown resumes decreasing (after the reset and any SI change)
        if decreasing:
            switch_edge_armed[0] = True

        # 6) File logging of PVs each update (throttle to ~10 Hz sim-time if very fast)
        try:
            if log_file_handle[0] is not None:
                # Throttle by sim-time step to avoid huge files when speed is high
                if (last_file_log_simtime[0] is None) or (t_now - last_file_log_simtime[0] >= 5.0):
                    mode = shared_state.get('mode', 'manual')
                    sf = shared_state.get('speed_factor', 1.0)
                    feed_man = components[0]['feed_concentration']
                    feed_gal = components[1]['feed_concentration']
                    line = (
                        f"t_sim={t_now:.3f}\t"
                        f"t_wall={get_elapsed_wall_time():.3f}\t"
                        f"t_to_switch={float(smb.countdown):.3f}\t"
                        f"c_ex_Man={extract_man:.6g}\t"
                        f"c_ex_Gal={extract_gal:.6g}\t"
                        f"c_ra_Man={raffinate_man:.6g}\t"
                        f"c_ra_Gal={raffinate_gal:.6g}\t"
                        f"c_feed_Man={components[0]['feed_concentration']:.6g}\t"
                        f"c_feed_Gal={components[1]['feed_concentration']:.6g}\t"
                        f"F={shared_state['active_feed']:.6g}\t"
                        f"Q1={shared_state['active_q1']:.6g}\t"
                        f"Q2={shared_state['active_q2']:.6g}\t"
                        f"Q4={shared_state['active_q4']:.6g}\t"
                        f"SI={shared_state['active_switch_interval']:.6g}\t"
                        f"mode={shared_state.get('mode','manual')}\tspd={shared_state.get('speed_factor',1.0)}\n"
                    )
                    log_file_handle[0].write(line)
                    # light flushing to keep data relatively safe
                    if int(t_now) % 5 == 0:
                        log_file_handle[0].flush()
                    last_file_log_simtime[0] = t_now
        except Exception as e:
            # file logging must never kill the UI
            pass

        canvas.draw_idle()

    def apply_params():
        try:
            F  = float(feed_var.get())
            Q1 = float(q1_var.get())
            Q2 = float(q2_var.get())
            Q4 = float(q4_var.get())
            switch_time = float(switch_var.get())

            c = smb_all_flows_F_Q(F, Q1, Q2, Q4)

            # live apply zone flows
            for zone, flow in enumerate([c['zone1'], c['zone2'], c['zone3'], c['zone4']], 1):
                smb.setFlowRateZone(zone, flow)

            # --- Decide whether to defer SI change to the next switch ---
            # Defer if: currently running OR sim_time > 0 (was running but may be stopped now).
            sim_time = float(shared_state.get('simulation_time', 0.0))
            is_running = bool(shared_state.get('is_running', False))
            defer_si = (is_running or sim_time > 0.0)

            if not defer_si:
                # Simulation hasn't started since last Reset → apply SI immediately (no counter reset beyond normal init)
                smb.setSwitchInterval(switch_time)
                # Also reflect that the active SI is now this value
                shared_state['active_switch_interval'] = switch_time
            # Always stage the target for reference / auto-apply-at-switch later
            params['switch_interval'] = switch_time

            # --- Update shared_state (NEXT buffer) ---
            shared_state['feed'] = F
            shared_state['q1']   = Q1
            shared_state['q2']   = Q2
            shared_state['q4']   = Q4
            shared_state['switch_interval'] = switch_time

            # --- Keep Active* in sync on manual apply ---
            if shared_state.get('mode') == 'manual':
                shared_state['active_feed'] = F
                shared_state['active_q1']   = Q1
                shared_state['active_q2']   = Q2
                shared_state['active_q4']   = Q4
                if not defer_si:
                    shared_state['active_switch_interval'] = switch_time
                # Reflect changes to OPC UA immediately (manual mode)
                try:
                    opcua_server.push_from_shared_state()
                except Exception as _e:
                    log.insert('end', f"[WARN] OPC UA push failed: {_e}\n")

            # --- UI log ---
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            log.insert('end',
                f"[{now_str}] Parameters applied (live update):\n"
                f" F={F:.2f}  Q1={Q1:.2f}  Q2={Q2:.2f}  Q4={Q4:.2f}\n"
                f" Switch interval: {switch_time:.1f} s\n"
            )
            if defer_si:
                log.insert('end', " (SI change deferred: will take effect at the next switch)\n")
            else:
                log.insert('end', " (SI applied immediately)\n")

            # --- file log ---
            if log_file_handle[0] is not None:
                log_file_handle[0].write(
                    f"[{now_str}] params_apply F={F:.2f} Q1={Q1:.2f} Q2={Q2:.2f} Q4={Q4:.2f} SI={switch_time:.6g}\n"
                )
        except Exception as e:
            log.insert('end', f"Error: {e}\n")
            log.see('end')

    def start_simulation():
        if sim_state["running"]:
            log.insert('end', "Simulation already running.\n")
            log.see('end')
            return
        log.insert('end', f"Simulation started!\n")
        log.see('end')
        sim_state["running"] = True
        shared_state['is_running'] = True

        now = time.time()
        wall_time_start[0] = now

        # Set sync reference to avoid lag on restart
        sim_start_wall[0] = now - smb.timer / speed_factor.get()

        # open logfile
        close_logfile()
        fh, path = open_new_logfile()
        log_file_handle[0] = fh
        last_file_log_simtime[0] = None
        log.insert('end', f"[LOG] Writing to: {path}\n")
        log.see('end')

        hdr = (
            "# SMB simulation log\n"
            f"# started_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# params dt={params['dt']} Nx={params['Nx']} L={params['col_length']}mm D={params['col_diameter']}mm eps={params['porosity']} Vdead={params['dead_volume']}\n"
            f"# components: {', '.join([c['name'] for c in components])}; feed_conc={[c['feed_concentration'] for c in components]}\n"
            f"# initial flows F={shared_state['active_feed']} Q1={shared_state['active_q1']} Q2={shared_state['active_q2']} Q4={shared_state['active_q4']} SI={shared_state['active_switch_interval']}\n"
            "# columns: t_sim\tt_wall\tt_to_switch\tc_ex_Man\tc_ex_Gal\tc_ra_Man\tc_ra_Gal\t"
            "c_feed_Man\tc_feed_Gal\tF\tQ1\tQ2\tQ4\tSI\tmode\tspd\n"
        )
        fh.write(hdr)
        fh.flush()

        sim_thread[0] = SimulationThread(
            smb, params, components, show_step, speed_factor, sim_start_wall,
            ui_post=lambda fn, *a, **k: root.after(0, lambda: fn(*a, **k)),
            throttle_sec=0.25
        )
        sim_thread[0].start()
        speed_combo.config(state='disabled')
        start_btn.config(state='disabled')
        stop_btn.config(state='normal')
        reset_btn.config(state='disabled')

    def stop_simulation():
        if sim_thread[0] is not None:
            sim_thread[0].stop()
            sim_thread[0] = None  # Optionally clear the thread reference
        sim_state["running"] = False
        shared_state['is_running'] = False
        log.insert('end', "Simulation stopped.\n")
        log.see('end')
        elapsed_wall_accum[0] += time.time() - wall_time_start[0]
        speed_combo.config(state='readonly')
        start_btn.config(state='normal')
        stop_btn.config(state='disabled')
        reset_btn.config(state='normal')
        # close logfile
        close_logfile()

    def reset_simulation():
        # Stop simulation if running
        if sim_state["running"] and sim_thread[0] is not None:
            sim_thread[0].stop()
            sim_thread[0] = None
            sim_state["running"] = False
            shared_state['is_running'] = False

        # --- Zero simulation time and countdown ---
        wall_time_start[0] = time.time()
        sim_start_wall[0] = wall_time_start[0]
        elapsed_wall_accum[0] = 0.0
        smb.timer = 0.0
        smb.countdown = getattr(smb, 'interval', 0.0)
        shared_state['active_switch_interval'] = float(getattr(smb, 'interval', params['switch_interval']))
        shared_state['simulation_time'] = 0.0
        shared_state['switch_countdown'] = float(smb.countdown)
        shared_state['elapsed_wall_time'] = 0.0
        last_switch_countdown[0] = float(smb.countdown)
        switch_edge_armed[0] = True

        # --- Zero all concentrations in columns and tubes ---
        for zone in smb.zones:
            for obj in smb.zones[zone]:
                for comp in getattr(obj, 'components', []):
                    if hasattr(comp, 'c'):
                        comp.c[:] = 0.0

        # --- Reset GUI time/countdown displays ---
        time_label.config(text="Simulation time: 0.0 s")
        wall_label.config(text="Elapsed wall time: 0.0 s")
        switch_countdown_label.config(text=f"Time to next switch: {smb.countdown:.1f} s")
        log.insert('end', "[INFO] Simulation fully reset to zero (in place).\n")
        log.see('end')

        # reset outlets history too
        for side in ("extract", "raffinate"):
            for comp in ("Man", "Gal"):
                history[side][comp].clear()

        res = get_current_state(smb)
        show_step(res)

    def get_current_state(smb):
        # Mimic the structure returned by step(), for use by show_step
        res = {}
        for zone in smb.zones:
            res[zone] = []
            for obj in smb.zones[zone]:
                arrs = []
                for comp in getattr(obj, 'components', []):
                    if hasattr(comp, 'c'):
                        arrs.append(comp.c.copy())
                res[zone].append(arrs)
        return res

    def get_elapsed_wall_time():
        if sim_state["running"]:
            return elapsed_wall_accum[0] + (time.time() - wall_time_start[0])
        else:
            return elapsed_wall_accum[0]

    def on_close():
        opcua_server.stop()
        close_logfile()
        os._exit(0)  # Terminates Python

    def on_speed_change(*args):
        log.insert('end', f"Simulation speed set to {speed_factor.get()}x\n")
        log.see('end')
        shared_state['speed_factor'] = float(speed_factor.get())
        if log_file_handle[0] is not None:
            log_file_handle[0].write(f"[INFO] speed_factor={speed_factor.get()}x\n")

        # NEW: re-pin baseline so target == current sim time after a speed change
        now = time.time()
        sim_start_wall[0] = now - float(smb.timer) / float(speed_factor.get())

    def on_mode_change():
        mode = mode_var.get()
        shared_state['mode'] = mode
        if log_file_handle[0] is not None:
            log_file_handle[0].write(f"[INFO] mode={mode}\n")

        widgets = [feed_scale, q1_scale, q2_scale, q4_scale, switch_scale,
                   speed_combo, start_btn, stop_btn, reset_btn, apply_btn]

        if mode == "manual":
            for w in widgets:
                # ttk.Combobox must be 'readonly' to remain selectable
                if w is speed_combo:
                    w.config(state='readonly')
                else:
                    w.config(state='normal')
        else:
            for w in widgets:
                # disable everything except the radio buttons themselves
                w.config(state='disabled')

    speed_combo.bind('<<ComboboxSelected>>', on_speed_change)

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Initial draw
    res0 = {}
    for z in range(1, 5):
        res0[z] = [
            [np.zeros(params['Nx']), np.zeros(params['Nx'])],  # object 0
            [np.zeros(params['Nx']), np.zeros(params['Nx'])],  # object 1
        ]
    show_step(res0)

    root.mainloop()

if __name__ == "__main__":
    main()
