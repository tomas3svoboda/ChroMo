import os
import threading
import time
import tkinter as tk
from tkinter import ttk
from opcua_server import SMBOPCUAServer
#from SMB.SMBStation import SMBStation
#from SMB.LinColumn_working import LinColumn
#from SMB.Tube import Tube
from ChroMo_SMB.EKF.SMB.SMBStation import SMBStation
from ChroMo_SMB.EKF.SMB.LinColumn import LinColumn
from ChroMo_SMB.EKF.SMB.Tube import Tube

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
def smb_all_flows(feed, eluent, extract, recycle):
    # V_F, V_El, V_Ex, V_Re = feed, eluent, extract, recycle
    V_I   = eluent + recycle
    V_II  = V_I - extract
    V_III = V_II + feed
    V_IV  = recycle
    raffinate = V_III - recycle  # V_Ra = V_III - V_IV
    # kontrola: V_F + V_El == V_Ex + V_Ra
    assert abs((feed + eluent) - (extract + raffinate)) < 1e-6, "Balance error!"
    return {
        "feed": feed,
        "eluent": eluent,
        "extract": extract,
        "recycle": recycle,
        "raffinate": raffinate,
        "zone1": V_I,
        "zone2": V_II,
        "zone3": V_III,
        "zone4": V_IV,
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
                    time.sleep(0.001)
                    continue

                dt = self.params['dt']
                # use regular division; int() will floor
                nsteps = int(min(sim_time_needed / dt, self.steps_per_update))
                if nsteps <= 0:
                    # we’re already at (or within one dt of) the target → let wall time advance
                    time.sleep(0.001)
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
        smb.addColZone(zone + 1, LinColumn(params['col_length'], params['col_diameter'], params['porosity']), Tube(params['dead_volume']))
    for i, fr in enumerate(zone_flows, 1):
        smb.setFlowRateZone(i, fr)
    smb.setSwitchInterval(params['switch_interval'])
    smb.setdt(params['dt'])
    smb.setNx(params['Nx'])
    for comp in components:
        smb.createComponentAB(comp["name"], comp["feed_concentration"], comp["henry_constant"], comp["delta"], comp["Di"])
    smb.initCols()
    return smb


def main():

    # --- Default system parameters ---
    params = {
        'dt': 0.05,
        'Nx': 100,
        'switch_interval': 1894,
        'col_length': 320,
        'col_diameter': 10,
        'porosity': 0.376,
        'dead_volume': 0.5
    }
    components = [
        {"name": "Man", "feed_concentration": 9, "henry_constant": 4.55, "delta": 54, "Di": 0.0007},
        {"name": "Gal", "feed_concentration": 6, "henry_constant": 2.77, "delta": 84, "Di": 0.0007}
    ]
    # Default operating values
    flows = {
        "feed": 20.0,
        "eluent": 170.0,
        "extract": 32.0,
        "recycle": 2.0
    }

    # OPC UA inplementation
    shared_state = {
        'mode': 'manual',
        'active_feed': flows['feed'],
        'active_eluent': flows['eluent'],
        'active_extract': flows['extract'],
        'active_recycle': flows['recycle'],
        'active_switch_interval': params['switch_interval'],
        'next_revision': 0,
        'last_applied_revision': -1,
        'last_apply_timestamp': '',
        'feed': flows["feed"],
        'eluent': flows["eluent"],
        'extract': flows["extract"],
        'recycle': flows["recycle"],
        'switch_interval': params['switch_interval'],
        'extract_concentration_man': 0.0,
        'extract_concentration_gal': 0.0,
        'raffinate_concentration_man': 0.0,
        'raffinate_concentration_gal': 0.0,
        'extract_outlet_flow': 0.0,
        'raffinate_outlet_flow': 0.0,
        'simulation_time': 0.0,
        'is_running': False,
        'switch_countdown': 0.0,
        'elapsed_wall_time': 0.0,
        'speed_factor': 1.0,
    }

    wall_time_start = [None]   # Use list for mutability
    elapsed_wall_accum = [0.0]  # Accumulates wall time over multiple runs
    sim_start_wall = [None]     # Used foe elapsed wall time and simulation time synchronization logic

    last_switch_countdown = [None]  # tracks previous countdown to detect switch edge in auto mode

    computed = smb_all_flows(flows["feed"], flows["eluent"], flows["extract"], flows["recycle"])
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
    feed_scale = tk.Scale(ctrl, from_=0, to=50, variable=feed_var, orient='horizontal', resolution=1)
    feed_scale.pack(fill='x')

    tk.Label(ctrl, text="Eluent (D) [mL/h]").pack(anchor='w')
    eluent_var = tk.DoubleVar(value=flows["eluent"])
    eluent_scale = tk.Scale(ctrl, from_=0, to=200, variable=eluent_var, orient='horizontal', resolution=1)
    eluent_scale.pack(fill='x')

    tk.Label(ctrl, text="Extract (E) [mL/h]").pack(anchor='w')
    extract_var = tk.DoubleVar(value=flows["extract"])
    extract_scale = tk.Scale(ctrl, from_=0, to=100, variable=extract_var, orient='horizontal', resolution=1)
    extract_scale.pack(fill='x')

    tk.Label(ctrl, text="Recycle (Q4) [mL/h]").pack(anchor='w')
    recycle_var = tk.DoubleVar(value=flows["recycle"])
    recycle_scale = tk.Scale(ctrl, from_=0, to=100, variable=recycle_var, orient='horizontal', resolution=1)
    recycle_scale.pack(fill='x')

    tk.Label(ctrl, text="Switch interval [s]").pack(anchor='w')
    switch_var = tk.DoubleVar(value=params['switch_interval'])
    switch_scale = tk.Scale(ctrl, from_=100, to=4000, variable=switch_var, orient='horizontal', resolution=1)
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
            F = feed_var.get()
            D = eluent_var.get()
            E = extract_var.get()
            Q4 = recycle_var.get()
            computed = smb_all_flows(F, D, E, Q4)
            flows_display.delete('1.0', 'end')
            flows_display.insert('end',
                f" Raffinate: {computed['raffinate']:.3f} mL/h\n"
                f" Zone I:    {computed['zone1']:.3f} mL/h\n"
                f" Zone II:   {computed['zone2']:.3f} mL/h\n"
                f" Zone III:  {computed['zone3']:.3f} mL/h\n"
                f" Zone IV:   {computed['zone4']:.3f} mL/h\n"
            )
        except Exception as e:
            flows_display.delete('1.0', 'end')
            flows_display.insert('end', f"Invalid flows: {e}")

    for var in [feed_var, eluent_var, extract_var, recycle_var]:
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
    speed_combo = ttk.Combobox(ctrl, textvariable=speed_factor, values=[1,2,5,10,20], state="readonly")
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

        # Flows at outlets
        computed = smb_all_flows(
            shared_state['feed'],
            shared_state['eluent'],
            shared_state['extract'],
            shared_state['recycle']
        )
        shared_state['extract_outlet_flow'] = computed['extract']
        shared_state['raffinate_outlet_flow'] = computed['raffinate']

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
        try:
            prev = last_switch_countdown[0]
            curr = float(smb.countdown)
            last_switch_countdown[0] = curr
            switched = (prev is not None and curr > prev)

            if switched and shared_state.get('mode') == 'automatic':
                F = float(shared_state['feed'])
                D = float(shared_state['eluent'])
                E = float(shared_state['extract'])
                Q4 = float(shared_state['recycle'])
                si = float(shared_state['switch_interval'])
                computed2 = smb_all_flows(F, D, E, Q4)
                for zone, flow in enumerate([computed2['zone1'], computed2['zone2'], computed2['zone3'], computed2['zone4']], 1):
                    smb.setFlowRateZone(zone, flow)
                smb.setSwitchInterval(si)
                # Update Active* mirror & diagnostics
                shared_state['active_feed'] = F
                shared_state['active_eluent'] = D
                shared_state['active_extract'] = E
                shared_state['active_recycle'] = Q4
                shared_state['active_switch_interval'] = si
                shared_state['last_applied_revision'] = int(shared_state.get('next_revision', 0))
                shared_state['last_apply_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
                now_str = shared_state['last_apply_timestamp']
                log.insert('end', f"[AUTO] {now_str} Applied OPC UA setpoints at switch: "
                                  f"F={F:.2f} D={D:.2f} E={E:.2f} Q4={Q4:.2f}  SI={si:.1f}\n")
                log.see('end')
                if log_file_handle[0] is not None:
                    log_file_handle[0].write(f"[AUTO] {now_str} apply-at-switch F={F:.2f} D={D:.2f} E={E:.2f} Q4={Q4:.2f} SI={si:.1f}\n")

            elif switched:
                # MANUAL mode: only apply a *deferred* switch interval (flows already applied live on button press)
                si_target = float(shared_state['switch_interval'])
                si_active = float(shared_state.get('active_switch_interval', getattr(smb, 'interval', si_target)))
                if abs(si_target - si_active) > 1e-9:
                    smb.setSwitchInterval(si_target)
                    shared_state['active_switch_interval'] = si_target
                    shared_state['last_apply_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    now_str = shared_state['last_apply_timestamp']
                    log.insert('end', f"[MANUAL] {now_str} Applied deferred switch interval at boundary: SI={si_target:.1f}\n")
                    log.see('end')
                    if log_file_handle[0] is not None:
                        log_file_handle[0].write(f"[MANUAL] {now_str} apply-SI-at-switch SI={si_target:.6g}\n")

        except Exception as _e:
            # Don't crash the UI if something goes wrong; just note it.
            try:
                log.insert('end', f"[AUTO] Apply-on-switch failed: {_e}\n")
                log.see('end')
            except Exception:
                pass

        # 6) File logging of PVs each update (throttle to ~10 Hz sim-time if very fast)
        try:
            if log_file_handle[0] is not None:
                # Throttle by sim-time step to avoid huge files when speed is high
                if (last_file_log_simtime[0] is None) or (t_now - last_file_log_simtime[0] >= params['dt']):
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
                        f"c_feed_Man={feed_man:.6g}\t"
                        f"c_feed_Gal={feed_gal:.6g}\t"
                        f"F={shared_state['active_feed']:.6g}\tD={shared_state['active_eluent']:.6g}\tE={shared_state['active_extract']:.6g}\tQ4={shared_state['active_recycle']:.6g}\tSI={shared_state['active_switch_interval']:.6g}\t"
                        f"mode={mode}\tspd={sf}\n"
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
            F = feed_var.get()
            D = eluent_var.get()
            E = extract_var.get()
            Q4 = recycle_var.get()
            switch_time = switch_var.get()

            computed = smb_all_flows(F, D, E, Q4)

            # --- Live update parameters in the simulator ---
            # Flows are applied immediately
            for zone, flow in enumerate(
                [computed['zone1'], computed['zone2'], computed['zone3'], computed['zone4']], 1
            ):
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
            shared_state['eluent'] = D
            shared_state['extract'] = E
            shared_state['recycle'] = Q4
            shared_state['switch_interval'] = switch_time

            # --- Keep Active* in sync on manual apply ---
            if shared_state.get('mode') == 'manual':
                shared_state['active_feed'] = F
                shared_state['active_eluent'] = D
                shared_state['active_extract'] = E
                shared_state['active_recycle'] = Q4
                shared_state['last_applied_revision'] = int(shared_state.get('next_revision', 0))
                shared_state['last_apply_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
                if not defer_si:
                    shared_state['active_switch_interval'] = switch_time
                # Reflect changes to OPC UA immediately (manual mode)
                try:
                    opcua_server.push_from_shared_state()
                except Exception as _e:
                    log.insert('end', f"[WARN] OPC UA push failed: {_e}\n")

            # --- UI log ---
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            log.insert(
                'end',
                f"[{now_str}] Parameters applied (live update):\n"
                f" F={F:.2f}  D={D:.2f}  E={E:.2f}  Q4={Q4:.2f}\n"
                f" R={computed['raffinate']:.2f}\n"
                f" Zones: {computed['zone1']:.2f}, {computed['zone2']:.2f}, "
                f"{computed['zone3']:.2f}, {computed['zone4']:.2f}\n"
                f" Switch interval: {switch_time:.1f} s\n"
            )
            if defer_si:
                log.insert('end', " (SI change deferred: will take effect at the next switch)\n")
            else:
                log.insert('end', " (SI applied immediately)\n")

            # --- file log ---
            if log_file_handle[0] is not None:
                log_file_handle[0].write(
                    f"[{now_str}] params_apply F={F:.6g} D={D:.6g} E={E:.6g} Q4={Q4:.6g} SI={switch_time:.6g}\n"
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

        # header
        hdr = (
            "# SMB simulation log\n"
            f"# started_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# params dt={params['dt']} Nx={params['Nx']} L={params['col_length']}mm D={params['col_diameter']}mm eps={params['porosity']} Vdead={params['dead_volume']}\n"
            f"# components: {', '.join([c['name'] for c in components])}; feed_conc={[c['feed_concentration'] for c in components]}\n"
            f"# initial flows F={shared_state['active_feed']} D={shared_state['active_eluent']} E={shared_state['active_extract']} Q4={shared_state['active_recycle']} SI={shared_state['active_switch_interval']}\n"
            "# columns: t_sim\tt_wall\tt_to_switch\tc_ex_Man\tc_ex_Gal\tc_ra_Man\tc_ra_Gal\tc_feed_Man\tc_feed_Gal\tF\tD\tE\tQ4\tSI\tmode\tspd\n"
        )
        fh.write(hdr)
        fh.flush()

        sim_thread[0] = SimulationThread(
            smb, params, components, show_step, speed_factor, sim_start_wall,
            ui_post=lambda fn, *a, **k: root.after(0, lambda: fn(*a, **k)),
            throttle_sec=0.10  # ~10 Hz UI
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
        shared_state['mode'] = mode_var.get()
        if log_file_handle[0] is not None:
            log_file_handle[0].write(f"[INFO] mode={mode_var.get()}\n")
        if mode_var.get() == "manual":
            # Enable all controls
            for widget in [feed_scale, eluent_scale, extract_scale, recycle_scale, switch_scale, speed_combo,
                        start_btn, stop_btn, reset_btn, apply_btn]:
                widget.config(state='normal' if widget != speed_combo else 'readonly')
        else:
            # Disable all controls (except mode switch)
            for widget in [feed_scale, eluent_scale, extract_scale, recycle_scale, switch_scale, speed_combo,
                        start_btn, stop_btn, reset_btn, apply_btn]:
                widget.config(state='disabled')

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
