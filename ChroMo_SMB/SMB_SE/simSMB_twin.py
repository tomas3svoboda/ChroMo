
import os
import time
import threading
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Backends: let matplotlib choose a GUI backend available on the host.
# (No explicit Tk embedding is used.)
# matplotlib.use('Qt5Agg')  # uncomment if you prefer a specific backend

from ChroMo_SMB.opcua_server import SMBOPCUAServer
from ChroMo_SMB.EKF.SMB.SMBStation import SMBStation
from ChroMo_SMB.EKF.SMB.LinColumn import LinColumn
from ChroMo_SMB.EKF.SMB.Tube import Tube

# ---- Matplotlib defaults for readability ----
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.2
})

# ==================== Helpers & Model Init ====================
def smb_all_flows(feed, eluent, extract, recycle):
    V_I   = eluent + recycle
    V_II  = V_I - extract
    V_III = V_II + feed
    V_IV  = recycle
    raffinate = V_III - recycle  # V_Ra = V_III - V_IV
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

def init_smb(params, components, zone_flows):
    smb = SMBStation()
    for zone in range(4):
        smb.addColZone(
            zone + 1,
            LinColumn(params['col_length'], params['col_diameter'], params['porosity']),
            Tube(params['dead_volume'])
        )
    for i, fr in enumerate(zone_flows, 1):
        smb.setFlowRateZone(i, fr)
    smb.setSwitchInterval(params['switch_interval'])
    smb.setdt(params['dt'])
    smb.setNx(params['Nx'])
    for comp in components:
        smb.createComponentAB(
            comp["name"],
            comp["feed_concentration"],
            comp["henry_constant"],
            comp["delta"],
            comp["Di"]
        )
    smb.initCols()
    return smb

def get_current_state(smb):
    # Capture the structure like smb.step() returns: per-zone lists of objects with comp arrays
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

# ==================== Main ====================
def main():
    # ---- Parameters & components (same defaults as interactive app) ----
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
    flows = {
        "feed": 20.0,
        "eluent": 170.0,
        "extract": 32.0,
        "recycle": 2.0
    }

    # ---- Shared state for OPC UA publishing ----
    shared_state = {
        'mode': 'automatic',             # PoC runs in automatic setpoint-apply-at-switch
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

    # ---- Build simulator with consistent zone flows ----
    computed = smb_all_flows(flows["feed"], flows["eluent"], flows["extract"], flows["recycle"])
    zone_flows = [computed['zone1'], computed['zone2'], computed['zone3'], computed['zone4']]
    smb = init_smb(params, components, zone_flows)

    # ---- Start OPC UA server ----
    opcua_server = SMBOPCUAServer(shared_state)
    opcua_server.start()

    # ==================== Three-Plot Figure ====================
    fig = plt.figure(figsize=(10.5, 5.2), dpi=110)
    # Layout: left = big spatial profile, right top = extract history, right bottom = raffinate history
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
    ax_profile = fig.add_subplot(gs[:, 0])
    ax_ex = fig.add_subplot(gs[0, 1])
    ax_ra = fig.add_subplot(gs[1, 1])
    fig.canvas.manager.set_window_title("SMB Digital Twin PoC (3 plots)")

    x_axis = np.linspace(0, 4 * params['col_length'], 4 * params['Nx'])
    history = {
        "extract": {"Man": [], "Gal": []},     # store (t, val) tuples
        "raffinate": {"Man": [], "Gal": []},
    }

    # History window = last 5 switch intervals (using ACTIVE, not target)
    def history_window_seconds():
        return 5.0 * float(shared_state.get('active_switch_interval', params['switch_interval']))

    def trim_history(window_s, t_now):
        for side in ("extract", "raffinate"):
            for comp in ("Man", "Gal"):
                seq = history[side][comp]
                # drop leading data older than window
                while seq and seq[0][0] < t_now - window_s:
                    seq.pop(0)

    # ---- Plot refresh ----
    def refresh_plots(res):
        # Spatial profile over 4 columns (concatenate 4 zones' internal profiles)
        ax_profile.clear()
        for idx, comp in enumerate(components):
            profiles = []
            for zone in range(4):
                objs = res[zone + 1]                  # list of objects in the zone
                obj_idx = 1 if len(objs) > 1 else 0   # prefer the 2nd object if present (column), else first
                arr = np.asarray(objs[obj_idx])       # (n_components, Nx) or (Nx,)
                if arr.ndim == 2:
                    profiles.append(arr[idx, :])
                elif arr.ndim == 1:
                    profiles.append(arr)
            full_profile = np.hstack(profiles) if profiles else np.zeros_like(x_axis)
            ax_profile.plot(x_axis, full_profile, label=comp["name"])

        for boundary in range(1, 4):
            ax_profile.axvline(boundary * params['col_length'], color='black', linestyle='--', linewidth=0.8)
        ax_profile.set_xlabel("Continuous Column Length [mm]")
        ax_profile.set_ylabel("Concentration")
        ax_profile.legend()
        ax_profile.grid(False)

        # Extract & Raffinate outlets (last element of last object in zone 1 and 3 respectively)
        try:
            ex_man = float(res[1][-1][0][-1])
            ex_gal = float(res[1][-1][1][-1])
            ra_man = float(res[3][-1][0][-1])
            ra_gal = float(res[3][-1][1][-1])
        except Exception:
            ex_man = ex_gal = ra_man = ra_gal = 0.0

        t_now = float(smb.timer)
        history['extract']['Man'].append((t_now, ex_man))
        history['extract']['Gal'].append((t_now, ex_gal))
        history['raffinate']['Man'].append((t_now, ra_man))
        history['raffinate']['Gal'].append((t_now, ra_gal))
        trim_history(history_window_seconds(), t_now)

        # Update shared_state for OPC UA consumers
        shared_state['extract_concentration_man'] = ex_man
        shared_state['extract_concentration_gal'] = ex_gal
        shared_state['raffinate_concentration_man'] = ra_man
        shared_state['raffinate_concentration_gal'] = ra_gal

        # Time-series plots
        def plot_hist(ax, dq_man, dq_gal, title):
            ax.clear()
            if dq_man:
                t_m, v_m = zip(*dq_man)
                ax.plot(t_m, v_m, label='Man')
            if dq_gal:
                t_g, v_g = zip(*dq_gal)
                ax.plot(t_g, v_g, label='Gal')
            tmax = max(t_now, 1.0)
            ax.set_xlim(max(0.0, t_now - history_window_seconds()), tmax)
            ax.set_xlabel('t [s]')
            ax.set_ylabel('c [g/L]')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        plot_hist(ax_ex, history['extract']['Man'], history['extract']['Gal'], "Extract outlet (last 5 SI)")
        plot_hist(ax_ra, history['raffinate']['Man'], history['raffinate']['Gal'], "Raffinate outlet (last 5 SI)")

        fig.tight_layout()
        plt.pause(0.001)  # yield to GUI loop

    # ==================== Real-Time Stepping Thread ====================
    class Stepper(threading.Thread):
        def __init__(self, speed_factor=1.0, throttle_sec=0.10, max_steps_per_update=200):
            super().__init__(daemon=True)
            self._stop = threading.Event()
            self.speed_factor = float(speed_factor)
            self.throttle_sec = float(throttle_sec)
            self.max_steps_per_update = int(max_steps_per_update)
            self.sim_start_wall = time.time()  # pinned so both PoC and plant sim can align
            self.last_draw = time.perf_counter()
            self.last_switch_countdown = None

        def set_speed(self, sf):
            self.speed_factor = float(sf)
            # Re-pin baseline so target == current sim time after a speed change
            self.sim_start_wall = time.time() - float(smb.timer) / self.speed_factor

        def stop(self):
            self._stop.set()

        def run(self):
            shared_state['is_running'] = True
            shared_state['speed_factor'] = self.speed_factor
            try:
                while not self._stop.is_set():
                    # Target sim time derived from wall clock & speed
                    wall_elapsed = time.time() - self.sim_start_wall
                    target_sim_time = wall_elapsed * self.speed_factor

                    sim_time_now = float(smb.timer)
                    sim_time_needed = target_sim_time - sim_time_now
                    if sim_time_needed <= 0:
                        time.sleep(0.001)
                        continue

                    dt = float(params['dt'])
                    nsteps = int(min(sim_time_needed / dt, self.max_steps_per_update))
                    if nsteps <= 0:
                        time.sleep(0.001)
                        continue

                    res = smb.step(nsteps)

                    # Publish basic telemetry to shared_state
                    shared_state['simulation_time'] = float(smb.timer)
                    shared_state['switch_countdown'] = float(smb.countdown)
                    shared_state['elapsed_wall_time'] = wall_elapsed

                    # Apply staged setpoints exactly at switch boundary (rising countdown edge)
                    try:
                        prev = self.last_switch_countdown
                        curr = float(smb.countdown)
                        self.last_switch_countdown = curr
                        switched = (prev is not None and curr > prev)

                        if switched and shared_state.get('mode') == 'automatic':
                            F = float(shared_state['feed'])
                            D = float(shared_state['eluent'])
                            E = float(shared_state['extract'])
                            Q4 = float(shared_state['recycle'])
                            si = float(shared_state['switch_interval'])
                            comp2 = smb_all_flows(F, D, E, Q4)
                            for zone, flow in enumerate([comp2['zone1'], comp2['zone2'], comp2['zone3'], comp2['zone4']], 1):
                                smb.setFlowRateZone(zone, flow)
                            smb.setSwitchInterval(si)
                            shared_state['active_feed'] = F
                            shared_state['active_eluent'] = D
                            shared_state['active_extract'] = E
                            shared_state['active_recycle'] = Q4
                            shared_state['active_switch_interval'] = si
                            shared_state['last_applied_revision'] = int(shared_state.get('next_revision', 0))
                            shared_state['last_apply_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass

                    # Throttled redraw
                    now = time.perf_counter()
                    if (now - self.last_draw) >=  self.throttle_sec:
                        refresh_plots(res)
                        self.last_draw = now

            finally:
                shared_state['is_running'] = False

    # ==================== Run ====================
    # Initial blank draw
    res0 = {z: [[np.zeros(params['Nx']), np.zeros(params['Nx'])],
                [np.zeros(params['Nx']), np.zeros(params['Nx'])]] for z in range(1,5)}
    refresh_plots(res0)

    stepper = Stepper(speed_factor=1.0, throttle_sec=0.10, max_steps_per_update=200)
    stepper.start()

    print("[SMB Twin PoC] Running. Close the plot window or press Ctrl+C to stop.")
    try:
        while plt.fignum_exists(fig.number):
            # Keep GUI alive; user can inspect values via OPC UA
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stepper.stop()
        stepper.join(timeout=2.0)
        try:
            opcua_server.stop()
        except Exception:
            pass
        print("[SMB Twin PoC] Stopped.")

if __name__ == "__main__":
    main()
