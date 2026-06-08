import time
import math
from collections import deque
import csv
import os

from EKF.SMB.SMBStation import SMBStation
from EKF.SMB.LinColumn import LinColumn
from EKF.SMB.Tube import Tube
from opcua_client import SMB_OPCUAClient

# =========================
# Config
# =========================
# Monitoring
_MONITOR_MIN_PERIOD = 20.0   # seconds
ERR_WINDOW = 600            # ~30 s if loop ~50 ms
ALERT_THRESH_ABS = 5e-4     # absolute error threshold (domain sensible for 0–10 scale)
ALERT_HOLD = 20             # consecutive samples above threshold before alert
LOG_CSV = True
CSV_PATH = "delta_monitor3.csv"

# Equality tolerances
ABS_ATOL = 1e-9   # treat <1e-9 as 0
REL_RTOL = 1e-6   # 1 ppm relative tol

# Station defaults (your system)
PARAMS = {
    'dt': 0.05, 'Nx': 100, 'switch_interval': 1894,
    'col_length': 320.0, 'col_diameter': 10.0,
    'porosity': 0.376, 'dead_volume': 0.5
}
COMPONENTS = [
    {"name": "Man", "feed_concentration": 9.0, "henry_constant": 4.55, "delta": 54.0, "Di": 0.0007},
    {"name": "Gal", "feed_concentration": 6.0, "henry_constant": 2.77, "delta": 84.0, "Di": 0.0007},
]
FLOWS_DEFAULT = {"feed": 20.0, "eluent": 170.0, "extract": 32.0, "recycle": 2.0}

# Auto-detect EX/RA mapping? (True recommended)
AUTO_MAP = True
# Start assuming EX = zone 1, RA = zone 3 (this matches your engine’s comment)
EX_ZONE, RA_ZONE = 1, 3

# Chunking for step catch-up
MAX_STEPS_PER_LOOP = 2000  # keep UI responsive while deterministically catching up

# =========================
# Globals / diagnostics
# =========================
buf_ex_m, buf_ex_g = deque(maxlen=ERR_WINDOW), deque(maxlen=ERR_WINDOW)
buf_ra_m, buf_ra_g = deque(maxlen=ERR_WINDOW), deque(maxlen=ERR_WINDOW)
cnt_ex_m = cnt_ex_g = cnt_ra_m = cnt_ra_g = 0

_last_print = 0.0
_last_pv_print = 0.0
_last_plant_t = None
_stall_count = 0

_prev_cd = None
_interval_guess = float(PARAMS['switch_interval'])

# for CSV (batch-flushed)
_csv_writer = None
_csv_fh = None
_csv_pending = 0
_CSV_FLUSH_EVERY = 50

# mapping auto-detect accumulators
_map_score_normal = 0.0
_map_score_swapped = 0.0
_map_eval_every = 50  # samples
_map_counter = 0

# =========================
# Helpers (NO ADVANCE!)
# =========================
def _stats(dq: deque):
    if not dq:
        return 0.0, 0.0, 0.0
    n = len(dq)
    s = sum(dq)
    mx = max(dq)
    rmse = (sum(e*e for e in dq) / n) ** 0.5
    return s / n, mx, rmse

def _maybe_alert(name: str, err_abs: float, counter: int) -> int:
    if err_abs >= ALERT_THRESH_ABS:
        counter += 1
        if counter == ALERT_HOLD:
            print(f"[monitor][ALERT] {name} abs error ≥ {ALERT_THRESH_ABS:g} for {ALERT_HOLD} samples", flush=True)
    else:
        counter = 0
    return counter

def close(a, b, atol=ABS_ATOL, rtol=REL_RTOL):
    return abs(a - b) <= max(atol, rtol * max(1.0, abs(a), abs(b)))

def target_time_from_plant(plant_t: float, dt: float) -> float:
    """Compare at the plant's last completed discrete step (never ahead)."""
    k = math.floor((plant_t + 1e-12) / dt)
    return k * dt

def steps_to_catch_up(twin_t: float, target_t: float, dt: float) -> int:
    """Exact integer step deficit from twin to plant target step."""
    k_twin   = int(round(twin_t / dt))
    k_target = int(round(target_t / dt))
    need = k_target - k_twin
    return need if need > 0 else 0

def countdown_rotate_if_needed(twin: SMBStation, snap: dict):
    """Rotate when SwitchCountdown wraps (works even if no revision info)."""
    global _prev_cd
    cd_raw = snap.get("SwitchCountdown")
    if cd_raw is None:
        return
    try:
        cd = float(cd_raw)
    except Exception:
        return
    if _prev_cd is None:
        _prev_cd = cd
        return
    # wrap if cd jumps up a lot or gets close to interval
    if cd > _prev_cd + 0.5 * _interval_guess or cd > _interval_guess * 0.8:
        twin.rotate()
        twin.countdown = cd
    _prev_cd = cd

def build_station_like_plant() -> SMBStation:
    """Instantiate SMBStation exactly like the plant uses."""
    p = PARAMS
    smb = SMBStation()
    smb.addColZone(1, LinColumn(p['col_length'], p['col_diameter'], p['porosity']), Tube(p['dead_volume']))
    smb.addColZone(2, LinColumn(p['col_length'], p['col_diameter'], p['porosity']), Tube(p['dead_volume']))
    smb.addColZone(3, LinColumn(p['col_length'], p['col_diameter'], p['porosity']), Tube(p['dead_volume']))
    smb.addColZone(4, LinColumn(p['col_length'], p['col_diameter'], p['porosity']), Tube(p['dead_volume']))
    smb.setdt(p['dt']); smb.setNx(p['Nx']); smb.setSwitchInterval(p['switch_interval'])
    c0, c1 = COMPONENTS
    smb.createComponentAB(c0["name"], c0["feed_concentration"], c0["henry_constant"], c0["delta"], c0["Di"])
    smb.createComponentAB(c1["name"], c1["feed_concentration"], c1["henry_constant"], c1["delta"], c1["Di"])
    # initial flows (overwritten by Active* every loop)
    smb.setFlowRateZone(2, FLOWS_DEFAULT["recycle"])
    smb.setFlowRateZone(3, FLOWS_DEFAULT["recycle"] + FLOWS_DEFAULT["feed"])
    smb.setFlowRateZone(1, FLOWS_DEFAULT["eluent"])
    smb.setFlowRateZone(4, FLOWS_DEFAULT["extract"])
    smb.initCols()
    return smb

def apply_active_flows(smb: SMBStation, snap: dict) -> None:
    feed    = float(snap.get("ActiveFeedFlow",    FLOWS_DEFAULT["feed"]))
    eluent  = float(snap.get("ActiveEluentFlow",  FLOWS_DEFAULT["eluent"]))
    extract = float(snap.get("ActiveExtractFlow", FLOWS_DEFAULT["extract"]))
    recycle = float(snap.get("ActiveRecycleFlow", FLOWS_DEFAULT["recycle"]))
    # Mapping consistent with engine mixing:
    #   zone3 inlet is (recycle + feed)
    #   zone1 inlet uses recycle scaling
    smb.setFlowRateZone(2, recycle)
    smb.setFlowRateZone(3, recycle + feed)
    smb.setFlowRateZone(1, eluent)
    smb.setFlowRateZone(4, extract)

def apply_switch_timing(smb: SMBStation, snap: dict) -> None:
    """Mirror applied switch interval + countdown; track interval for wrap detection."""
    global _interval_guess
    si = snap.get("ActiveSwitchInterval")
    if si not in (None, "", "None"):
        smb.setSwitchInterval(float(si))
        _interval_guess = float(si)
    else:
        smb.setSwitchInterval(float(PARAMS["switch_interval"]))
        _interval_guess = float(PARAMS["switch_interval"])
    cd = snap.get("SwitchCountdown")
    if cd is not None:
        try:
            smb.countdown = float(cd)
        except Exception:
            pass

def _peek_zone_outlet(smb: SMBStation, zone: int):
    """
    NO-ADVANCE peek of (Man, Gal) at the outlet of a zone: read last object's c[-1].
    Works if the last object is Tube or Column; both hold Component.c arrays.
    """
    try:
        comps = smb.zones[zone][-1].components
        return float(comps[0].c[-1]), float(comps[1].c[-1])
    except Exception:
        return 0.0, 0.0

def peek_outlets_now(smb: SMBStation, ex_zone: int, ra_zone: int):
    """Return (EX_Man, EX_Gal, RA_Man, RA_Gal) WITHOUT advancing time."""
    ex_m, ex_g = _peek_zone_outlet(smb, ex_zone)
    ra_m, ra_g = _peek_zone_outlet(smb, ra_zone)
    return ex_m, ex_g, ra_m, ra_g

# =========================
# Main
# =========================
def main():
    print("[sync] starting…", flush=True)
    cli = SMB_OPCUAClient(endpoint="opc.tcp://127.0.0.1:4840")

    twin = None
    waiting_logged = False

    global _last_plant_t, _stall_count, _last_print, _last_pv_print, _prev_cd
    global cnt_ex_m, cnt_ex_g, cnt_ra_m, cnt_ra_g
    global _csv_writer, _csv_fh, _csv_pending
    global EX_ZONE, RA_ZONE, _map_score_normal, _map_score_swapped, _map_counter

    # CSV setup (single open; batch flush)
    if LOG_CSV and _csv_writer is None:
        try:
            _csv_fh = open(CSV_PATH, "w", newline="", buffering=1_048_576)
            _csv_writer = csv.writer(_csv_fh)
            _csv_writer.writerow(["t",
                                  "EX_M_twin","EX_M_plant","EX_M_abs_err",
                                  "EX_G_twin","EX_G_plant","EX_G_abs_err",
                                  "RA_M_twin","RA_M_plant","RA_M_abs_err",
                                  "RA_G_twin","RA_G_plant","RA_G_abs_err",
                                  "map", "ex_zone", "ra_zone"])
        except Exception:
            _csv_writer = None

    while True:
        snap = cli.read_snapshot()

        # 0) Wait for the plant to run
        if not bool(snap.get("IsRunning", False)):
            if not waiting_logged:
                print("[sync] Plant is not running; twin idling (waiting for IsRunning=True).", flush=True)
                waiting_logged = True
            time.sleep(0.2)
            continue
        waiting_logged = False

        # 1) Heartbeat & stall detection
        now = time.time()
        plant_t = float(snap.get("SimulationTime", 0.0))
        if _last_plant_t is not None:
            _stall_count = _stall_count + 1 if plant_t <= _last_plant_t + 1e-12 else 0
        _last_plant_t = plant_t
        if now - _last_print >= 120.0:
            print(f"[sync] heartbeat: plant_t={plant_t:.2f}s, twin_t={(twin.timer if twin else 0.0):.2f}s, "
                  f"SwitchCountdown={snap.get('SwitchCountdown', 'NA')}, stall_count={_stall_count}", flush=True)
            _last_print = now
        if _stall_count >= 20:
            print("[sync][warn] Plant SimulationTime not increasing while IsRunning=True. "
                  "Is the server paused or SpeedFactor=0?", flush=True)
            _stall_count = 0

        # 2) Build twin on first run and align numerics
        if twin is None:
            twin = build_station_like_plant()
            apply_active_flows(twin, snap)
            apply_switch_timing(twin, snap)
            if not getattr(twin, "dt", 0.0):
                raise RuntimeError("Twin dt not set after build; check builder order.")
            print("[sync] Twin initialized; locking to plant…", flush=True)
            _prev_cd = float(snap.get("SwitchCountdown", twin.countdown))

        # 3) Always mirror latest Active* flows & timing (cheap + avoids drift)
        apply_active_flows(twin, snap)
        apply_switch_timing(twin, snap)

        # 4) Rotate on countdown wrap
        countdown_rotate_if_needed(twin, snap)

        # 5) Deterministic catch-up to plant's last completed step
        dt = float(twin.dt)
        t_target = target_time_from_plant(plant_t, dt)
        need = steps_to_catch_up(float(twin.timer), t_target, dt)
        if need > 0:
            # scale chunk with lag but cap it
            chunk = min(max(100, need // 4), MAX_STEPS_PER_LOOP)
            twin.step(steps=chunk)

        # only compare when we are exactly at the target instant (never ahead)
        lag_steps = steps_to_catch_up(float(twin.timer), t_target, dt)
        if lag_steps > 0:
            if time.time() - _last_print >= 1.0:
                print(f"[sync] catching-up: lag_steps={lag_steps}, plant_t={plant_t:.2f}s, "
                      f"twin_t={twin.timer:.2f}s, t_target={t_target:.2f}s", flush=True)
                _last_print = time.time()
            time.sleep(0.01)
            continue

        # 6) Compare PVs at the same discrete instant (NO ADVANCE)
        # Read plant PVs
        ex_m_p = float(snap.get("ExtractConcentration_Man", 0.0))
        ex_g_p = float(snap.get("ExtractConcentration_Gal", 0.0))
        ra_m_p = float(snap.get("RaffinateConcentration_Man", 0.0))
        ra_g_p = float(snap.get("RaffinateConcentration_Gal", 0.0))

        # Peek twin both mappings to auto-detect if needed
        ex_m_A, ex_g_A, ra_m_A, ra_g_A = peek_outlets_now(twin, 1, 3)  # normal map
        ex_m_B, ex_g_B, ra_m_B, ra_g_B = peek_outlets_now(twin, 3, 1)  # swapped map

        # Score maps by total abs error (Man+Gal)
        errA = abs(ex_m_A - ex_m_p) + abs(ex_g_A - ex_g_p) + abs(ra_m_A - ra_m_p) + abs(ra_g_A - ra_g_p)
        errB = abs(ex_m_B - ex_m_p) + abs(ex_g_B - ex_g_p) + abs(ra_m_B - ra_m_p) + abs(ra_g_B - ra_g_p)

        chosen = "A"
        if AUTO_MAP:
            _map_score_normal  += errA
            _map_score_swapped += errB
            _map_counter += 1
            if _map_counter >= _map_eval_every:
                if _map_score_swapped + 1e-15 < _map_score_normal * 0.5:
                    # swapped fits MUCH better → switch mapping
                    EX_ZONE, RA_ZONE = 3, 1
                    print("[sync] Auto-mapping switched: EX=zone3, RA=zone1", flush=True)
                elif _map_score_normal + 1e-15 < _map_score_swapped * 0.5:
                    EX_ZONE, RA_ZONE = 1, 3
                    print("[sync] Auto-mapping switched: EX=zone1, RA=zone3", flush=True)
                # reset scores
                _map_score_normal = _map_score_swapped = 0.0
                _map_counter = 0
            # choose current reading accordingly
            if EX_ZONE == 1:
                ex_m_t, ex_g_t, ra_m_t, ra_g_t = ex_m_A, ex_g_A, ra_m_A, ra_g_A
                chosen = "A"
            else:
                ex_m_t, ex_g_t, ra_m_t, ra_g_t = ex_m_B, ex_g_B, ra_m_B, ra_g_B
                chosen = "B"
        else:
            # fixed mapping
            ex_m_t, ex_g_t, ra_m_t, ra_g_t = peek_outlets_now(twin, EX_ZONE, RA_ZONE)
            chosen = "fixed"

        # monitoring: abs errors, buffers, CSV
        e_ex_m = abs(ex_m_t - ex_m_p); e_ex_g = abs(ex_g_t - ex_g_p)
        e_ra_m = abs(ra_m_t - ra_m_p); e_ra_g = abs(ra_g_t - ra_g_p)
        buf_ex_m.append(e_ex_m); buf_ex_g.append(e_ex_g)
        buf_ra_m.append(e_ra_m); buf_ra_g.append(e_ra_g)

        if LOG_CSV and _csv_writer is not None:
            _csv_writer.writerow([twin.timer,
                                  ex_m_t, ex_m_p, e_ex_m,
                                  ex_g_t, ex_g_p, e_ex_g,
                                  ra_m_t, ra_m_p, e_ra_m,
                                  ra_g_t, ra_g_p, e_ra_g,
                                  chosen, EX_ZONE, RA_ZONE])
            _csv_pending += 1
            if _csv_pending >= _CSV_FLUSH_EVERY:
                try:
                    _csv_fh.flush()
                except Exception:
                    pass
                _csv_pending = 0

        # periodic monitor summary (doesn't block)
        if now - _last_pv_print >= _MONITOR_MIN_PERIOD:
            d_ex_m = ex_m_t - ex_m_p; d_ex_g = ex_g_t - ex_g_p
            d_ra_m = ra_m_t - ra_m_p; d_ra_g = ra_g_t - ra_g_p
            print(f"[monitor] t={twin.timer:.2f}s map={chosen}[EX={EX_ZONE},RA={RA_ZONE}] "
                  f"EX twin({ex_m_t:.2e},{ex_g_t:.2e}) Δ=({d_ex_m:.2e},{d_ex_g:.2e}); "
                  f"RA twin({ra_m_t:.2e},{ra_g_t:.2e}) Δ=({d_ra_m:.2e},{d_ra_g:.2e})", flush=True)
            _last_pv_print = now

        # alerts
        cnt_ex_m = _maybe_alert("EX Mannose",   e_ex_m, cnt_ex_m)
        cnt_ex_g = _maybe_alert("EX Galactose", e_ex_g, cnt_ex_g)
        cnt_ra_m = _maybe_alert("RA Mannose",   e_ra_m, cnt_ra_m)
        cnt_ra_g = _maybe_alert("RA Galactose", e_ra_g, cnt_ra_g)

        time.sleep(0.05)

if __name__ == "__main__":
    main()
