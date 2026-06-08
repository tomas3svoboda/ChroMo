# -*- coding: utf-8 -*-
"""
MPC_application.py — GUI from backup1 + simple EKF status & integration
- Preserves the full layout and workflow of MPC_application_backup1.py
- Adds a lightweight outlet-bias EKF running in real time beside the plant
- EKF status (Bias, Residual) + buttons (Reset EKF, Zero bias)
- Optimizer starts from a frozen, measurement-corrected initial condition each cycle
"""

import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from opcua_client import SMB_OPCUAClient
from objective_function import ObjectiveWeights
from objective_function import smb_all_flows
from MPC_optimizer import MPCOptimizer, MPCOptimizerConfig
from smb_builder import build_smb
from state_estimator import estimate_next_template
# Profile-aware EKF
from ekf_profile_clean import SMBProfileEstimator as SMBProfileEstimatorPA, LinColumnAdapters



# ---------------------------------------------------------------------------
# EKF core (self-contained): outlet-bias EKF + wrapper + real-time estimator
# ---------------------------------------------------------------------------

class OutletBiasEKF:
    """
    4D EKF over outlet biases b = [Ex.Man, Ex.Gal, Ra.Man, Ra.Gal]
    y_meas = y_pred + b + v;  b_{k+1} = b_k + w
    Small, robust, and fast — good first step before profile-aware EKF.
    """
    def __init__(self, q=1e-3, r=5e-3, init_var=1e-2):
        self.b = np.zeros(4, float)
        self.P = np.eye(4) * float(init_var)
        self.Q = np.eye(4) * float(q)
        self.R = np.eye(4) * float(r)
        self._last_res = None

    def configure_noise(self, q=None, r=None):
        if q is not None: self.Q = np.eye(4) * float(q)
        if r is not None: self.R = np.eye(4) * float(r)

    def reset(self, init_var=1e-2):
        self.b[:] = 0.0
        self.P = np.eye(4) * float(init_var)
        self._last_res = None

    def zero(self):
        self.b[:] = 0.0

    def predict(self):
        # b^- = b, P^- = P + Q
        self.P = self.P + self.Q

    def update(self, y_meas, y_pred):
        H = np.eye(4)
        innov = y_meas - (y_pred + self.b)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.b = self.b + K @ innov
        self.P = (np.eye(4) - K @ H) @ self.P
        self._last_res = float(np.linalg.norm(innov))

    # snapshots
    def bias(self): return self.b.copy()
    def resnorm(self): return self._last_res


class OutletBiasWrapper:
    """
    Decorator for SMB station: adds EKF outlet bias to step_fast_outlets().
    All other attributes/methods are delegated unchanged.
    """
    def __init__(self, smb_station, bias_vec):
        self._smb = smb_station
        self._b = bias_vec.copy() if bias_vec is not None else None

    def deepCopy(self):
        return OutletBiasWrapper(self._smb.deepCopy(), (self._b.copy() if self._b is not None else None))

    def __getattr__(self, name):
        return getattr(self._smb, name)

    def step_fast_outlets(self):
        c_ex_m, c_ex_g, c_ra_m, c_ra_g = self._smb.step_fast_outlets()
        if self._b is None:
            return c_ex_m, c_ex_g, c_ra_m, c_ra_g
        return (c_ex_m + float(self._b[0]),
                c_ex_g + float(self._b[1]),
                c_ra_m + float(self._b[2]),
                c_ra_g + float(self._b[3]))


class SMBRealTimeEstimator(threading.Thread):
    """
    Real-time estimator thread:
      - advances a working SMB copy by dt
      - reads outlet concentrations from OPC UA
      - runs outlet-bias EKF update
    Exposes a thread-safe snapshot (SMB deepCopy + bias copy) for optimizer seeding.
    """
    def __init__(self, smb_template, opc_client, q=1e-3, r=5e-3):
        super().__init__(daemon=True)
        self._opc = opc_client
        self._dt = float(smb_template.settings["dt"])
        self._smb = smb_template.deepCopy()
        self._ekf = OutletBiasEKF(q=q, r=r)
        self._lock = threading.Lock()
        self._run = threading.Event(); self._run.set()
        self._last_wall = None

    # control
    def stop(self): self._run.clear()
    def configure(self, q=None, r=None): self._ekf.configure_noise(q, r)
    def reset(self): self._ekf.reset()
    def zero(self): self._ekf.zero()

    # snapshots for GUI/optimizer
    def snapshot_for_optimizer(self):
        with self._lock:
            return self._smb.deepCopy(), self._ekf.bias()
    def bias(self): return self._ekf.bias()
    def resnorm(self): return self._ekf.resnorm()
    def last_update_age(self):
        return None if self._last_wall is None else max(0.0, time.time() - self._last_wall)

    def run(self):
        use_fast = hasattr(self._smb, "step_fast_outlets")
        while self._run.is_set():
            # model predict by dt
            if use_fast:
                y_pred = np.array(self._smb.step_fast_outlets(), float)
            else:
                self._smb.step(1)
                y_pred = np.array(self._smb.step_fast_outlets(), float)

            # measurement from OPC UA
            try:
                s = self._opc.read_snapshot()
                y_meas = np.array([
                    s["ExtractConcentration_Man"], s["ExtractConcentration_Gal"],
                    s["RaffinateConcentration_Man"], s["RaffinateConcentration_Gal"]
                ], float)
                # EKF update
                self._ekf.predict()
                self._ekf.update(y_meas, y_pred)
                self._last_wall = time.time()
            except Exception:
                # missing data? continue predicting
                pass

            # pace with wall time (adjust if needed)
            time.sleep(max(0.0, self._dt * 0.5))


# ---------------------------------------------------------------------------
# GUI application (restored layout from backup1 + tiny EKF area)
# ---------------------------------------------------------------------------

class MPCApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SMB MPC")
        self.geometry("1180x680")

        # core state
        self.estimator = None
        self.running = False
        self.opc = None
        self.smb_template = None
        self.best = None
        self.mpc_thread = None
        self.stop_event = threading.Event()
        self._align_to_next = False  # controls whether we align seed to next period

        # ----- GUI state vars -----
        self.endpoint_var = tk.StringVar(value="opc.tcp://127.0.0.1:4840")
        self.E_var = tk.DoubleVar(value=32.0)
        self.horizon_var = tk.IntVar(value=4)

        self.dt_var = tk.DoubleVar(value=0.05)
        self.Nx_var = tk.IntVar(value=100)

        self.w_pME = tk.DoubleVar(value=1.0)
        self.w_pGR = tk.DoubleVar(value=1.0)
        self.w_yME = tk.DoubleVar(value=1.0)
        self.w_yGR = tk.DoubleVar(value=1.0)
        self.w_cons = tk.DoubleVar(value=0.2)

        self.method_var = tk.StringVar(value="powell")
        self.bF_lo = tk.DoubleVar(value=1.0);   self.bF_hi = tk.DoubleVar(value=100.0)
        self.bD_lo = tk.DoubleVar(value=3.0);   self.bD_hi = tk.DoubleVar(value=200.0)
        self.bQ_lo = tk.DoubleVar(value=2.0);   self.bQ_hi = tk.DoubleVar(value=150.0)

        self.maxiter_var = tk.IntVar(value=70)
        self.xatol_var = tk.DoubleVar(value=1e-3)
        self.fatol_var = tk.DoubleVar(value=1e-3)

        self.plateau_delta_var = tk.DoubleVar(value=0.05)
        self.plateau_window_var = tk.IntVar(value=15)

        self.safety_margin_var = tk.DoubleVar(value=0.5)
        self.timeout_var = tk.StringVar(value="")

        self.stepF_var = tk.DoubleVar(value=0.5)
        self.stepD_var = tk.DoubleVar(value=5.0)
        self.stepQ_var = tk.DoubleVar(value=3.0)

        # EKF status vars (simple)
        self.ekf_bias_txt = tk.StringVar(value="[0, 0, 0, 0]")
        self.ekf_res_txt  = tk.StringVar(value="-")

        # ----- Build GUI (backup1 layout) -----
        self._build_gui_backup1_layout()

        # start periodic snapshot poll
        self.after(500, self._poll_snapshot)
        self._update_button_states()

    # ---------------- GUI (backup1) ----------------
    def _build_gui_backup1_layout(self):
        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        for c in (0, 1, 2):
            frm.columnconfigure(c, weight=(0 if c < 2 else 1))
        frm.rowconfigure(0, weight=1)

        # Left column
        left = ttk.Frame(frm)
        left.grid(row=0, column=0, sticky="ns", padx=(0,10))
        ttk.Label(left, text="OPC UA endpoint").pack(anchor="w")
        ttk.Entry(left, textvariable=self.endpoint_var, width=28).pack(anchor="w", pady=(0,6))
        ttk.Label(left, text="Fixed Extract E [mL/h]").pack(anchor="w")
        ttk.Entry(left, textvariable=self.E_var, width=12).pack(anchor="w", pady=(0,6))
        ttk.Label(left, text="Horizon (periods)").pack(anchor="w")
        ttk.Spinbox(left, from_=1, to=12, textvariable=self.horizon_var, width=10).pack(anchor="w", pady=(0,6))

        ttk.Separator(left).pack(fill="x", pady=6)
        ttk.Label(left, text="Solver settings (edit only when stopped)").pack(anchor="w")
        ttk.Label(left, text="dt [s]").pack(anchor="w")
        self.dt_entry = ttk.Entry(left, textvariable=self.dt_var, width=10)
        self.dt_entry.pack(anchor="w")
        ttk.Label(left, text="Nx [-]").pack(anchor="w")
        self.Nx_entry = ttk.Entry(left, textvariable=self.Nx_var, width=10)
        self.Nx_entry.pack(anchor="w", pady=(0,6))

        ttk.Separator(left).pack(fill="x", pady=6)
        ttk.Label(left, text="Objective Weights").pack(anchor="w")
        def weight_row(parent, label, var):
            row = ttk.Frame(parent); row.pack(anchor="w", fill="x")
            ttk.Label(row, text=label, width=20).pack(side="left")
            ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        weight_row(left, "Purity Man @ Extract", self.w_pME)
        weight_row(left, "Purity Gal @ Raff", self.w_pGR)
        weight_row(left, "Yield Man @ Extract", self.w_yME)
        weight_row(left, "Yield Gal @ Raff", self.w_yGR)
        weight_row(left, "Eluent consumption", self.w_cons)

        # Middle column
        mid = ttk.Frame(frm)
        mid.grid(row=0, column=1, sticky="ns", padx=(0,10))
        ttk.Label(mid, text="Optimizer Settings").pack(anchor="w")

        mrow = ttk.Frame(mid); mrow.pack(anchor="w", fill="x")
        ttk.Label(mrow, text="Method", width=12).pack(side="left")
        ttk.Combobox(mrow, textvariable=self.method_var,
                     values=["powell", "nelder-mead", "bruteforce"], width=14, state="readonly").pack(side="left")

        bfr = ttk.LabelFrame(mid, text="Bounds")
        bfr.pack(anchor="w", fill="x", pady=(6,4))
        def bound_row(parent, name, lo_var, hi_var):
            row = ttk.Frame(parent); row.pack(anchor="w", fill="x", pady=1)
            ttk.Label(row, text=f"{name} min").pack(side="left")
            ttk.Entry(row, textvariable=lo_var, width=8).pack(side="left", padx=(2,10))
            ttk.Label(row, text=f"{name} max").pack(side="left")
            ttk.Entry(row, textvariable=hi_var, width=8).pack(side="left")
        bound_row(bfr, "F",  self.bF_lo, self.bF_hi)
        bound_row(bfr, "D",  self.bD_lo, self.bD_hi)
        bound_row(bfr, "Q4", self.bQ_lo, self.bQ_hi)

        ofr = ttk.LabelFrame(mid, text="Convergence")
        ofr.pack(anchor="w", fill="x", pady=(6,4))
        row = ttk.Frame(ofr); row.pack(anchor="w", fill="x")
        ttk.Label(row, text="maxiter", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.maxiter_var, width=12).pack(side="left")
        row = ttk.Frame(ofr); row.pack(anchor="w", fill="x")
        ttk.Label(row, text="xatol", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.xatol_var, width=12).pack(side="left")
        row = ttk.Frame(ofr); row.pack(anchor="w", fill="x")
        ttk.Label(row, text="fatol", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.fatol_var, width=12).pack(side="left")

        pfr = ttk.LabelFrame(mid, text="Early stop (plateau)")
        pfr.pack(anchor="w", fill="x", pady=(6,4))
        row = ttk.Frame(pfr); row.pack(anchor="w", fill="x")
        ttk.Label(row, text="ΔJ min", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.plateau_delta_var, width=12).pack(side="left")
        row = ttk.Frame(pfr); row.pack(anchor="w", fill="x")
        ttk.Label(row, text="window", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.plateau_window_var, width=12).pack(side="left")

        tfr = ttk.LabelFrame(mid, text="Time guarding")
        tfr.pack(anchor="w", fill="x", pady=(6,4))
        row = ttk.Frame(tfr); row.pack(anchor="w", fill="x")
        ttk.Label(row, text="safety margin [s]", width=18).pack(side="left")
        ttk.Entry(row, textvariable=self.safety_margin_var, width=10).pack(side="left")
        row = ttk.Frame(tfr); row.pack(anchor="w", fill="x")
        ttk.Label(row, text="hard timeout [s]", width=18).pack(side="left")
        ttk.Entry(row, textvariable=self.timeout_var, width=10).pack(side="left")
        ttk.Label(tfr, text="(leave blank for unlimited)").pack(anchor="w")

        nmfr = ttk.LabelFrame(mid, text="Nelder–Mead initial simplex")
        nmfr.pack(anchor="w", fill="x", pady=(6,8))
        row = ttk.Frame(nmfr); row.pack(anchor="w", fill="x")
        ttk.Label(row, text="step F", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.stepF_var, width=12).pack(side="left")
        row = ttk.Frame(nmfr); row.pack(anchor="w", fill="x")
        ttk.Label(row, text="step D", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.stepD_var, width=12).pack(side="left")
        row = ttk.Frame(nmfr); row.pack(anchor="w", fill="x")
        ttk.Label(row, text="step Q4", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.stepQ_var, width=12).pack(side="left")

        ttk.Separator(mid).pack(fill="x", pady=8)
        actions = ttk.LabelFrame(mid, text="Actions")
        actions.pack(anchor="w", fill="x", pady=(2,6))
        self.btn_connect  = ttk.Button(actions, text="Connect OPC", command=self.connect_opc); self.btn_connect.pack(fill="x", pady=2)
        self.btn_prestart = ttk.Button(actions, text="Pre-start Optimize + Apply", command=self.prestart_optimize); self.btn_prestart.pack(fill="x", pady=2)
        self.btn_start    = ttk.Button(actions, text="Start MPC", command=self.start_mpc); self.btn_start.pack(fill="x", pady=2)
        self.btn_stop     = ttk.Button(actions, text="Stop MPC", command=self.stop_mpc); self.btn_stop.pack(fill="x", pady=2)

        # Right column
        right = ttk.Frame(frm)
        right.grid(row=0, column=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)  # log area expands

        self.status_lbl = ttk.Label(right, text="Status: disconnected")
        self.status_lbl.grid(row=0, column=0, sticky="w")

        snap_fr = ttk.LabelFrame(right, text="Plant Snapshot")
        snap_fr.grid(row=1, column=0, sticky="ew", pady=(6,6))
        self.lab_simtime = ttk.Label(snap_fr, text="SimTime: -"); self.lab_simtime.pack(anchor="w")
        self.lab_switch  = ttk.Label(snap_fr, text="Time to switch: -"); self.lab_switch.pack(anchor="w")
        self.lab_mode    = ttk.Label(snap_fr, text="Mode: -"); self.lab_mode.pack(anchor="w")

        opt_fr = ttk.LabelFrame(right, text="Optimizer")
        opt_fr.grid(row=2, column=0, sticky="ew", pady=(6,6))
        self.lab_running = ttk.Label(opt_fr, text="Running: no"); self.lab_running.pack(anchor="w")
        self.lab_evals   = ttk.Label(opt_fr, text="Evaluations: 0"); self.lab_evals.pack(anchor="w")
        self.lab_elapsed = ttk.Label(opt_fr, text="Elapsed [s]: 0.0"); self.lab_elapsed.pack(anchor="w")
        self.lab_bestJ   = ttk.Label(opt_fr, text="Best cost J: -"); self.lab_bestJ.pack(anchor="w")
        self.lab_bestx   = ttk.Label(opt_fr, text="Best x: -"); self.lab_bestx.pack(anchor="w")
        self.pb = ttk.Progressbar(opt_fr, mode="indeterminate"); self.pb.pack(fill="x", pady=(6,0))
        self._pb_spinning = False

        # --- EKF status (simple) ---
        ekf_fr = ttk.LabelFrame(right, text="EKF Status")
        ekf_fr.grid(row=3, column=0, sticky="ew", pady=(0,6))
        row = ttk.Frame(ekf_fr); row.pack(anchor="w", fill="x", pady=(2,0))
        ttk.Label(row, text="Bias:").pack(side="left")
        ttk.Label(row, textvariable=self.ekf_bias_txt).pack(side="left", padx=(6,0))
        row = ttk.Frame(ekf_fr); row.pack(anchor="w", fill="x", pady=(2,2))
        ttk.Label(row, text="Residual:").pack(side="left")
        ttk.Label(row, textvariable=self.ekf_res_txt).pack(side="left", padx=(6,0))
        rowb = ttk.Frame(ekf_fr); rowb.pack(anchor="w", fill="x", pady=(2,2))
        ttk.Button(rowb, text="Reset EKF", command=self._on_reset_ekf).pack(side="left")
        ttk.Button(rowb, text="Zero bias", command=self._on_zero_bias).pack(side="left", padx=6)

        # Log
        self.log = tk.Text(right, height=16)
        self.log.grid(row=4, column=0, sticky="nsew", pady=(6,0))

    # ---------------- Helpers ----------------
    def _update_button_states(self):
        connected = self.opc is not None
        running = self.running
        self.btn_connect.config(state="normal")
        self.btn_prestart.config(state=("normal" if connected and not running else "disabled"))
        self.btn_start.config(state=("normal" if connected and not running else "disabled"))
        self.btn_stop.config(state=("normal" if running else "disabled"))

    def _gui_running_state(self, running: bool):
        self.running = running
        state = "disabled" if running else "normal"
        for w in (self.dt_entry, self.Nx_entry):
            w.config(state=state)
        self._update_button_states()

    def _append_log(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def _cfg_from_gui(self, for_prestart: bool) -> MPCOptimizerConfig:
        ht_raw = self.timeout_var.get().strip()
        hard_timeout = None if ht_raw == "" else float(ht_raw)
        return MPCOptimizerConfig(
            horizon_periods=self.horizon_var.get(),
            weights=ObjectiveWeights(
                purity_man_extract=self.w_pME.get(),
                purity_gal_raff=self.w_pGR.get(),
                yield_man_extract=self.w_yME.get(),
                yield_gal_raff=self.w_yGR.get(),
                eluent_consumption=self.w_cons.get(),
            ),
            bounds={'F': (self.bF_lo.get(), self.bF_hi.get()),
                    'D': (self.bD_lo.get(), self.bD_hi.get()),
                    'Q4': (self.bQ_lo.get(), self.bQ_hi.get())},
            maxiter=self.maxiter_var.get(),
            xatol=self.xatol_var.get(),
            fatol=self.fatol_var.get(),
            method=self.method_var.get().lower(),
            safety_margin_s=self.safety_margin_var.get(),
            hard_timeout_s=hard_timeout,
            plateau_delta=self.plateau_delta_var.get(),
            plateau_window=self.plateau_window_var.get(),
            step_F=self.stepF_var.get(),
            step_D=self.stepD_var.get(),
            step_Q=self.stepQ_var.get(),
        )

    # ---------------- OPC ----------------
    def connect_opc(self):
        try:
            self.opc = SMB_OPCUAClient(self.endpoint_var.get())
            self.opc.connect()
            self.status_lbl.config(text=f"Status: connected to {self.endpoint_var.get()}")
            self._append_log("[INFO] Connected to OPC UA")
        except Exception as e:
            messagebox.showerror("OPC Connect", str(e))
        finally:
            self._update_button_states()

    def _poll_snapshot(self):
        try:
            if self.opc:
                self.lab_simtime.config(text=f"SimTime: {self.opc.sim_time():.1f} s")
                # if your client has both time_to_switch (sim) and wall variant, keep the sim display here:
                self.lab_switch.config(text=f"Time to switch: {self.opc.time_to_switch():.1f} s")
                try:
                    mode = self.opc._nodes.get('Mode').get_value()
                    self.lab_mode.config(text=f"Mode: {'AUTO' if mode=='automatic' else 'MANUAL'}")
                except Exception:
                    self.lab_mode.config(text="Mode: ?")

                # --- EKF status live ---
                if self.estimator:
                    b = self.estimator.bias()
                    if b is not None:
                        self.ekf_bias_txt.set("[" + ", ".join(f"{v:.3f}" for v in b) + "]")
                    rn = self.estimator.resnorm()
                    if rn is not None:
                        self.ekf_res_txt.set(f"{rn:.4f}")
        except Exception:
            pass
        self.after(500, self._poll_snapshot)
        # --- Auto-start EKF when plant starts ---
        if self.opc and self.estimator is None:
            try:
                if self.opc.is_running():  # IsRunning on OPC UA
                    if self.smb_template is None:
                        # Build a model from current OPC UA setpoints (same recipe as start_mpc)
                        snap = self.opc.read_snapshot()
                        params = {
                            'dt': float(self.dt_var.get()),
                            'Nx': int(self.Nx_var.get()),
                            'switch_interval': float(self._current_switch_interval()),
                            'col_length': 320, 'col_diameter': 10, 'porosity': 0.376, 'dead_volume': 0.5,
                        }
                        components = [
                            {"name": "Man", "feed_concentration": 9, "henry_constant": 4.55, "delta": 54, "Di": 0.0007},
                            {"name": "Gal", "feed_concentration": 6, "henry_constant": 2.77, "delta": 84, "Di": 0.0007},
                        ]
                        flows_start = {
                            "feed":    float(snap["FeedFlow"]),
                            "eluent":  float(snap["EluentFlow"]),
                            "recycle": float(snap["RecycleFlow"]),
                            "extract": float(self.E_var.get()),
                        }
                        self.smb_template = build_smb(params, components, flows_start)
                    # Profile-aware EKF nad outlet profily (Z1/Z3 poslední kolona)
                    adapters = LinColumnAdapters(
                        zone1_last_col_getter=lambda smb: smb.zones[1][-1],
                        zone3_last_col_getter=lambda smb: smb.zones[3][-1],
                    )
                    self.estimator = SMBProfileEstimatorPA(
                        self.smb_template,
                        self.opc,
                        adapters,
                        measurement_model="lag",      # ← force lag
                        tau=(0.5, 0.5, 0.5, 0.5),     # ← 0.5 s on all channels
                        dt_model=float(self.dt_var.get()),
                        ekf_period_s=15.0,
                        R=(2e-4, 2e-4, 4e-4, 4e-4),
                    )
                    self.estimator.start()
                    self._append_log("[EKF] Auto-started (IsRunning=True)")
            except Exception:
                pass

    # ---------------- Optimizer callbacks ----------------
    def _on_status(self, st):
        self.lab_running.config(text=f"Running: {'yes' if st.running else 'no'}")
        self.lab_evals.config(text=f"Evaluations: {st.nevals}")
        self.lab_elapsed.config(text=f"Elapsed [s]: {st.elapsed_s:.1f}")
        if st.best_cost is not None:
            self.lab_bestJ.config(text=f"Best cost J: {st.best_cost:.3f}")
        if st.best_x:
            F1, D1, Q41 = st.best_x[:3]
            self.lab_bestx.config(text=f"Best x: F={F1:.2f}, D={D1:.2f}, Q4={Q41:.2f}")
        if st.running and not self._pb_spinning:
            self.pb.start(50); self._pb_spinning = True
        elif not st.running and self._pb_spinning:
            self.pb.stop(); self._pb_spinning = False

    def _on_done(self, st):
        self._on_status(st)
        msg = getattr(st, "message", "") or ""
        if msg.startswith("error"):
            self._append_log(f"[ERROR] {msg}")
        else:
            if st.best_cost is not None:
                self._append_log(f"[MPC] Done: J={st.best_cost:.3f}, evals={st.nevals}")
            else:
                self._append_log(f"[MPC] Done: No valid solution, evals={st.nevals}")

    def _estimator_hook(self, smb_template):
        """
        Seed the optimizer with a live, measurement-corrected state.
        If self._align_to_next is True, first advance that state to the end
        of the current period using current OPC setpoints, so the horizon
        starts exactly at the next switch.
        """
        try:
            if self.estimator is not None:
                smb_copy, bias = self.estimator.snapshot_for_optimizer()

                if self._align_to_next and self.opc is not None:
                    # Advance to the end of the *current* period
                    rem = max(0.0, float(self.opc.time_to_switch_wall()))   # wall seconds to switch
                    dt = float(smb_copy.settings["dt"])
                    steps = int(max(0, round(rem / max(dt, 1e-9))))

                    # Use *current* setpoints for the remaining partial period
                    snap = self.opc.read_snapshot()
                    flows = smb_all_flows(float(snap["FeedFlow"]),
                                          float(snap["EluentFlow"]),
                                          float(self.E_var.get()),
                                          float(snap["RecycleFlow"]))
                    smb_copy.setFlowRateZone(1, flows["zone1"])
                    smb_copy.setFlowRateZone(2, flows["zone2"])
                    smb_copy.setFlowRateZone(3, flows["zone3"])
                    smb_copy.setFlowRateZone(4, flows["zone4"])

                    for _ in range(steps):
                        smb_copy.step(1)

                return OutletBiasWrapper(smb_copy, bias)
        except Exception:
            pass
        return smb_template

    # ---------------- Pre-start ----------------
    def prestart_optimize(self):
        if self.running:
            messagebox.showinfo("Pre-start", "Optimizer already running."); return
        if not self.opc:
            messagebox.showwarning("Pre-start", "Connect OPC first."); return
        try:
            params = {
                'dt': float(self.dt_var.get()),
                'Nx': int(self.Nx_var.get()),
                'switch_interval': float(self._current_switch_interval()),
                'col_length': 320, 'col_diameter': 10, 'porosity': 0.376, 'dead_volume': 0.5
            }
            components = [
                {"name": "Man", "feed_concentration": 9, "henry_constant": 4.55, "delta": 54, "Di": 0.0007},
                {"name": "Gal", "feed_concentration": 6, "henry_constant": 2.77, "delta": 84, "Di": 0.0007},
            ]
            snap = self.opc.read_snapshot()
            x0 = [snap["FeedFlow"], snap["EluentFlow"], snap["RecycleFlow"]]
            flows_start = {"feed": x0[0], "eluent": x0[1], "recycle": x0[2], "extract": self.E_var.get()}

            self._append_log(
                "[MPC] Initial guess from OPC UA — "
                f"F={x0[0]:.2f}, D={x0[1]:.2f}, Q4={x0[2]:.2f}, "
                f"E={self.E_var.get():.2f}, t_switch={params['switch_interval']:.1f}s; "
                f"horizon={self.horizon_var.get()}p, dt={params['dt']}, Nx={params['Nx']}"
            )

            self.smb_template = build_smb(params, components, flows_start)
            cfg = self._cfg_from_gui(for_prestart=True)

            self.lab_running.config(text="Running: yes")
            self._prestart_opt = MPCOptimizer(
                fixed_extract=self.E_var.get(),
                on_status=self._ui_on_status,
                on_done=self._ui_on_prestart_done,
            )
            self._gui_running_state(True)
            # generous time budget for pre-start (not synced to a real switch yet)
            self._prestart_opt.start_async(self.smb_template, x0, cfg, get_time_remaining_s=lambda: 1e9)
            self._append_log("[MPC] Pre-start optimization launched...")
        except Exception as e:
            messagebox.showerror("Pre-start", str(e))
            self._gui_running_state(False)

    # ---------------- MPC rolling loop ----------------
    def start_mpc(self):
        if self.running:
            return
        if not self.opc:
            messagebox.showwarning("Start", "Connect OPC first.")
            return

        # If we don't have a model yet, build it from current OPC UA params
        try:
            if self.smb_template is None:
                snap = self.opc.read_snapshot()
                params = {
                    'dt': float(self.dt_var.get()),
                    'Nx': int(self.Nx_var.get()),
                    'switch_interval': float(self._current_switch_interval()),
                    'col_length': 320, 'col_diameter': 10, 'porosity': 0.376, 'dead_volume': 0.5,
                }
                components = [
                    {"name": "Man", "feed_concentration": 9, "henry_constant": 4.55, "delta": 54, "Di": 0.0007},
                    {"name": "Gal", "feed_concentration": 6, "henry_constant": 2.77, "delta": 84, "Di": 0.0007},
                ]
                flows_start = {
                    "feed":    float(snap["FeedFlow"]),
                    "eluent":  float(snap["EluentFlow"]),
                    "recycle": float(snap["RecycleFlow"]),
                    "extract": float(self.E_var.get()),
                }
                self.smb_template = build_smb(params, components, flows_start)
                self._append_log("[MPC] Bootstrapped model from current OPC UA setpoints.")
        except Exception as e:
            messagebox.showerror("Start", f"Failed to bootstrap SMB template: {e}")
            return

        # Start the real-time estimator if not running (fix: no nonexistent 'use_bias' arg)
        if self.estimator is None:
            adapters = LinColumnAdapters(
                zone1_last_col_getter=lambda smb: smb.zones[1][-1],
                zone3_last_col_getter=lambda smb: smb.zones[3][-1],
            )
            self.estimator = SMBProfileEstimatorPA(
                self.smb_template, self.opc, adapters,
                dt_model=float(self.dt_var.get()),
                ekf_period_s=15.0,
            )
            self.estimator.start()

        # Launch the rolling MPC loop
        self._gui_running_state(True)
        self.stop_event.clear()
        self.mpc_thread = threading.Thread(target=self._mpc_loop, daemon=True)
        self.mpc_thread.start()

    def stop_mpc(self):
        if self.estimator:
            try: self.estimator.stop()
            except Exception: pass
            self.estimator = None
        self.stop_event.set()
        self._gui_running_state(False)
        self._append_log("[MPC] Stop requested.")

    def _mpc_loop(self):
        try:
            last_best = None
            snap = self.opc.read_snapshot()
            x_last = [snap["FeedFlow"], snap["EluentFlow"], snap["RecycleFlow"]]

            # --- Kickoff optimization if we started mid-period ---
            remaining = float(self.opc.time_to_switch_wall())
            if remaining > 0.7 and not self.stop_event.is_set():  # > ~0.7 s left? start now
                cfg = self._cfg_from_gui(for_prestart=False)

                opt = MPCOptimizer(
                    fixed_extract=self.E_var.get(),
                    state_estimator_hook=self._estimator_hook,   # will align to next period
                    on_status=self._ui_on_status,
                    on_done=self._ui_on_cycle_done,
                )

                self._align_to_next = True
                self._safe_log(f"[MPC] Kickoff: optimizing for next period (remaining {remaining:.2f}s)")
                opt.start_async(self.smb_template, x_last, cfg, self.opc.time_to_switch_wall)

                # Let it run until close to the switch (leave GUI-configured margin for writing)
                apply_margin = float(self.safety_margin_var.get())
                while (opt.is_running()
                       and not self.stop_event.is_set()
                       and self.opc.time_to_switch_wall() > apply_margin):
                    time.sleep(0.05)

                last_best = opt.status()
                self._align_to_next = False
            else:
                last_best = None

            while not self.stop_event.is_set():
                ok = self.opc.wait_for_next_switch(threshold=0.5, poll_s=0.1, timeout_s=3600)
                if not ok or self.stop_event.is_set():
                    break

                if last_best and last_best.best_x:
                    x_last = last_best.best_x[:]

                cfg = self._cfg_from_gui(for_prestart=False)

                opt = MPCOptimizer(
                    fixed_extract=self.E_var.get(),
                    state_estimator_hook=self._estimator_hook,  # <-- seed from EKF
                    on_status=self._ui_on_status,
                    on_done=self._ui_on_cycle_done,
                )
                self.after(0, lambda: self.lab_running.config(text="Running: yes"))
                self._safe_log(f"[MPC] Cycle warm start: F={x_last[0]:.2f}, D={x_last[1]:.2f}, Q4={x_last[2]:.2f}")
                opt.start_async(self.smb_template, x_last, cfg, self.opc.time_to_switch_wall)

                while opt.is_running() and not self.stop_event.is_set():
                    time.sleep(0.05)

                last_best = opt.status()
                if self.stop_event.is_set():
                    break
                if not last_best.best_x:
                    self._safe_log("[MPC] No solution this cycle; skipping apply.")
                    continue

                F1, D1, Q41 = last_best.best_x[:3]
                try:
                    # Apply very close to the switch using GUI-configured margin
                    apply_margin = float(self.safety_margin_var.get())
                    while self.opc.time_to_switch_wall() > apply_margin and not self.stop_event.is_set():
                        time.sleep(0.02)

                    # Late-edge guard: if we're inside a “don’t write” zone, skip this period
                    min_guard = 0.08  # 80 ms “don’t write” zone to avoid mid-period writes
                    if self.opc.time_to_switch_wall() <= min_guard:
                        self._safe_log("[GUARD] Edge missed; deferring apply to next period.")
                        continue

                    # Write setpoints atomically just before the boundary
                    self.opc.write_setpoints(feed=F1, eluent=D1, recycle=Q41,
                                             switch_interval=self._current_switch_interval(),
                                             extract=self.E_var.get())
                    self._bump_next_revision()

                    # Verify the boundary actually occurred (optional but recommended)
                    edge_ok = self.opc.wait_for_next_switch(threshold=0.5, poll_s=0.05, timeout_s=5.0)
                    if edge_ok:
                        self._safe_log(f"[APPLY] Confirmed at boundary; new setpoints active. F={F1:.2f}, D={D1:.2f}, Q4={Q41:.2f}")
                    else:
                        self._safe_log("[WARN] Edge confirmation timed out; monitor next cycle.")
                except Exception as e:
                    self._safe_log(f"[ERROR] Apply failed: {e}")
                    continue

                # advance internal template by one period with chosen controls
                self.smb_template = estimate_next_template(
                    self.smb_template, last_best.best_x, self.E_var.get(), periods_to_advance=1
                )
        except Exception as e:
            self._safe_log(f"[ERROR] MPC loop: {e}")
        finally:
            self.after(0, lambda: self._gui_running_state(False))
            self._safe_log("[MPC] Loop ended.")

    # ---------------- Utils / Callbacks ----------------
    def _bump_next_revision(self):
        try:
            node = self.opc._nodes.get('NextRevision')
            if node is None: return
            current = int(node.get_value())
            node.set_value(current + 1)
        except Exception:
            pass

    def _current_switch_interval(self) -> float:
        try:
            return float(self.opc._nodes["SwitchInterval"].get_value())
        except Exception:
            return 1894.0

    def _ui_on_status(self, st):
        self.after(0, lambda: self._on_status(st))

    def _ui_on_prestart_done(self, st):
        def finish():
            self._on_done(st)
            if st.best_x:
                F1, D1, Q41 = st.best_x[:3]
                try:
                    self.opc.set_mode("automatic")
                    self.opc.write_setpoints(feed=F1, eluent=D1, recycle=Q41,
                                             switch_interval=self._current_switch_interval(),
                                             extract=self.E_var.get())
                    self._bump_next_revision()
                    self._append_log(f"[APPLY] Pre-start applied: F={F1:.2f}, D={D1:.2f}, Q4={Q41:.2f}")
                    self.smb_template = estimate_next_template(
                        self.smb_template, st.best_x, self.E_var.get(), periods_to_advance=1
                    )
                finally:
                    self._gui_running_state(False)
            else:
                self._append_log("[MPC] Pre-start returned no solution.")
                self._gui_running_state(False)
        self.after(0, finish)

    def _ui_on_cycle_done(self, st):
        def finish():
            self._on_status(st)
            if st.best_cost is not None:
                self._append_log(f"[MPC] Cycle done: J={st.best_cost:.3f}, evals={st.nevals}")
            else:
                self._append_log(f"[MPC] Cycle done: No valid solution, evals={st.nevals}")
        self.after(0, finish)

    def _safe_log(self, msg: str):
        self.after(0, lambda: self._append_log(msg))

    def _on_status(self, st):
        self.lab_running.config(text=f"Running: {'yes' if st.running else 'no'}")
        self.lab_evals.config(text=f"Evaluations: {st.nevals}")
        self.lab_elapsed.config(text=f"Elapsed [s]: {st.elapsed_s:.1f}")
        if st.best_cost is not None:
            self.lab_bestJ.config(text=f"Best cost J: {st.best_cost:.3f}")
        if st.best_x:
            F1, D1, Q41 = st.best_x[:3]
            self.lab_bestx.config(text=f"Best x: F={F1:.2f}, D={D1:.2f}, Q4={Q41:.2f}")
        if st.running and not self._pb_spinning:
            self.pb.start(50); self._pb_spinning = True
        elif not st.running and self._pb_spinning:
            self.pb.stop(); self._pb_spinning = False

    def _on_done(self, st):
        self._on_status(st)
        msg = getattr(st, "message", "") or ""
        if msg.startswith("error"):
            self._append_log(f"[ERROR] {msg}")
        else:
            if st.best_cost is not None:
                self._append_log(f"[MPC] Done: J={st.best_cost:.3f}, evals={st.nevals}")
            else:
                self._append_log(f"[MPC] Done: No valid solution, evals={st.nevals}")

    def _on_reset_ekf(self):
        if self.estimator:
            self.estimator.reset()
            self._append_log("[EKF] reset requested")

    def _on_zero_bias(self):
        if self.estimator:
            self.estimator.zero()
            self.ekf_bias_txt.set("[0, 0, 0, 0]")
            self._append_log("[EKF] bias set to zero")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = MPCApp()
    app.mainloop()
