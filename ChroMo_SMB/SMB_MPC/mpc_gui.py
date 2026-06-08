# mpc_gui.py — Tkinter GUI for SI-synchronized SMB MPC
# ====================================================
# See docstring below for usage.

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import asdict
from typing import Dict, Tuple, Optional

from mpc_orchestrator import MPCOrchestrator, Bounds, Weights, Policy
from mpc_optimizer import NMConfig

_PAD = 6
_NUM_WIDTH = 9


def _fmt_float(x: Optional[float], digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}g}"
    except Exception:
        return "—"


def _fmt_int(x: Optional[int]) -> str:
    try:
        return f"{int(x)}"
    except Exception:
        return "—"


class MPCGuiApp:
    """
    Usage
    -----
    from opcua_client import SMB_OPCUAClient
    from smb_engine import SMBTwinEngine

    opc = SMB_OPCUAClient(endpoint="opc.tcp://127.0.0.1:4840")
    eng = SMBTwinEngine(opc_client=opc, apply_mode="boundary")
    orch = MPCOrchestrator(engine=eng, opc=opc)

    from mpc_gui import MPCGuiApp
    MPCGuiApp(orch).run()
    """

    def __init__(self, orchestrator: MPCOrchestrator) -> None:
        self.orch = orchestrator

        self.root = tk.Tk()
        self.root.title("SMB MPC Controller")
        self.root.geometry("1200x820")

        self._build_ui()
        self._prefill_from_orch()

        # subscribe for live updates
        self.orch.subscribe(self._on_status)

        # periodic UI keep-alive
        self._last_ui_tick = time.time()
        self._tick()

    # ---------------- UI construction ----------------

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(4, weight=1)

        self._build_header()
        self._build_optimizer_panel()
        self._build_controls_panel()
        self._build_state_panel()
        self._build_log_panel()

    def _build_header(self) -> None:
        frm = ttk.Frame(self.root, padding=_PAD)
        frm.grid(row=0, column=0, sticky="ew")
        for c in (0, 1, 2, 3, 4, 5, 6, 7):
            frm.columnconfigure(c, weight=0)
        frm.columnconfigure(8, weight=1)

        self.btn_run = ttk.Button(frm, text="RUN", command=self._on_run_clicked)
        self.btn_stop = ttk.Button(frm, text="STOP", command=self._on_stop_clicked)
        self.btn_run.grid(row=0, column=0, padx=_PAD)
        self.btn_stop.grid(row=0, column=1, padx=_PAD)

        self.lbl_opc = ttk.Label(frm, text="OPC: —")
        self.lbl_mode = ttk.Label(frm, text="Mode: —")
        self.lbl_running = ttk.Label(frm, text="Plant: —")
        self.lbl_tts = ttk.Label(frm, text="T→switch: — s")

        self.lbl_opc.grid(row=0, column=2, padx=_PAD)
        self.lbl_mode.grid(row=0, column=3, padx=_PAD)
        self.lbl_running.grid(row=0, column=4, padx=_PAD)
        self.lbl_tts.grid(row=0, column=5, padx=_PAD)

        # --- MPC eval dt only ---
        dtf = ttk.LabelFrame(frm, text="MPC eval dt", padding=_PAD)
        dtf.grid(row=0, column=7, padx=_PAD, sticky="e")

        ttk.Label(dtf, text="dt [s]").grid(row=0, column=0, sticky="e")
        self.ent_dt_eval = ttk.Entry(dtf, width=_NUM_WIDTH)
        self.ent_dt_eval.grid(row=0, column=1, padx=(4, 8))

        self.btn_apply_dt = ttk.Button(dtf, text="Apply", command=self._apply_dt_clicked)
        self.btn_apply_dt.grid(row=0, column=2, padx=4)

        self.lbl_sf = ttk.Label(frm, text="SF: —")
        self.lbl_snap = ttk.Label(frm, text="Snapshot: —")
        self.lbl_sf.grid(row=0, column=6, padx=_PAD)
        self.lbl_snap.grid(row=0, column=8, padx=_PAD, sticky="e")

        self.lbl_budget = ttk.Label(frm, text="Budget: — s")
        self.lbl_budget.grid(row=0, column=9, padx=_PAD)

    def _build_optimizer_panel(self) -> None:
        frm = ttk.LabelFrame(self.root, text="Optimizer Status", padding=_PAD)
        frm.grid(row=1, column=0, sticky="ew", padx=_PAD)
        for c in range(8):
            frm.columnconfigure(c, weight=0)
        frm.columnconfigure(7, weight=1)

        self.lbl_solving = ttk.Label(frm, text="Idle")
        self.lbl_iter = ttk.Label(frm, text="Iter: —")
        self.lbl_elapsed = ttk.Label(frm, text="Elapsed [s]: —")
        self.lbl_bestJ = ttk.Label(frm, text="Best J: —")
        self.lbl_baseJ = ttk.Label(frm, text="Baseline J: —")

        self.lbl_solving.grid(row=0, column=0, padx=_PAD)
        self.lbl_iter.grid(row=0, column=1, padx=_PAD)
        self.lbl_elapsed.grid(row=0, column=2, padx=_PAD)
        self.lbl_bestJ.grid(row=0, column=3, padx=_PAD)
        self.lbl_baseJ.grid(row=0, column=4, padx=_PAD)

        ttk.Label(frm, text="Best x (Q1,Q2,Q4,SI):").grid(row=1, column=0, padx=_PAD, sticky="e")
        self.lbl_bestx = ttk.Label(frm, text="—")
        self.lbl_bestx.grid(row=1, column=1, columnspan=3, sticky="w")

        # KPIs (from best_metrics)
        self.lbl_pur = ttk.Label(frm, text="Pur_ex / Pur_ra: — / —")
        self.lbl_dil = ttk.Label(frm, text="Dil_ex / Dil_ra: — / —")
        self.lbl_pur.grid(row=2, column=0, columnspan=3, padx=_PAD, sticky="w")
        self.lbl_dil.grid(row=2, column=3, columnspan=3, padx=_PAD, sticky="w")

    def _build_controls_panel(self) -> None:
        frm = ttk.LabelFrame(self.root, text="Controller Settings", padding=_PAD)
        frm.grid(row=2, column=0, sticky="ew", padx=_PAD)
        for c in range(20):
            frm.columnconfigure(c, weight=0)

        # Policy
        ttk.Label(frm, text="Horizon [SI]").grid(row=0, column=0, sticky="e")
        self.ent_hzn = ttk.Entry(frm, width=_NUM_WIDTH)
        self.ent_hzn.grid(row=0, column=1, padx=(4, 12))

        ttk.Label(frm, text="Guard [s]").grid(row=0, column=2, sticky="e")
        self.ent_guard = ttk.Entry(frm, width=_NUM_WIDTH)
        self.ent_guard.grid(row=0, column=3, padx=(4, 12))

        ttk.Label(frm, text="ε (min ΔJ)").grid(row=0, column=4, sticky="e")
        self.ent_eps = ttk.Entry(frm, width=_NUM_WIDTH)
        self.ent_eps.grid(row=0, column=5, padx=(4, 12))

        # Weights
        wrow = 1
        ttk.Label(frm, text="w_dil_ex").grid(row=wrow, column=0, sticky="e")
        ttk.Label(frm, text="w_dil_ra").grid(row=wrow, column=2, sticky="e")
        ttk.Label(frm, text="w_pur_ex").grid(row=wrow, column=4, sticky="e")
        ttk.Label(frm, text="w_pur_ra").grid(row=wrow, column=6, sticky="e")
        self.ent_w_dil_ex = ttk.Entry(frm, width=_NUM_WIDTH); self.ent_w_dil_ex.grid(row=wrow, column=1, padx=(4, 12))
        self.ent_w_dil_ra = ttk.Entry(frm, width=_NUM_WIDTH); self.ent_w_dil_ra.grid(row=wrow, column=3, padx=(4, 12))
        self.ent_w_pur_ex = ttk.Entry(frm, width=_NUM_WIDTH); self.ent_w_pur_ex.grid(row=wrow, column=5, padx=(4, 12))
        self.ent_w_pur_ra = ttk.Entry(frm, width=_NUM_WIDTH); self.ent_w_pur_ra.grid(row=wrow, column=7, padx=(4, 12))

        # Bounds
        brow = 2
        ttk.Label(frm, text="Bounds Q1 [min,max]").grid(row=brow, column=0, sticky="e")
        self.ent_b_q1 = ttk.Entry(frm, width=2 * _NUM_WIDTH); self.ent_b_q1.grid(row=brow, column=1, padx=(4, 12))
        ttk.Label(frm, text="Q2").grid(row=brow, column=2, sticky="e")
        self.ent_b_q2 = ttk.Entry(frm, width=2 * _NUM_WIDTH); self.ent_b_q2.grid(row=brow, column=3, padx=(4, 12))
        ttk.Label(frm, text="Q4").grid(row=brow, column=4, sticky="e")
        self.ent_b_q4 = ttk.Entry(frm, width=2 * _NUM_WIDTH); self.ent_b_q4.grid(row=brow, column=5, padx=(4, 12))
        ttk.Label(frm, text="SI").grid(row=brow, column=6, sticky="e")
        self.ent_b_si = ttk.Entry(frm, width=2 * _NUM_WIDTH); self.ent_b_si.grid(row=brow, column=7, padx=(4, 12))

        # Nelder–Mead config
        nrow = 3
        ttk.Label(frm, text="NM maxiter").grid(row=nrow, column=0, sticky="e")
        self.ent_nm_maxiter = ttk.Entry(frm, width=_NUM_WIDTH); self.ent_nm_maxiter.grid(row=nrow, column=1, padx=(4, 12))
        ttk.Label(frm, text="xatol").grid(row=nrow, column=2, sticky="e")
        self.ent_nm_xatol = ttk.Entry(frm, width=_NUM_WIDTH); self.ent_nm_xatol.grid(row=nrow, column=3, padx=(4, 12))
        ttk.Label(frm, text="fatol").grid(row=nrow, column=4, sticky="e")
        self.ent_nm_fatol = ttk.Entry(frm, width=_NUM_WIDTH); self.ent_nm_fatol.grid(row=nrow, column=5, padx=(4, 12))
        ttk.Label(frm, text="simplex_rel").grid(row=nrow, column=6, sticky="e")
        self.ent_nm_simplex = ttk.Entry(frm, width=_NUM_WIDTH); self.ent_nm_simplex.grid(row=nrow, column=7, padx=(4, 12))

        self.btn_apply_settings = ttk.Button(frm, text="Apply Settings", command=self._apply_settings_clicked)
        self.btn_apply_settings.grid(row=0, column=9, rowspan=4, padx=12)

    def _build_state_panel(self) -> None:
        frm = ttk.LabelFrame(self.root, text="Plant & Setpoints", padding=_PAD)
        frm.grid(row=3, column=0, sticky="ew", padx=_PAD)
        for c in range(6):
            frm.columnconfigure(c, weight=0)
        frm.columnconfigure(5, weight=1)

        ttk.Label(frm, text="ACTIVE F:").grid(row=0, column=0, sticky="e")
        ttk.Label(frm, text="ACTIVE Q1:").grid(row=1, column=0, sticky="e")
        ttk.Label(frm, text="ACTIVE Q2:").grid(row=2, column=0, sticky="e")
        ttk.Label(frm, text="ACTIVE Q4:").grid(row=3, column=0, sticky="e")
        ttk.Label(frm, text="ACTIVE SI:").grid(row=4, column=0, sticky="e")

        ttk.Label(frm, text="NEXT F:").grid(row=0, column=3, sticky="e")
        ttk.Label(frm, text="NEXT Q1:").grid(row=1, column=3, sticky="e")
        ttk.Label(frm, text="NEXT Q2:").grid(row=2, column=3, sticky="e")
        ttk.Label(frm, text="NEXT Q4:").grid(row=3, column=3, sticky="e")
        ttk.Label(frm, text="NEXT SI:").grid(row=4, column=3, sticky="e")

        self.lbl_act_F = ttk.Label(frm, text="—"); self.lbl_act_F.grid(row=0, column=1, sticky="w")
        self.lbl_act_Q1 = ttk.Label(frm, text="—"); self.lbl_act_Q1.grid(row=1, column=1, sticky="w")
        self.lbl_act_Q2 = ttk.Label(frm, text="—"); self.lbl_act_Q2.grid(row=2, column=1, sticky="w")
        self.lbl_act_Q4 = ttk.Label(frm, text="—"); self.lbl_act_Q4.grid(row=3, column=1, sticky="w")
        self.lbl_act_SI = ttk.Label(frm, text="—"); self.lbl_act_SI.grid(row=4, column=1, sticky="w")

        self.lbl_nxt_F = ttk.Label(frm, text="—"); self.lbl_nxt_F.grid(row=0, column=4, sticky="w")
        self.lbl_nxt_Q1 = ttk.Label(frm, text="—"); self.lbl_nxt_Q1.grid(row=1, column=4, sticky="w")
        self.lbl_nxt_Q2 = ttk.Label(frm, text="—"); self.lbl_nxt_Q2.grid(row=2, column=4, sticky="w")
        self.lbl_nxt_Q4 = ttk.Label(frm, text="—"); self.lbl_nxt_Q4.grid(row=3, column=4, sticky="w")
        self.lbl_nxt_SI = ttk.Label(frm, text="—"); self.lbl_nxt_SI.grid(row=4, column=4, sticky="w")

    def _build_log_panel(self) -> None:
        frm = ttk.LabelFrame(self.root, text="Event Log", padding=_PAD)
        frm.grid(row=4, column=0, sticky="nsew", padx=_PAD, pady=(0, _PAD))
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(0, weight=1)

        self.txt_log = tk.Text(frm, height=8, wrap="word")
        self.txt_log.grid(row=0, column=0, sticky="nsew")
        self.txt_log.config(state="disabled")

    # ---------------- Prefill ----------------

    def _prefill_from_orch(self) -> None:
        # policy
        p: Policy = getattr(self.orch, "policy", Policy())
        self.ent_hzn.insert(0, str(p.horizon_si))
        self.ent_guard.insert(0, str(p.guard_time_s))
        self.ent_eps.insert(0, str(getattr(p, "improvement_epsilon", 0.0)))

        # weights
        w: Weights = getattr(self.orch, "weights", Weights())
        self.ent_w_dil_ex.insert(0, str(w.w_dil_ex))
        self.ent_w_dil_ra.insert(0, str(w.w_dil_ra))
        self.ent_w_pur_ex.insert(0, str(w.w_pur_ex))
        self.ent_w_pur_ra.insert(0, str(w.w_pur_ra))

        # bounds
        b: Bounds = getattr(self.orch, "bounds", Bounds((10, 400), (10, 400), (10, 400), (100, 10800)))
        self.ent_b_q1.insert(0, f"{b.Q1[0]},{b.Q1[1]}")
        self.ent_b_q2.insert(0, f"{b.Q2[0]},{b.Q2[1]}")
        self.ent_b_q4.insert(0, f"{b.Q4[0]},{b.Q4[1]}")
        self.ent_b_si.insert(0, f"{b.SI[0]},{b.SI[1]}")

        # NM config
        cfg: NMConfig = getattr(self.orch, "nm_cfg", NMConfig())
        self.ent_nm_maxiter.insert(0, str(cfg.maxiter))
        self.ent_nm_xatol.insert(0, str(cfg.xatol))
        self.ent_nm_fatol.insert(0, str(cfg.fatol))
        self.ent_nm_simplex.insert(0, str(cfg.simplex_rel_size))

        # dt default (eval only)
        self.ent_dt_eval.insert(0, str(getattr(self.orch, "eval_dt", 5.0)))

    # ---------------- Button handlers ----------------

    def _on_run_clicked(self) -> None:
        try:
            self.orch.start()
        except Exception as e:
            messagebox.showerror("RUN failed", str(e))

    def _on_stop_clicked(self) -> None:
        try:
            self.orch.stop()
        except Exception as e:
            messagebox.showerror("STOP failed", str(e))

    def _apply_settings_clicked(self) -> None:
        try:
            # Policy
            p = Policy(
                horizon_si=int(float(self.ent_hzn.get())),
                guard_time_s=float(self.ent_guard.get()),
                improvement_epsilon=float(self.ent_eps.get() or 0.0),
            )
            self.orch.set_policy(p)

            # Weights
            w = Weights(
                w_dil_ex=float(self.ent_w_dil_ex.get()),
                w_dil_ra=float(self.ent_w_dil_ra.get()),
                w_pur_ex=float(self.ent_w_pur_ex.get()),
                w_pur_ra=float(self.ent_w_pur_ra.get()),
            )
            self.orch.set_weights(w)

            # Bounds
            def _pair(s: str) -> Tuple[float, float]:
                a, b = [float(x.strip()) for x in s.split(",")]
                return (a, b)

            b = Bounds(
                Q1=_pair(self.ent_b_q1.get()),
                Q2=_pair(self.ent_b_q2.get()),
                Q4=_pair(self.ent_b_q4.get()),
                SI=_pair(self.ent_b_si.get()),
            )
            self.orch.set_bounds(b)

            # NM config
            cfg = NMConfig(
                maxiter=int(float(self.ent_nm_maxiter.get())),
                xatol=float(self.ent_nm_xatol.get()),
                fatol=float(self.ent_nm_fatol.get()),
                simplex_rel_size=float(self.ent_nm_simplex.get()),
            )
            self.orch.set_nm_config(cfg)

            messagebox.showinfo("Settings", "Controller settings applied.")
        except Exception as e:
            messagebox.showerror("Apply Settings failed", str(e))

    def _apply_dt_clicked(self) -> None:
        # MPC eval dt
        try:
            dte = float(self.ent_dt_eval.get())
            if hasattr(self.orch, "set_eval_dt"):
                self.orch.set_eval_dt(dte)  # type: ignore[attr-defined]
            else:
                # fallback: stash on orchestrator; your orchestrator should read this when building Objective
                setattr(self.orch, "eval_dt", dte)
            messagebox.showinfo("MPC eval dt", "Evaluation dt updated.")
        except Exception as e:
            messagebox.showerror("MPC eval dt", str(e))

    # ---------------- Live status updates ----------------

    def _on_status(self, st) -> None:
        """Called from orchestrator thread; marshal to Tk thread."""
        self.root.after(0, self._apply_status_to_ui, st)

    def _apply_status_to_ui(self, st) -> None:
        # header
        self.lbl_opc.config(text=f"OPC: {'Connected' if st.opc_connected else 'Disconnected'}")
        self.lbl_mode.config(text=f"Mode: {getattr(st, 'mode', '—')}")
        self.lbl_running.config(text=f"Plant: {'RUNNING' if getattr(st, 'is_running', False) else 'STOPPED'}")
        self.lbl_tts.config(text=f"T→switch: {_fmt_float(getattr(st, 'time_to_switch_s', None), 3)} s")
        self.lbl_budget.config(text=f"Budget: {_fmt_float(getattr(st, 'time_to_switch_s', None),3)} / SF={_fmt_float(getattr(st,'speed_factor',None),3)}")

        # optimizer panel
        solving = getattr(st, "last_cycle_status", "Idle")
        self.lbl_solving.config(text=solving)
        self.lbl_iter.config(text=f"Iter: {_fmt_int(getattr(st, 'last_cycle_iter', None))}")
        self.lbl_elapsed.config(text=f"Elapsed [s]: {_fmt_float(getattr(st, 'last_cycle_elapsed_s', None), 4)}")
        self.lbl_bestJ.config(text=f"Best J: {_fmt_float(getattr(st, 'best_J', None), 5)}")
        self.lbl_baseJ.config(text=f"Baseline J: {_fmt_float(getattr(st, 'baseline_J', None), 5)}")

        bx = getattr(st, "best_x", None)
        if bx:
            self.lbl_bestx.config(text=f"({', '.join(_fmt_float(v, 4) for v in bx)})")
        else:
            self.lbl_bestx.config(text="—")

        # KPIs from best_metrics
        pur_ex = _fmt_float(getattr(st, "kpi_pur_ex", None), 3)
        pur_ra = _fmt_float(getattr(st, "kpi_pur_ra", None), 3)
        dil_ex = _fmt_float(getattr(st, "kpi_dil_ex", None), 3)
        dil_ra = _fmt_float(getattr(st, "kpi_dil_ra", None), 3)

        self.lbl_pur.config(text=f"Pur_ex / Pur_ra: {pur_ex} / {pur_ra}")
        self.lbl_dil.config(text=f"Dil_ex / Dil_ra: {dil_ex} / {dil_ra}")

        # ACTIVE/NEXT block
        act = getattr(st, "active_setpoints", {}) or {}
        nxt = getattr(st, "next_setpoints", {}) or {}
        def g(d: Dict[str, float], k: str) -> str: return _fmt_float(d.get(k))
        self.lbl_act_F.config(text=g(act, "F")); self.lbl_nxt_F.config(text=g(nxt, "F"))
        self.lbl_act_Q1.config(text=g(act, "Q1")); self.lbl_nxt_Q1.config(text=g(nxt, "Q1"))
        self.lbl_act_Q2.config(text=g(act, "Q2")); self.lbl_nxt_Q2.config(text=g(nxt, "Q2"))
        self.lbl_act_Q4.config(text=g(act, "Q4")); self.lbl_nxt_Q4.config(text=g(nxt, "Q4"))
        self.lbl_act_SI.config(text=g(act, "SI")); self.lbl_nxt_SI.config(text=g(nxt, "SI"))

        # log line (only append when it changes)
        msg = getattr(st, "last_cycle_message", "")
        status = getattr(st, "last_cycle_status", "")
        self._append_log_once(status, msg)

        self.lbl_sf.config(text=f"SF: {_fmt_float(getattr(st, 'speed_factor', None),3)}")
        snap_idx = getattr(st, "last_snapshot_index", None)
        snap_ts = getattr(st, "last_snapshot_created_utc", "")
        if snap_idx is not None:
            self.lbl_snap.config(text=f"Snapshot s{snap_idx} @ {snap_ts}")
        else:
            self.lbl_snap.config(text="Snapshot: —")

    def _append_log_once(self, status: str, message: str) -> None:
        key = (status, message)
        if not hasattr(self, "_last_log_key") or self._last_log_key != key:
            self._last_log_key = key
            self.txt_log.config(state="normal")
            ts = time.strftime("%H:%M:%S")
            self.txt_log.insert("end", f"[{ts}] {status}: {message}\n")
            self.txt_log.see("end")
            self.txt_log.config(state="disabled")

    def _tick(self) -> None:
        # can be used for periodic cosmetics in future
        self.root.after(500, self._tick)

    # ---------------- Public API ----------------

    def run(self) -> None:
        self.root.mainloop()


# If you want to launch directly:
if __name__ == "__main__":
    raise SystemExit(
        "This GUI is meant to be launched by your app after constructing MPCOrchestrator."
    )
