# mpc_objective.py  Objective for SMB MPC (Q1,Q2,Q4,SI version)
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional, Tuple
from collections import deque, OrderedDict
import copy, math, time, numpy as np
from ChroMo_SMB.EKF.SMB.SMBStation import SMBStation


@dataclass
class Bounds:
    Q1: Tuple[float, float]
    Q2: Tuple[float, float]
    Q4: Tuple[float, float]
    SI: Tuple[float, float]


def _ramp_weights(n: int, start: float = 0.5, end: float = 1.0) -> np.ndarray:
    if n <= 0: return np.zeros(0, dtype=float)
    if n == 1: return np.array([end], dtype=float)
    return np.linspace(start, end, num=n, dtype=float)


class _LRUCache:
    def __init__(self, capacity: int = 256) -> None:
        self._cap = int(capacity)
        self._od: OrderedDict = OrderedDict()
    def get(self, key):
        if key in self._od:
            v = self._od.pop(key); self._od[key] = v; return v
        return None
    def put(self, key, value) -> None:
        self._od[key] = value
        if len(self._od) > self._cap: self._od.popitem(last=False)
    def clear(self) -> None: self._od.clear()


class Objective:
    """
    Cost (minimize):
        J = - w_pur_ex * Pu_ex(%) - w_pur_ra * Pu_ra(%) + w_dil_ex * Di_ex(%) + w_dil_ra * Di_ra(%)

    KPIs use dt-invariant integrals over the scoring window (mass/volume averages):
        <c> = (Σ c * Q * dt) / (Σ Q * dt)

    Decision variables:  x = (Q1, Q2, Q4, SI)
        Zone flows applied as: V1=Q1, V2=Q2, V3=Q2+F, V4=Q4
        Outlet flows used for KPI integration:
            Extract E = V1 - V2 = Q1 - Q2
            Raff    R = V3 - V4 = (Q2 + F) - Q4
    """
    def __init__(
        self,
        *,
        horizon_si: int,
        weights: Tuple[float, float, float, float],     # (w_dil_ex, w_dil_ra, w_pur_ex, w_pur_ra)
        bounds: Dict[str, Tuple[float, float]],         # keys: 'Q1','Q2','Q4','SI'
        feed_conc_A: float = 7.27,
        feed_conc_B: float = 3.42,
        purity_floor_ratio: float = 0.01,               # < feed/20 => purity := 0%
        cap_factor: float = 10.0,                       # clamp instant c to [0, cap_factor*feed]
        eval_dt: float | None = None,
        comp_index: Dict[str, int] | None = None,
        outlet_zone: Dict[str, int] | None = None,
        eps: float = 1e-12,
        burn_in_si: int = 0,
        score_si: Optional[int] = None,
        taper: bool = False,
        taper_span: int = 4,
        taper_start: float = 0.5,
        cache_enable: bool = True,
        cache_decimals: Tuple[int, int, int, int] = (3, 3, 3, 1),
    ) -> None:

        # --- recycle-start cleanliness penalty params ---
        self.recycle_window_s: float = 10.0
        self.recycle_thr: float = 0.05
        self.recycle_max: float = 10.0 #g/L
        self.recycle_scale: float = 200.0
        self.recycle_print_threshold: float = 10.0

        self.horizon_si = int(horizon_si)
        self.weights = tuple(float(w) for w in weights)
        self.bounds = dict(bounds)
        self.comp_index = comp_index or {"man": 0, "gal": 1}
        self.outlet_zone = outlet_zone or {"extract": 1, "raffinate": 3}
        self.eps = float(eps)

        self.feed_conc_A = float(feed_conc_A)
        self.feed_conc_B = float(feed_conc_B)
        self.purity_floor_ratio = float(purity_floor_ratio)
        self.cap_factor = float(cap_factor)
        self._eval_dt = eval_dt

        self.burn_in_si = max(0, int(burn_in_si))
        self.score_si = None if score_si is None else max(0, int(score_si))
        self.taper = bool(taper)
        self.taper_span = max(1, int(taper_span))
        self.taper_start = float(taper_start)

        self._seed_station = None
        self._active = {"F": None, "Q1": None, "Q2": None, "Q4": None, "SI": None}
        self._station_t0 = None
        self._t0_ready = False

        self._progress: Dict[str, object] = {
            "n_eval": 0, "best_x": None, "best_J": None, "best_metrics": None,
            "last_x": None, "last_J": None, "last_metrics": None,
            "history_tail": deque(maxlen=50),
        }
        self._eval_hook: Optional[Callable[[Dict[str, object]], None]] = None

        self._cache_enable = bool(cache_enable)
        self._cache_round = tuple(int(d) for d in cache_decimals)
        self._cache = _LRUCache(capacity=256)

    # ---------- Public API ----------

    def set_seed(self, station_copy, active_setpoints: Dict[str, float]) -> None:
        self._seed_station = (
            station_copy.deepCopy() if hasattr(station_copy, "deepCopy") else copy.deepcopy(station_copy)
        )
        self._active = {
            k: float(active_setpoints.get(k)) if active_setpoints.get(k) is not None else None
            for k in ("F", "Q1", "Q2", "Q4", "SI")
        }
        self._station_t0 = None
        self._t0_ready = False
        self.reset_progress()
        self._cache.clear()

    def prepare_t0(self) -> None:
        if self._seed_station is None:
            raise RuntimeError("Objective: seed station not set. Call set_seed(...) first.")
        s = self._clone_station(self._seed_station)
        if self._eval_dt is not None and hasattr(s, "retime"):
            s.retime(self._eval_dt)
        self._t0_station = s
        F  = self._need("F")
        Q1 = self._need("Q1"); Q2 = self._need("Q2"); Q4 = self._need("Q4"); SI = self._need("SI")
        self._apply_controls(s, Q1, Q2, Q4, SI, F)
        self._step_until_next_boundary(s)
        self._station_t0 = s
        self._t0_ready = True

    def evaluate(self, x: np.ndarray) -> Tuple[float, Dict[str, float]]:
        if not self._t0_ready or self._station_t0 is None:
            raise RuntimeError("Objective: t0 not prepared. Call prepare_t0() first.")

        # raw x and hard OOB reject (no prediction)
        x = np.asarray(x, dtype=float)
        lb = np.array([self.bounds['Q1'][0], self.bounds['Q2'][0], self.bounds['Q4'][0], self.bounds['SI'][0]], float)
        ub = np.array([self.bounds['Q1'][1], self.bounds['Q2'][1], self.bounds['Q4'][1], self.bounds['SI'][1]], float)
        if np.any(x < lb) or np.any(x > ub):
            J = 1e12
            metrics = {"feasible": False}
            xc = np.minimum(np.maximum(x, lb), ub)  # send clipped to telemetry
            self._maybe_cache_and_record(xc, J, metrics)
            return float(J), metrics

        # clipped x and active F
        xc = self._clip_x(x)
        Q1, Q2, Q4, SI = [float(v) for v in xc]
        F = self._need("F")

        # derived outlet flows (must be strictly positive) – still before any prediction
        E_flow = Q1 - Q2
        R_flow = (Q2 + F) - Q4
        D_flow = Q1 - Q4
        if (E_flow <= 0.0) or (R_flow <= 0.0) or (D_flow <= 0.0):
            violations = int(E_flow <= 0.0) + int(R_flow <= 0.0) + int(D_flow <= 0.0)
            J = 1e12 + 500.0 * violations
            metrics = {"feasible": False}
            self._maybe_cache_and_record(xc, J, metrics)
            return float(J), metrics

        # ---- prediction path (only feasible) ----
        s = self._clone_station(self._station_t0)
        if self._eval_dt is not None and hasattr(s, "retime"):
            if abs(float(getattr(s, "dt", s.settings.get("dt", 0.0))) - self._eval_dt) > 1e-12:
                s.retime(self._eval_dt)
        self._apply_controls(s, Q1, Q2, Q4, SI, F)

        dt = float(getattr(s, "dt", 0.0) or s.settings.get("dt", 0.05))
        steps_per_SI = max(1, int(round(SI / max(dt, 1e-12))))
        total_steps = self.horizon_si * steps_per_SI

        # recycle-start sampling (first ~1 s of each SI)
        n0 = max(1, int(round(self.recycle_window_s / max(dt, 1e-12))))
        ra_start_sum = np.zeros(self.horizon_si, dtype=float)
        ra_start_cnt = np.zeros(self.horizon_si, dtype=float)

        # per-SI integrals
        V_ex = np.zeros(self.horizon_si, dtype=float)
        V_ra = np.zeros_like(V_ex)
        M_ex_A = np.zeros_like(V_ex); M_ex_B = np.zeros_like(V_ex)
        M_ra_A = np.zeros_like(V_ex); M_ra_B = np.zeros_like(V_ex)

        # outlet flows (positive here)
        Q_ex = E_flow
        Q_ra = R_flow

        # robustness caps
        nonfinite_seen = False
        C_CAP_A = self.cap_factor * max(self.feed_conc_A, self.eps)
        C_CAP_B = self.cap_factor * max(self.feed_conc_B, self.eps)

        def _clip_c(v: float, cap: float) -> float:
            nonlocal nonfinite_seen
            if not np.isfinite(v):
                nonfinite_seen = True
                return 0.0
            if v < 0.0: return 0.0
            if v > cap: return cap
            return float(v)

        for step in range(total_steps):
            si_idx = step // steps_per_SI
            try:
                c_ex_A, c_ex_B, c_ra_A, c_ra_B = s.step_fast_outlets()
            except Exception:
                profs = s.step(1)
                c_ex_A, c_ex_B, c_ra_A, c_ra_B = self._last_cells(profs)

            c_ex_A = _clip_c(c_ex_A, C_CAP_A); c_ex_B = _clip_c(c_ex_B, C_CAP_B)
            c_ra_A = _clip_c(c_ra_A, C_CAP_A); c_ra_B = _clip_c(c_ra_B, C_CAP_B)

            dV_ex = Q_ex * dt
            dV_ra = Q_ra * dt

            if (step % steps_per_SI) < n0:
                c_sum_ra = c_ra_A + c_ra_B
                ra_start_sum[si_idx] += c_sum_ra
                ra_start_cnt[si_idx] += 1.0

            V_ex[si_idx]  += dV_ex; V_ra[si_idx]  += dV_ra
            M_ex_A[si_idx] += c_ex_A * dV_ex; M_ex_B[si_idx] += c_ex_B * dV_ex
            M_ra_A[si_idx] += c_ra_A * dV_ra; M_ra_B[si_idx] += c_ra_B * dV_ra

        # scoring window
        start = min(self.burn_in_si, self.horizon_si)
        end = self.horizon_si if self.score_si is None else min(self.horizon_si, start + self.score_si)
        win_len = max(0, end - start)
        if win_len == 0:
            J = 1e9
            metrics = {"pur_ex": 0.0, "pur_ra": 0.0, "dil_ex": 1e9, "dil_ra": 1e9,
                       "mass_ex_man": 0.0, "mass_ex_gal": 0.0, "mass_ra_man": 0.0, "mass_ra_gal": 0.0,
                       "vol_ex": 0.0, "vol_ra": 0.0, "feasible": False}
            self._maybe_cache_and_record(xc, J, metrics); return float(J), metrics

        if self.taper:
            n_taper = min(self.taper_span, win_len)
            w = _ramp_weights(n_taper, start=self.taper_start, end=1.0)
            if n_taper < win_len: w = np.concatenate([w, np.ones(win_len - n_taper, dtype=float)])
        else:
            w = np.ones(win_len, dtype=float)

        # weighted sums
        V_ex_sum  = float(np.dot(w, V_ex[start:end]))
        V_ra_sum  = float(np.dot(w, V_ra[start:end]))
        M_ex_A_sum = float(np.dot(w, M_ex_A[start:end]))
        M_ex_B_sum = float(np.dot(w, M_ex_B[start:end]))
        M_ra_A_sum = float(np.dot(w, M_ra_A[start:end]))
        M_ra_B_sum = float(np.dot(w, M_ra_B[start:end]))

        if not np.isfinite(V_ex_sum) or not np.isfinite(V_ra_sum) or V_ex_sum <= 0.0 or V_ra_sum <= 0.0:
            J = 1e12
            metrics = {"pur_ex": 0.0, "pur_ra": 0.0, "dil_ex": 1e9, "dil_ra": 1e9,
                       "mass_ex_man": M_ex_A_sum, "mass_ex_gal": M_ex_B_sum,
                       "mass_ra_man": M_ra_A_sum, "mass_ra_gal": M_ra_B_sum,
                       "vol_ex": V_ex_sum, "vol_ra": V_ra_sum, "feasible": False}
            self._maybe_cache_and_record(xc, J, metrics); return float(J), metrics

        # mean concentrations
        cA_ex_bar = M_ex_A_sum / max(V_ex_sum, self.eps)
        cB_ex_bar = M_ex_B_sum / max(V_ex_sum, self.eps)
        cA_ra_bar = M_ra_A_sum / max(V_ra_sum, self.eps)
        cB_ra_bar = M_ra_B_sum / max(V_ra_sum, self.eps)

        # purity with floor rule
        thr_A = self.purity_floor_ratio * max(self.feed_conc_A, self.eps)
        thr_B = self.purity_floor_ratio * max(self.feed_conc_B, self.eps)
        pur_ex = 0.0
        pur_ra = 0.0
        if cA_ex_bar >= thr_A and (cA_ex_bar + cB_ex_bar) > self.eps:
            pur_ex = (cA_ex_bar / (cA_ex_bar + cB_ex_bar)) * 100.0
        if cB_ra_bar >= thr_B and (cA_ra_bar + cB_ra_bar) > self.eps:
            pur_ra = (cB_ra_bar / (cA_ra_bar + cB_ra_bar)) * 100.0
        pur_ex = float(min(100.0, max(0.0, pur_ex)))
        pur_ra = float(min(100.0, max(0.0, pur_ra)))

        # dilution (% relative to feed)
        dil_ex = 100.0 * (1.0 - (cA_ex_bar / max(self.feed_conc_A, self.eps)))
        dil_ra = 100.0 * (1.0 - (cB_ra_bar / max(self.feed_conc_B, self.eps)))
        dil_ex = float(min(100.0, max(-100.0, dil_ex)))
        dil_ra = float(min(100.0, max(-100.0, dil_ra)))

        w_dil_ex, w_dil_ra, w_pur_ex, w_pur_ra = self.weights

        # recycle-start cleanliness (not added to J unless you uncomment below)
        pen_ra = 0.0
        if self.recycle_scale > 0.0 and (end - start) > 0:
            c0 = np.zeros(win_len, dtype=float)
            for k in range(start, end):
                denom = max(1.0, ra_start_cnt[k])
                c_avg = float(ra_start_sum[k] / denom)
                if c_avg <= self.recycle_thr:
                    z = 0.0
                else:
                    z = (min(c_avg, self.recycle_max) - self.recycle_thr) / max(1e-12, (self.recycle_max - self.recycle_thr))
                c0[k - start] = z
            pen_ra = float(self.recycle_scale * np.dot(w, c0) / max(1.0, np.sum(w)))
            if pen_ra > self.recycle_print_threshold:
                print(f"[OBJ] Recycle-start penalty = {pen_ra:.2f}  (avg c@start window mapped from {self.recycle_thr}..{self.recycle_max} g/L)")

        J = (- w_pur_ex * pur_ex - w_pur_ra * pur_ra
             + w_dil_ex * dil_ex + w_dil_ra * dil_ra) / (w_pur_ex + w_pur_ra + w_dil_ex + w_dil_ra)
        J += pen_ra
        if not np.isfinite(J):
            J = 1e12

        metrics = {
            "pur_ex": float(pur_ex), "pur_ra": float(pur_ra),
            "dil_ex": float(dil_ex), "dil_ra": float(dil_ra),
            "cA_ex_bar": float(cA_ex_bar), "cB_ex_bar": float(cB_ex_bar),
            "cA_ra_bar": float(cA_ra_bar), "cB_ra_bar": float(cB_ra_bar),
            "mass_ex_man": float(M_ex_A_sum), "mass_ex_gal": float(M_ex_B_sum),
            "mass_ra_man": float(M_ra_A_sum), "mass_ra_gal": float(M_ra_B_sum),
            "vol_ex": float(V_ex_sum), "vol_ra": float(V_ra_sum),
            "pen_recycle_start": float(pen_ra),
            "feasible": True,
        }

        self._maybe_cache_and_record(xc, J, metrics)
        return float(J), metrics

    # ---------- Telemetry ----------
    def reset_progress(self) -> None:
        self._progress.update({
            "n_eval": 0, "best_x": None, "best_J": None, "best_metrics": None,
            "last_x": None, "last_J": None, "last_metrics": None,
            "history_tail": deque(maxlen=50),
        })

    def get_progress(self) -> Dict[str, object]:
        p = dict(self._progress); p["history_tail"] = list(self._progress["history_tail"])
        if p.get("best_x") is not None: p["best_x"] = np.asarray(p["best_x"]).tolist()
        if p.get("last_x") is not None: p["last_x"] = np.asarray(p["last_x"]).tolist()
        return p

    def set_eval_hook(self, hook: Callable[[Dict[str, object]], None]) -> None:
        self._eval_hook = hook

    # ---------- Internals ----------
    def _clone_station(self, s):
        return s.deepCopy() if hasattr(s, "deepCopy") else copy.deepcopy(s)

    def _apply_controls(self, s, Q1: float, Q2: float, Q4: float, SI: float, F: float) -> None:
        V1 = Q1
        V2 = Q2
        V3 = Q2 + F
        V4 = Q4
        s.setFlowRateZone(1, V1); s.setFlowRateZone(2, V2)
        s.setFlowRateZone(3, V3); s.setFlowRateZone(4, V4)
        try:
            s.apply_zone_flows_now()
        except AttributeError:
            if hasattr(s, "dt") and hasattr(s, "retime"):
                s.retime(float(getattr(s, "dt", s.settings.get("dt", 0.5))))
        s.setSwitchInterval(SI)

    def _step_until_next_boundary(self, s) -> None:
        prev_cd = float(getattr(s, "countdown", 0.0))
        dt = float(getattr(s, "dt", 0.0) or s.settings.get("dt", 0.05))
        SI = float(getattr(s, "interval", 0.0) or s.settings.get("Switch Interval", 0.0) or 1.0)
        max_steps = max(1, int(math.ceil(SI / max(dt, 1e-12))) + 5)
        for _ in range(max_steps):
            try: s.step_fast_outlets()
            except Exception: s.step(1)
            cd = float(getattr(s, "countdown", 0.0))
            if cd > prev_cd: return
            prev_cd = cd
        remaining = int(round((SI - float(getattr(s, "countdown", 0.0))) / max(dt, 1e-12)))
        if remaining > 0:
            try:
                for _ in range(remaining): s.step_fast_outlets()
            except Exception:
                s.step(remaining)

    def _last_cells(self, profs) -> Tuple[float, float, float, float]:
        def safe(zone: int, comp: int) -> float:
            try: return float(profs[zone][-1][comp][-1])
            except Exception: return 0.0
        ex_A = safe(self.outlet_zone["extract"], self.comp_index["man"])
        ex_B = safe(self.outlet_zone["extract"], self.comp_index["gal"])
        ra_A = safe(self.outlet_zone["raffinate"], self.comp_index["man"])
        ra_B = safe(self.outlet_zone["raffinate"], self.comp_index["gal"])
        return ex_A, ex_B, ra_A, ra_B

    def _clip_x(self, x: np.ndarray) -> np.ndarray:
        lb = np.array([self.bounds['Q1'][0], self.bounds['Q2'][0], self.bounds['Q4'][0], self.bounds['SI'][0]], float)
        ub = np.array([self.bounds['Q1'][1], self.bounds['Q2'][1], self.bounds['Q4'][1], self.bounds['SI'][1]], float)
        return np.minimum(np.maximum(x, lb), ub)

    def _need(self, key: str) -> float:
        v = self._active.get(key)
        if v is None:
            raise RuntimeError(f"Objective: missing ACTIVE setpoint '{key}'.")
        return float(v)

    def _maybe_cache_and_record(self, xc: np.ndarray, J: float, metrics: Dict[str, float]) -> None:
        if self._cache_enable:
            key = (round(float(xc[0]), self._cache_round[0]),
                   round(float(xc[1]), self._cache_round[1]),
                   round(float(xc[2]), self._cache_round[2]),
                   round(float(xc[3]), self._cache_round[3]))
            self._cache.put(key, (float(J), dict(metrics)))
        self._record_progress(xc, J, metrics)

    def _record_progress(self, xc: np.ndarray, J: float, metrics: Dict[str, float]) -> None:
        is_best = self._progress["best_J"] is None or (J < float(self._progress["best_J"]))
        self._progress["n_eval"] = int(self._progress["n_eval"]) + 1
        self._progress["last_x"] = xc.copy()
        self._progress["last_J"] = float(J)
        self._progress["last_metrics"] = dict(metrics)
        self._progress["history_tail"].append((xc.copy(), float(J), time.time()))
        if is_best:
            self._progress["best_x"] = xc.copy()
            self._progress["best_J"] = float(J)
            self._progress["best_metrics"] = dict(metrics)
        if self._eval_hook is not None:
            try:
                self._eval_hook({"x": xc.copy(), "J": float(J), "metrics": dict(metrics),
                                 "is_best": bool(is_best), "t_eval_s": time.time()})
            except Exception:
                pass
