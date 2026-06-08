# objective_function.py
"""
Horizon-based objective function for SMB MPC, using the SAME SMBStation model.
Each optimizer evaluation deep-copies the given SMBStation, runs for N periods,
and computes horizon KPIs + a scalar objective.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import copy
import math

SAFE_EPS = 1e-12


def smb_all_flows(feed: float, eluent: float, extract: float, recycle: float) -> Dict[str, float]:
    V_I   = eluent + recycle
    V_II  = V_I - extract
    V_III = V_II + feed
    V_IV  = recycle
    raffinate = V_III - recycle
    if abs((feed + eluent) - (extract + raffinate)) > 1e-6:
        raise ValueError("Flow balance error: F + D must equal E + R.")
    return {
        "feed": feed, "eluent": eluent, "extract": extract, "recycle": recycle,
        "raffinate": raffinate,
        "zone1": V_I, "zone2": V_II, "zone3": V_III, "zone4": V_IV,
    }


@dataclass
class ObjectiveWeights:
    purity_man_extract: float = 1.0
    purity_gal_raff: float   = 1.0
    yield_man_extract: float = 1.0
    yield_gal_raff: float    = 1.0
    eluent_consumption: float = 1.0


class SMBObjectiveFunction:
    def __init__(
        self,
        smb_current,
        horizon_periods: int = 4,
        fixed_extract: float = 32.0,
        man_name: str = "Man",
        gal_name: str = "Gal",
        weights: Optional[ObjectiveWeights] = None,
        bounds: Optional[Dict[str, tuple]] = None,
    ):
        self._plant_template = smb_current
        self.horizon = int(horizon_periods)
        self.E = float(fixed_extract)
        self.man_name = man_name
        self.gal_name = gal_name
        self.weights = weights or ObjectiveWeights()
        self.bounds = bounds or {}

        self.dt = float(smb_current.settings["dt"])
        self.switch_interval = float(smb_current.interval)
        self.steps_per_period = int(round(self.switch_interval / self.dt))
        if self.steps_per_period <= 0:
            raise ValueError("switch_interval/dt must be positive.")

        self.man_idx = self._find_comp_idx(smb_current, self.man_name)
        self.gal_idx = self._find_comp_idx(smb_current, self.gal_name)
        self.c_feed_man = float(smb_current.components[self.man_idx].feedConc)
        self.c_feed_gal = float(smb_current.components[self.gal_idx].feedConc)

    def __call__(self, x: List[float]) -> float:
        start_time = time.time()
        print(f"[OBJ] Evaluation start: x={x}")

        metrics = self.evaluate_sequence(x)

        # Scalar objective (to MINIMIZE):
        #  - Purities & Yields enter with a MINUS sign → they are MAXIMIZED
        #  - Eluent consumption enters with a PLUS sign → it is MINIMIZED
        J = (
            - float(self.weights.purity_man_extract) * float(metrics["Purity_Man_Extract_pct"])
            - float(self.weights.purity_gal_raff)    * float(metrics["Purity_Gal_Raff_pct"])
            - float(self.weights.yield_man_extract)  * float(metrics["Yield_Man_Extract_pct"])
            - float(self.weights.yield_gal_raff)     * float(metrics["Yield_Gal_Raff_pct"])
            + float(self.weights.eluent_consumption) * float(metrics["EluentConsumption_VperMass"])
        )
        if not math.isfinite(J):
            J = math.inf

        if self.bounds:
            J += self._bounds_penalty(x)

        elapsed = time.time() - start_time
        print(f"[OBJ] Done in {elapsed:.3f} s → J={J:.3f}")
        return float(J)

    def evaluate_sequence(self, x: List[float]) -> Dict[str, float]:
        # Expect only 3 decision variables now
        if len(x) != 3:
            raise ValueError(f"Expected 3 decision vars (F,D,Q4), got {len(x)}.")

        smb = self._plant_template.deepCopy()
        use_fast = hasattr(smb, "step_fast_outlets")

        m_ex_man = m_ex_gal = 0.0
        m_ra_man = m_ra_gal = 0.0
        vol_desorb_in = 0.0
        m_feed_man = m_feed_gal = 0.0

        # Unpack same parameters for all periods
        F, D, Q4 = float(x[0]), float(x[1]), float(x[2])
        flows = smb_all_flows(F, D, self.E, Q4)
        Q_ex = flows["extract"]
        Q_ra = flows["raffinate"]
        dt = self.dt

        for _ in range(self.horizon):
            # Set same flows for every period
            smb.setFlowRateZone(1, flows["zone1"])
            smb.setFlowRateZone(2, flows["zone2"])
            smb.setFlowRateZone(3, flows["zone3"])
            smb.setFlowRateZone(4, flows["zone4"])

            for _step in range(self.steps_per_period):
                if use_fast:
                    c_ex_man, c_ex_gal, c_ra_man, c_ra_gal = smb.step_fast_outlets()
                else:
                    res = smb.step(1)
                    c_ex_man = self._outlet(res, zone=1, comp_idx=self.man_idx)
                    c_ex_gal = self._outlet(res, zone=1, comp_idx=self.gal_idx)
                    c_ra_man = self._outlet(res, zone=3, comp_idx=self.man_idx)
                    c_ra_gal = self._outlet(res, zone=3, comp_idx=self.gal_idx)

                m_ex_man += c_ex_man * Q_ex * dt
                m_ex_gal += c_ex_gal * Q_ex * dt
                m_ra_man += c_ra_man * Q_ra * dt
                m_ra_gal += c_ra_gal * Q_ra * dt

                vol_desorb_in += (D + F) * dt
                m_feed_man += self.c_feed_man * F * dt
                m_feed_gal += self.c_feed_gal * F * dt

        total_ex = m_ex_man + m_ex_gal
        total_ra = m_ra_man + m_ra_gal

        purity_man_ex = 100.0 * (m_ex_man / total_ex) if total_ex > 0 else 0.0
        purity_gal_ra = 100.0 * (m_ra_gal / total_ra) if total_ra > 0 else 0.0

        yield_man_ex = 100.0 * (m_ex_man / m_feed_man) if m_feed_man > 0 else 0.0
        yield_gal_ra = 100.0 * (m_ra_gal / m_feed_gal) if m_feed_gal > 0 else 0.0

        m_products_out = m_ex_man + m_ra_gal
        eluent_cons = (vol_desorb_in / m_products_out) if m_products_out > 0 else float("inf")

        return {
            "Purity_Man_Extract_pct": max(0.0, min(100.0, float(purity_man_ex))),
            "Purity_Gal_Raff_pct":    max(0.0, min(100.0, float(purity_gal_ra))),
            "Yield_Man_Extract_pct":  max(0.0, min(100.0, float(yield_man_ex))),
            "Yield_Gal_Raff_pct":     max(0.0, min(100.0, float(yield_gal_ra))),
            "EluentConsumption_VperMass": float(eluent_cons) if math.isfinite(eluent_cons) else math.inf,
        }

    def _outlet(self, res: Dict[int, list], zone: int, comp_idx: int) -> float:
        try:
            arrays_last_obj = res[zone][-1]
            profile = arrays_last_obj[comp_idx]
            return float(profile[-1])
        except Exception:
            return 0.0

    @staticmethod
    def _find_comp_idx(smb, name: str) -> int:
        for i, comp in enumerate(smb.components):
            if comp.name == name:
                return i
        raise ValueError(f"Component '{name}' not found in SMBStation.components")

    def _bounds_penalty(self, x: List[float]) -> float:
        if not self.bounds:
            return 0.0
        loF, hiF = self.bounds.get('F', (-1e9, 1e9))
        loD, hiD = self.bounds.get('D', (-1e9, 1e9))
        loQ, hiQ = self.bounds.get('Q4', (-1e9, 1e9))

        F, D, Q4 = x[0], x[1], x[2]
        pen  = self._quad_violation(F, loF, hiF)
        pen += self._quad_violation(D, loD, hiD)
        pen += self._quad_violation(Q4, loQ, hiQ)
        return pen

    @staticmethod
    def _quad_violation(val: float, lo: float, hi: float) -> float:
        if val < lo:
            d = lo - val
            return d * d
        if val > hi:
            d = val - hi
            return d * d
        return 0.0
