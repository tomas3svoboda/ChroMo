# state_estimator.py
"""
Dummy state estimator for SMB MPC.

It advances a COPY of the current template SMBStation forward by K periods
using the provided decision sequence x (per-period [F, D, Q4]) and returns
the *end state* as the next template.

Default K=1 (end of first period) is the sensible receding-horizon choice.
If you *really* want end-of-horizon, set periods_to_advance=horizon, but that
will jump too far ahead for standard MPC.
"""

from typing import List
import copy

from objective_function import smb_all_flows  # same balance used everywhere


def _steps_per_period(smb) -> int:
    dt = float(smb.settings["dt"])
    interval = float(smb.interval)
    steps = int(round(interval / dt))
    if steps <= 0:
        raise ValueError("Invalid steps_per_period (check dt and switch_interval).")
    return steps


def advance_template_by_periods(
    smb_template,
    x_sequence: List[float],
    fixed_extract: float,
    periods_to_advance: int = 1,
):
    """
    Returns a NEW SMBStation equal to smb_template advanced by `periods_to_advance`
    periods under the first K periods of x_sequence.

    x_sequence layout: [F1, D1, Q41,  F2, D2, Q42,  ...]
    """
    smb = smb_template.deepCopy()
    steps = _steps_per_period(smb)

    idx = 0
    for k in range(periods_to_advance):
        F = float(x_sequence[idx]); D = float(x_sequence[idx+1]); Q4 = float(x_sequence[idx+2]); idx += 3
        flows = smb_all_flows(F, D, fixed_extract, Q4)

        smb.setFlowRateZone(1, flows["zone1"])
        smb.setFlowRateZone(2, flows["zone2"])
        smb.setFlowRateZone(3, flows["zone3"])
        smb.setFlowRateZone(4, flows["zone4"])

        for _ in range(steps):
            smb.step(1)

    return smb


def estimate_next_template(
    smb_template,
    x_best: List[float],
    fixed_extract: float,
    *,
    periods_to_advance: int = 1,
):
    """
    Public entry: what the controller should call after each optimization.
    By default, advance 1 period using the *first* period of the optimal sequence.
    """
    return advance_template_by_periods(
        smb_template=smb_template,
        x_sequence=x_best,
        fixed_extract=fixed_extract,
        periods_to_advance=periods_to_advance,
    )
