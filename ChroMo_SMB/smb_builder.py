# smb_builder.py
"""
Build a fresh SMBStation identical to the simulator's setup.

Params (dict):
    {
        'dt': 0.05,
        'Nx': 100,
        'switch_interval': 1894,
        'col_length': 320,
        'col_diameter': 10,
        'porosity': 0.376,
        'dead_volume': 0.5,
    }

Components (list[dict]):
    [
        {"name":"Man","feed_concentration":9,"henry_constant":4.55,"delta":54,"Di":0.0007},
        {"name":"Gal","feed_concentration":6,"henry_constant":2.77,"delta":84,"Di":0.0007},
    ]

Flows (dict) — primary streams:
    {'feed': 4.0, 'eluent': 77.0, 'extract': 32.0, 'recycle': 48.0}
"""

from typing import Dict, List
from SMB.SMBStation import SMBStation
from SMB.LinColumn_working import LinColumn
from SMB.Tube import Tube


def _all_flows(feed: float, eluent: float, extract: float, recycle: float) -> List[float]:
    """Return [zone1, zone2, zone3, zone4] using the same balance as the sim."""
    V_I   = eluent + recycle
    V_II  = V_I - extract
    V_III = V_II + feed
    V_IV  = recycle
    raffinate = V_III - recycle  # = V_III - V_IV
    if abs((feed + eluent) - (extract + raffinate)) > 1e-6:
        raise ValueError("Flow balance error: F + D must equal E + R.")
    return [V_I, V_II, V_III, V_IV]


def build_smb(params: Dict, components: List[Dict], flows: Dict) -> SMBStation:
    """
    Construct and initialize an SMBStation with given geometry, grid, and flows.
    Returns a READY TO SIMULATE station (cols/tubes initialized).
    """
    # --- Station & geometry ---
    smb = SMBStation()
    for z in range(4):
        smb.addColZone(
            z + 1,
            LinColumn(params['col_length'], params['col_diameter'], params['porosity']),
            Tube(params['dead_volume'])
        )

    # --- Discretization & timing ---
    smb.setdt(params['dt'])
    smb.setNx(params['Nx'])
    smb.setSwitchInterval(params['switch_interval'])

    # --- Components (linear isotherm with AB dispersion parameters) ---
    for c in components:
        smb.createComponentAB(
            c["name"],
            c["feed_concentration"],
            c["henry_constant"],
            c["delta"],
            c["Di"],
        )

    # --- Initial zone flows from primary streams ---
    zflows = _all_flows(flows['feed'], flows['eluent'], flows['extract'], flows['recycle'])
    for zone, fr in enumerate(zflows, start=1):
        smb.setFlowRateZone(zone, fr)

    # --- Initialize columns/tubes (allocates profiles & sets inlet conditions) ---
    smb.initCols()

    return smb
