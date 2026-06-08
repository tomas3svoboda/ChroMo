
import pandas as pd

# Prefer package imports, fall back to local files if needed
try:
    from SMB.SMBStation import SMBStation                  # package-style
    from SMB.LinColumn import LinColumn
except Exception:
    from SMBStation import SMBStation                       # local fallback
    from LinColumn import LinColumn

# Optional: use Tube if available; otherwise a simple pass-through
try:
    from SMB.Tube import Tube
except Exception:
    class Tube:
        def __init__(self, dead_volume=0.0):
            self.dead_volume = float(dead_volume)
            self.components = []
            self.flowRate = 0.0
        def add(self, comp): self.components.append(comp)
        def init(self, flowRate, dt, Nx):
            self.flowRate = float(flowRate)
        def step(self, cins): return [list(cins)]  # pass-through
        def getInfo(self): return {"type": "Tube", "dead_volume": self.dead_volume}
        def deepCopy(self): return Tube(self.dead_volume)


def simSMB_linear(
    end_time,
    # --- component names & feed concentrations (g/L) ---
    name_compA="Man", feed_concA=7.27,
    name_compB="Gal", feed_concB=3.42,
    # --- parameters to be FIT (per call) ---
    K_A=6.36,                 # Henry const for A  [–]
    K_B=3.62,                 # Henry const for B  [–]
    delta_shared=20.0,        # Bo (δ) shared by both components (>0)
    # --- fixed hydraulics/ops ---
    flow_rates=None,          # [Z1, Z2, Z3, Z4] in mL/h (fixed, not optimized)
    switch_interval=780,      # seconds (fixed, not optimized)
    # --- numerics & geometry ---
    dt=1, Nx=100,
    length_mm=310.0, diameter_mm=10.0, porosity=0.376, tube_dead_volume=0.2,
    verbose=False,
):
    """
    Linear EDM SMB simulation (Crank–Nicolson) with Henry isotherm.
    Returns: (df_extract, df_raffinate) with columns:
        time [s], concentration_A [g/L], concentration_B [g/L]
    """
    if flow_rates is None:
        flow_rates = [180.0, 93.0, 114.0, 45.0]

    # --- build station ---
    smb = SMBStation()
    smb.set_isotherm_mode('noncomp')  # default; linear model uses GenericColumn.step()

    # Add 1 LinColumn + Tube per zone
    for zone in (1, 2, 3, 4):
        smb.addColZone(
            zone,
            LinColumn(length_mm, diameter_mm, porosity),
            Tube(tube_dead_volume),
        )

    # Fixed ops
    for i, q in enumerate(flow_rates, start=1):
        smb.setFlowRateZone(i, float(q))
    smb.setSwitchInterval(int(switch_interval))
    smb.setdt(int(dt))
    smb.setNx(int(Nx))

    # --- create LINEAR components (shared delta; per-species Henry K) ---
    # SMBStation.createComponentAB sets comp.henryConst = henryConst and comp.delta = A
    # B is unused by LinColumn (only NonLin uses Di offset), so set 0.0
    smb.createComponentAB(name_compA, feedConc=float(feed_concA), henryConst=float(K_A), A=float(delta_shared), B=0.0)
    smb.createComponentAB(name_compB, feedConc=float(feed_concB), henryConst=float(K_B), A=float(delta_shared), B=0.0)

    # Initialize numerics
    smb.initCols()

    # --- simulate using fast outlet accessor (robust) ---
    times = []
    ex_A, ex_B, ra_A, ra_B = [], [], [], []
    t = 0.0
    while t < float(end_time):
        c_ex_A, c_ex_B, c_ra_A, c_ra_B = smb.step_fast_outlets()
        t += dt
        times.append(t)
        ex_A.append(c_ex_A); ex_B.append(c_ex_B)
        ra_A.append(c_ra_A); ra_B.append(c_ra_B)

    df_extract = pd.DataFrame({
        "time": times,
        "concentration_A": ex_A,
        "concentration_B": ex_B,
    })
    df_raffinate = pd.DataFrame({
        "time": times,
        "concentration_A": ra_A,
        "concentration_B": ra_B,
    })

    # attempt to free heavy internals
    try:
        for zone in smb.zones:
            smb.zones[zone].clear()
        smb.components.clear()
        smb.cins.clear()
        smb.flowRates.clear()
    except Exception:
        pass
    del smb

    return df_extract, df_raffinate
