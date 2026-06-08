"""
Mass-balance sweep for the C++ dynamic (EDM) solver.
Run from the ChroMo_PE directory so edm_nonlinear_solver.pyd is found.

Reports two separate quantities:
  feed_err  – normalisation quality: (trapz(feed,t) / feedTime·feedConc − 1)·100 %
              should be ~0 % after the normalise_feed_pulse fix.
  out_err   – outlet vs theoretical mass (only meaningful when the peak fully
              elutes within `time`; flagged with '(not eluted)' otherwise).
"""

import ctypes, os, sys, pathlib
import numpy as np

# ── load compiled C++ extension ───────────────────────────────────────────────
# Prefer the freshly-built binary in build/Release (full ABI name) over the
# possibly-stale root-level .pyd, falling back to the root copy if needed.
_cwd   = pathlib.Path('.').resolve()
_candidates = [
    _cwd / "build" / "Release" / "edm_nonlinear_solver.cp38-win_amd64.pyd",
    *(_cwd / f"edm_nonlinear_solver{ext}"
      for ext in (".pyd", ".so")),
]
_lib = next((p for p in _candidates if p.exists()), None)
if _lib is None:
    sys.exit("edm_nonlinear_solver.pyd/.so not found — run from ChroMo_PE/")
ctypes.WinDLL(str(_lib))
# Insert the directory containing the chosen .pyd at the FRONT so Python picks
# the fresh build over any stale root-level copy with the same module name.
sys.path.insert(0, str(_lib.parent))
import edm_nonlinear_solver as cpp
print(f"Loaded: {_lib}\n")


# ── helpers ───────────────────────────────────────────────────────────────────
def feed_time(flowRate, feedVol):
    return (feedVol / flowRate) * 3600.0


def feed_error_pct(feed_arr, t_arr, feedTime, feedConc):
    """Normalisation quality: 0% = perfect."""
    area = float(np.trapz(feed_arr, t_arr))
    target = feedTime * feedConc
    if target == 0:
        return float("nan")
    return (area / target - 1.0) * 100.0


def outlet_error_pct(result, flowRate, feedVol, feedConc):
    """Outlet vs theoretical. Only meaningful when peak has fully exited."""
    t = np.array(result.timestamps)
    c = np.array(result.concentration).reshape(len(t), -1)
    feed_mass   = feedVol * feedConc
    outlet_mass = 0.5 * float(np.sum((t[1:] - t[:-1]) * (c[1:, -1] + c[:-1, -1]))) \
                  * flowRate / 3600.0
    return (feed_mass - abs(outlet_mass)) * 100.0 / feed_mass


# Positional order expected by edm_dynamic_solver:
# flowRate, length, diameter, feedVol, feedConc, porosity,
# langmuirConst, disperCoef, saturCoef, Nx, Nt, time, debug_print, full
DEFAULTS = dict(flowRate=150, length=320, diameter=10, feedVol=4,
                feedConc=6, porosity=0.4, langmuirConst=2,
                disperCoef=2, saturCoef=20, Nx=30, Nt=3000,
                time=10800, debug_print=False, full=True)

_ORDER = ["flowRate","length","diameter","feedVol","feedConc","porosity",
          "langmuirConst","disperCoef","saturCoef","Nx","Nt","time",
          "debug_print","full"]

def run(label, show_outlet=False, **kw):
    p = {**DEFAULTS, **kw}
    r = cpp.edm_dynamic_solver(*[p[k] for k in _ORDER])

    ft    = feed_time(p["flowRate"], p["feedVol"])
    feed  = np.array(r.feed) if r.feed is not None else np.zeros(len(r.timestamps))
    t_arr = np.array(r.timestamps)

    fe = feed_error_pct(feed, t_arr, ft, p["feedConc"])

    if show_outlet:
        oe = outlet_error_pct(r, p["flowRate"], p["feedVol"], p["feedConc"])
        print(f"  {label:<50}  feed_err={fe:+.3f}%   out_err={oe:+.2f}%")
    else:
        print(f"  {label:<50}  feed_err={fe:+.3f}%")


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 – Feed normalisation quality (all Langmuir default params)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 75)
print("PART 1 — Feed normalisation quality  (feed_err should be ~0 %)")
print("=" * 75)

print("\n--- baseline ---")
run("defaults  Nt=3000  Nx=30")

print("\n--- vary Nt ---")
for nt in [500, 1000, 2000, 3000, 5000, 8000]:
    run(f"Nt={nt}", Nt=nt)

print("\n--- vary Nx ---")
for nx in [10, 20, 30, 50, 80]:
    run(f"Nx={nx}", Nx=nx)

print("\n--- vary feedVol (feedTime changes) ---")
for fv in [1, 2, 4, 8, 16]:
    run(f"feedVol={fv} mL  (feedTime={feed_time(150,fv):.0f} s)", feedVol=fv)

print("\n--- vary flowRate ---")
for fr in [50, 100, 150, 300, 600]:
    run(f"flowRate={fr} mL/h  (feedTime={feed_time(fr,4):.0f} s)", flowRate=fr)

print("\n--- vary disperCoef ---")
for dc in [0.5, 1, 2, 5, 10, 20]:
    run(f"disperCoef={dc} mm2/s", disperCoef=dc)

print("\n--- vary langmuirConst ---")
for lc in [0.5, 1, 2, 5, 10]:
    run(f"langmuirConst={lc}", langmuirConst=lc)

print("\n--- vary saturCoef ---")
for sc in [5, 10, 20, 50, 100]:
    run(f"saturCoef={sc}", saturCoef=sc)

print("\n--- Nt × Nx grid ---")
for nt in [500, 1000, 3000, 6000]:
    for nx in [15, 30, 60]:
        run(f"Nt={nt}  Nx={nx}", Nt=nt, Nx=nx)

# ═════════════════════════════════════════════════════════════════════════════
# PART 2 – Outlet mass balance (fast-eluting near-linear isotherm)
# Conditions chosen so the peak fully exits within `time`:
#   langmuirConst=0.01, saturCoef=1 → initial slope ≈ 0.01 (almost no adsorption)
#   time=500 s is >> elution time (~245 s) so the peak is fully out.
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 75)
print("PART 2 — Outlet mass balance  (near-linear isotherm, peak fully elutes)")
print("         langmuirConst=0.01, saturCoef=1, time=500 s")
print("=" * 75)

FAST = dict(langmuirConst=0.01, saturCoef=1, time=500)

print("\n--- baseline ---")
run("defaults + near-linear  Nt=3000  Nx=30", show_outlet=True, **FAST)

print("\n--- vary Nt ---")
for nt in [500, 1000, 2000, 3000, 5000]:
    run(f"Nt={nt}", show_outlet=True, Nt=nt, **FAST)

print("\n--- vary Nx ---")
for nx in [10, 20, 30, 50, 80]:
    run(f"Nx={nx}", show_outlet=True, Nx=nx, **FAST)

print("\n--- vary feedVol ---")
for fv in [1, 2, 4, 8]:
    run(f"feedVol={fv} mL", show_outlet=True, feedVol=fv, **FAST)

print("\n--- vary flowRate ---")
for fr in [50, 100, 150, 300, 600]:
    run(f"flowRate={fr} mL/h", show_outlet=True, flowRate=fr, **FAST)

print("\n--- vary disperCoef ---")
for dc in [0.5, 1, 2, 5, 10]:
    run(f"disperCoef={dc} mm2/s", show_outlet=True, disperCoef=dc, **FAST)

print("\nDone.")
