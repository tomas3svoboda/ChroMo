# test.py — full-state EKF preflight without modifying cn_adapter.py

import importlib
import numpy as np

# Import project modules
mods = {m: importlib.import_module(m) for m in ("twin", "cn_adapter", "ekf_core", "sensors")}

# Build a minimal adapter class on-the-fly from the mixin only
CNAdapterFullMixin = mods["cn_adapter"].CNAdapterFullMixin
class _CN(CNAdapterFullMixin):
    """Minimal full-state CN adapter (no legacy 2-col API)."""
    pass

# Instantiate pieces
Twin        = mods["twin"].DigitalTwin
EKFCore     = mods["ekf_core"].EKFCore
EKFConfig   = mods["ekf_core"].EKFConfig
SensorModel = mods["sensors"].SensorModel

cn   = _CN()
twin = Twin()  # default params/plant
cfg  = EKFConfig(full_state=True)
ekf  = EKFCore(cfg)
sens = SensorModel()

# 1) Twin bring-up & dimension check
N = twin.full_state_dim()
assert N > 0, "Twin full_state_dim() returned 0"
t0 = twin.get_time()
# Advance twin a few internal steps so CN blocks are initialized
dt = float(twin.params.get("dt", 0.05))
twin.step_until(t0 + 3 * dt)

# 2) Build full-state A_seq and selector S
A_seq = cn.build_A_sequence_full(twin, t0, twin.get_time())
S     = cn.build_selector_S_full(twin)

assert len(A_seq) > 0, "A_seq_full is empty"
for i, A in enumerate(A_seq):
    assert A.shape == (N, N), f"A_seq[{i}] has shape {A.shape}, expected {(N, N)}"
assert S.shape == (4, N), f"S_full has shape {S.shape}, expected {(4, N)}"

# 3) EKF tick with zero measurements (just a math pass)
y = np.zeros(4, dtype=float)
diag = ekf.tick(
    A_seq=A_seq,
    S=S,
    y_meas=y,
    sensor_model=sens,
    twin=twin,
    dt_model=dt,
    t_start=t0,
    t_end=twin.get_time(),
)

# 4) Report essentials
ok = diag.get("status") == "ok"
print("=== Full-State EKF Preflight ===")
print("N (profiles)   :", N)
print("A_seq length   :", len(A_seq))
print("S shape        :", S.shape)
print("EKF status     :", diag.get("status"))
print("Residual norm  :", f"{diag.get('residual', float('nan')):.3e}")
print("NIS            :", f"{diag.get('nis', float('nan')):.3e}")
print("Innov          :", diag.get("innov"))
print("OK             :", ok)
