import numpy as np
from typing import List

# IMPORTANT: leave False in production; True only for a quick debug bypass
BYPASS_TUBE_DELAY = False

class Tube:
    """
    Connecting tube with a *fixed* number of axial cells Nx.
    Transport delay is realized by a Courant number alpha = dt / (t_res / Nx),
    integrated by sub-steps so that alpha_sub <= 1 (CFL-safe).
    """
    def __init__(self, deadVolume: float, Nx_fixed: int = 20):
        self.deadVolume = float(deadVolume)  # [mL]
        self.Nx = int(Nx_fixed)              # fixed axial cells (do NOT change during run)
        self.components: List = []
        self.flowRate = 1.0                  # [mL/h]
        self.dt = 0.05                       # [s]
        self._alpha = 0.0                    # cells shifted per dt

    # ---------------- plumbing ----------------
    def add(self, comp):
        """Attach a component; allocate its profile with the fixed Nx."""
        newc = comp.copy()
        newc.c = np.zeros(self.Nx, dtype=float)
        self.components.append(newc)

    # ---- helpers ----
    def _update_alpha(self):
        """Recompute Courant number from current flow and dt."""
        # residence time [s] through this tube at current flow
        t_res = (self.deadVolume / max(self.flowRate, 1e-12)) * 3600.0
        dt_cell = t_res / max(self.Nx, 1)  # time per cell
        self._alpha = self.dt / max(dt_cell, 1e-12)

    # ---------------- lifecycle ----------------
    def init(self, flowRate, dt, Nx=None):
        """
        Initialize with flow and dt. Nx is ignored intentionally: Tube uses a *fixed* Nx.
        Does NOT clear profiles if already present (startup will allocate zeros).
        """
        self.flowRate = float(max(flowRate, 1e-12))
        self.dt = float(dt)
        self._update_alpha()
        # ensure every component has the right-sized vector
        for comp in self.components:
            if not hasattr(comp, "c") or len(comp.c) != self.Nx:
                comp.c = np.zeros(self.Nx, dtype=float)

    def reconfigure(self, flowRate=None, dt=None):
        """
        Update flow and/or dt and recompute alpha **without** touching concentration profiles.
        This is what rotate() should call.
        """
        if flowRate is not None:
            self.flowRate = float(max(flowRate, 1e-12))
        if dt is not None:
            self.dt = float(dt)
        self._update_alpha()

    # ---------------- time stepping ----------------
    def step(self, cins):
        """
        Advance concentrations by alpha cells each dt, using sub-steps with alpha_sub <= 1.
        Boundary condition: c[0] = cin at each sub-step (upwind injection).
        Returns per-component axial profiles (list of 1D arrays).
        """
        outputs = []
        # number of sub-steps so that each sub-step satisfies CFL (alpha_sub <= 1)
        n_sub = max(1, int(np.ceil(max(self._alpha, 0.0))))
        alpha_sub = self._alpha / n_sub if n_sub > 0 else 0.0
        alpha_sub = float(min(max(alpha_sub, 0.0), 1.0))  # clamp to [0,1]

        for comp, cin in zip(self.components, cins):
            c = comp.c
            cin_val = float(cin)
            if not np.isfinite(cin_val):
                cin_val = 0.0

            if BYPASS_TUBE_DELAY:
                c[:] = cin_val
                outputs.append(c.copy())
                continue

            # sub-steps: simple upwind shift with injection at x=0
            for _ in range(n_sub):
                # interior nodes shift from upstream
                c[1:] = (1.0 - alpha_sub) * c[1:] + alpha_sub * c[:-1]
                # inlet
                c[0] = cin_val

            outputs.append(c.copy())

        return outputs

    # ---------------- misc ----------------
    def deepCopy(self):
        cp = Tube(self.deadVolume, self.Nx)
        cp.flowRate, cp.dt, cp._alpha = self.flowRate, self.dt, self._alpha
        cp.components = [c.copy() for c in self.components]
        for c, cc in zip(self.components, cp.components):
            cc.c = np.copy(c.c)
        return cp
