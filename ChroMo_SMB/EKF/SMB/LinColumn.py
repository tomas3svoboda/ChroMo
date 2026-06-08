import numpy as np
import math
from scipy.linalg import lu_factor, lu_solve   # <— use LU once per change
from .GenericColumn import GenericColumn

class LinColumn(GenericColumn):
    """Chromatography column using EDM with linear isotherm (Crank–Nicolson)."""

    def __init__(self, length, diameter, porosity):
        super().__init__(length, diameter, porosity)
        self.columnType = "EDM with Linear Isotherm"
        # tolerances for change detection
        self._tol_rel = 1e-9
        self._tol_abs = 1e-12

    def init(self, flowRate, dt, Nx):
        """Initialize numerical parameters and matrices; set up watchdog cache."""
        self.flowRate = float(flowRate)
        self.dt = float(dt)
        self.Nx = int(Nx)

        self.flowSpeed = (self.flowRate * 1000 / 3600) / ((math.pi * (self.diameter ** 2) / 4) * self.porosity)
        self.dx = self.length / self.Nx

        print(self.flowSpeed)

        if not self.components:
            raise ValueError("LinColumn has no components! Add components before init().")

        # Allocate per-component numerics + cache for change detection
        for comp in self.components:
            # Precompute static-ish terms for given comp
            self._build_matrices_for_component(comp)  # builds A,B and LU, allocates _b, keeps comp.c

        # Keep a snapshot of parameters for the watchdog
        self._watch = self._snapshot_params()

    # ---------------- internal helpers ----------------

    def _snapshot_params(self):
        """Pack all parameters that affect A/B into a dict for change detection."""
        # Any of these changing requires a rebuild: dt, Nx, porosity, flowRate (→ flowSpeed),
        # and component-wise henryConst, delta
        comp_keys = tuple((c.henryConst, c.delta) for c in self.components)
        return dict(
            dt=self.dt, Nx=self.Nx, porosity=self.porosity,
            flowRate=self.flowRate, flowSpeed=self.flowSpeed,
            comp_keys=comp_keys
        )

    def _changed(self):
        """Return True if relevant params changed beyond tolerance."""
        w = self._watch
        if self.Nx != w["Nx"]:
            return True
        if abs(self.dt - w["dt"]) > self._tol_abs:
            return True
        if abs(self.porosity - w["porosity"]) > self._tol_abs:
            return True
        if abs(self.flowRate - w["flowRate"]) > max(self._tol_abs, self._tol_rel * (1.0 + abs(w["flowRate"]))):
            return True
        # If flowRate changed, recompute flowSpeed now for consistent checks
        new_flowSpeed = (self.flowRate * 1000 / 3600) / ((math.pi * (self.diameter ** 2) / 4) * self.porosity)
        if abs(new_flowSpeed - w["flowSpeed"]) > max(self._tol_abs, self._tol_rel * (1.0 + abs(w["flowSpeed"]))):
            return True
        curr_keys = tuple((c.henryConst, c.delta) for c in self.components)
        if curr_keys != w["comp_keys"]:
            return True
        return False

    def _refresh_if_needed(self):
        """Rebuild A/B and LU only when something relevant changed."""
        if not self._changed():
            return
        # Recompute speed & dx if geometry/discretization changed
        self.flowSpeed = (self.flowRate * 1000 / 3600) / ((math.pi * (self.diameter ** 2) / 4) * self.porosity)
        self.dx = self.length / self.Nx
        # Rebuild per-component numerics
        for comp in self.components:
            self._build_matrices_for_component(comp)
        # Update snapshot
        self._watch = self._snapshot_params()

    def _build_matrices_for_component(self, comp):
        """(Re)build A,B and LU factors for a component; preallocate RHS and state."""
        # dispersion & transport coefficients
        disperCoef = self.length * self.flowSpeed / comp.delta
        denom = (((1 - self.porosity) * comp.henryConst) / self.porosity) + 1.0
        C1 = disperCoef / denom
        C2 = self.flowSpeed / denom

        Nx = self.Nx; dx = self.dx; dt = self.dt
        # Build full A and B once (tridiagonal)
        A = np.zeros((Nx, Nx))
        B = np.zeros((Nx, Nx))

        # --- Danckwerts/Neumann BC (full implicit, bez 1/(2dx) oscilačního vážení) ---
        # Inlet (z=0):  -D*(c1 - c0)/dx + u*(c0 - c_in) = 0  →  (u/D + 1/dx)*c0 - (1/dx)*c1 = (u/D)*c_in
        A[0, 0] = self.flowSpeed / disperCoef + 1.0 / dx
        A[0, 1] = -1.0 / dx
        B[0, 0] = 0.0
        B[0, 1] = 0.0

        # Outlet (z=L):  ∂c/∂z = 0  →  -(1/dx)*c_{N-2} + (1/dx)*c_{N-1} = 0
        A[Nx - 1, Nx - 2] = -1.0 / dx
        A[Nx - 1, Nx - 1] =  1.0 / dx
        B[Nx - 1, Nx - 2] = 0.0
        B[Nx - 1, Nx - 1] = 0.0


        # Interior (dispersion central CN; advection: central for low Pe, upwind for high Pe)
        Pe_cell = abs(C2) * dx / max(C1, 1e-30)   # cell Peclet number

        if Pe_cell <= 2.0:
            # --- původní CENTRÁLNÍ advekce (váš dosavadní kód) ---
            aL = -((dt * C1) / (2.0 * dx**2)) - ((dt * C2) / (4.0 * dx))
            aC =  1.0 + ((dt * C1) / (dx**2))
            aU = -((dt * C1) / (2.0 * dx**2)) + ((dt * C2) / (4.0 * dx))

            bL =  ((dt * C1) / (2.0 * dx**2)) + ((dt * C2) / (4.0 * dx))
            bC =  1.0 - ((dt * C1) / (dx**2))
            bU =  ((dt * C1) / (2.0 * dx**2)) - ((dt * C2) / (4.0 * dx))
        else:
            # --- UPWIND advekce (pro kladný průtok → backward differencing) ---
            # advection part: -v * (dc/dx) ≈ -(v/dx)*c_i + (v/dx)*c_{i-1}
            # LHS (n+1) s koef. dt/2 → přidá + (dt*v)/(2dx) na dolní a - (dt*v)/(2dx) na diagonálu
            aL = -((dt * C1) / (2.0 * dx**2)) + ((dt * C2) / (2.0 * dx))
            aC =  1.0 + ((dt * C1) / (dx**2))    - ((dt * C2) / (2.0 * dx))
            aU = -((dt * C1) / (2.0 * dx**2))    # advekce nezasahuje do horní sousední

            # RHS (n) jde se znaménkem opačně → + (dt*v)/(2dx) na diagonálu a - na dolní
            bL =  ((dt * C1) / (2.0 * dx**2))    - ((dt * C2) / (2.0 * dx))
            bC =  1.0 - ((dt * C1) / (dx**2))    + ((dt * C2) / (2.0 * dx))
            bU =  ((dt * C1) / (2.0 * dx**2))

        # vectorized fill
        idx = np.arange(1, Nx-1)
        A[idx, idx-1] = aL
        A[idx, idx]   = aC
        A[idx, idx+1] = aU

        B[idx, idx-1] = bL
        B[idx, idx]   = bC
        B[idx, idx+1] = bU

        # LU factorization of A (once per change)
        comp._lu = lu_factor(A)
        comp._B = B
        # Preallocate RHS buffer and keep state vector
        if not hasattr(comp, "c") or comp.c.shape != (Nx,):
            comp.c = np.zeros(Nx, dtype=float)
        comp._b = np.empty(Nx, dtype=float)
        # Cache some scalars for the inlet BC
        comp._inlet_coef = self.flowSpeed / disperCoef

    # ---------------- time step ----------------

    def step(self, cins):
        """One CN step; rebuilds numerics only if params changed; reuses LU otherwise."""
        self._refresh_if_needed()

        output = []
        for comp, cin in zip(self.components, cins):
            # b = B @ c
            np.dot(comp._B, comp.c, out=comp._b)
            # inlet BC
            comp._b[0] += comp._inlet_coef * cin
            # solve A c^{n+1} = b
            comp.c = lu_solve(comp._lu, comp._b)
            output.append(comp.c)  # keep as ndarray (no list conversion)

        return output

    def deepCopy(self):
        """Deep copy including cached factors (safe to reuse)."""
        copy = LinColumn(self.length, self.diameter, self.porosity)
        copy.flowRate = self.flowRate
        copy.Nx = self.Nx
        copy.dt = self.dt
        copy.flowSpeed = self.flowSpeed
        copy.dx = self.dx
        copy.components = [comp.copy() for comp in self.components]
        copy._tol_rel = self._tol_rel
        copy._tol_abs = self._tol_abs

        # rebuild numerics for the copy (factors are cheap relative to correctness here)
        for comp, cpy in zip(self.components, copy.components):
            copy._build_matrices_for_component(cpy)
            cpy.c = np.copy(comp.c)

        copy._watch = copy._snapshot_params()
        return copy
