import numpy as np
import math
from scipy import optimize
from .GenericColumn import GenericColumn

class NonLinColumn(GenericColumn):
    """Implementation of column using equilibrium dispersion model with Langmuir isotherm
    Inherites from GenericColumn abstract class.
    """
    def __init__(self, length, diameter, porosity):
        """Initialize NonlinColumn object by calling the parent class constructor and updating the column type"""
        GenericColumn.__init__(self, length, diameter, porosity)
        self.columnType = "EDM with Noncompetetive Langmuir isotherm"

    def init(self, flowRate, dt, Nx):
        """Initialize the NonlinColumn with given flow rate, time step, and number of elements"""
        self.flowRate = flowRate
        self.Nx = Nx
        self.dt = dt
        self._f = np.zeros(self.Nx, dtype=float)

        # Calculate flow speed
        self.flowSpeed = (self.flowRate * 1000 / 3600) / (math.pi * ((self.diameter / 2) ** 2) * self.porosity)

        # Create spatial grid
        self.x = np.linspace(0, self.length, self.Nx)
        self.dx = self.length/self.Nx # Calculating space step [mm]

        # Initialize component concentrations
        for comp in self.components:
            if not hasattr(comp, 'c'):
                comp.c = np.zeros(len(self.x))

    def step(self, cins):
        """Perform a step in the NonlinColumn by solving the system of equations"""
        output = []
        L = self.length         # [mm]
        u = self.flowSpeed      # [mm/s] (updated in init() after each switch)

        for comp, cin in zip(self.components, cins):
            # --- parameter & inlet validation ---
            if comp.delta <= 0 or comp.Di < 0:
                raise ValueError(f"NonLinColumn: invalid dispersion params for '{comp.name}': delta={comp.delta}, Di={comp.Di}")
            K  = float(comp.langmuirConst)
            qm = float(comp.saturCoef)
            if not np.isfinite(K) or not np.isfinite(qm) or K <= 0.0 or qm <= 0.0:
                raise ValueError(f"NonLinColumn: K_L and q_m must be > 0 (got K_L={K}, q_m={qm})")
            cin = float(cin)
            if not np.isfinite(cin):
                cin = 0.0

            # Axial dispersion from A (=delta) and B (=Di): Dax [mm^2/s]
            Dax = (L * u) / comp.delta + comp.Di
            if not np.isfinite(Dax) or Dax < 0.0:
                raise ValueError(f"NonLinColumn: invalid D_ax={Dax}")

            # --- bounded nonlinear least-squares (robust to NaN/Inf via bounds) ---
            x0 = comp.c
            # dynamic upper bound: keep x within a realistic envelope
            ub_scalar = max(1.0e-6 + cin, float(np.max(x0)) + 1.0) * 10.0
            lb = np.zeros_like(x0)                  # c >= 0
            ub = np.full_like(x0, ub_scalar)       # c <= ub_scalar

            lsq = optimize.least_squares(
                fun=self.function,
                x0=x0,
                bounds=(lb, ub),
                args=(x0, cin, self.porosity, K, qm, Dax, u),
                method='trf',
                ftol=1e-6, xtol=1e-6, gtol=1e-6,
                max_nfev=100
            )
            x = lsq.x

            # fallback: try a slightly larger envelope if it struggled
            if (not lsq.success) or (not np.all(np.isfinite(x))):
                ub_scalar *= 5.0
                ub = np.full_like(x0, ub_scalar)
                # better initial guess: inject inlet at first node
                x0b = x0.copy(); x0b[0] = max(x0b[0], cin)
                lsq2 = optimize.least_squares(
                    fun=self.function,
                    x0=x0b,
                    bounds=(lb, ub),
                    args=(x0b, cin, self.porosity, K, qm, Dax, u),
                    method='trf',
                    ftol=1e-7, xtol=1e-7, gtol=1e-7,
                    max_nfev=300
                )
                x = lsq2.x if lsq2.success and np.all(np.isfinite(lsq2.x)) else x0b

            # keep physical & commit
            x = np.nan_to_num(x, nan=0.0, posinf=ub_scalar, neginf=0.0)
            x = np.clip(x, 0.0, ub_scalar)
            comp.c = x
            output.append(comp.c.tolist())

        return output


    def function(self, c1, c0, feedCur, porosity, langmuirConst, saturCoef, disperCoef, flowSpeed):
        """
        Vectorized residual:
          (D/den0 * c0'' + D/den1 * c1'')/2  -  (u/den0 * c0' + u/den1 * c1')/2  -  (c1 - c0)/dt  = 0
        Returns a fresh array; internally clamps states to keep numerics finite.
        """
        # --- sanitize & clip inputs ---
        dx  = float(self.dx);  dt = float(self.dt)
        u   = float(flowSpeed); D = float(disperCoef)
        K   = float(langmuirConst); qm = float(saturCoef); eps = float(porosity)
        fc  = float(feedCur)
        if not np.isfinite(fc): fc = 0.0

        # clip current iterate to a safe envelope (prevents NaN/Inf inside MINPACK math)
        c0 = np.asarray(c0, dtype=float)
        c1 = np.asarray(c1, dtype=float)
        ub_scalar = max(1.0e-6 + fc, float(np.max(c0)) + 1.0) * 10.0
        c1v = np.clip(np.nan_to_num(c1, nan=0.0, posinf=ub_scalar, neginf=0.0), 0.0, ub_scalar)
        c0v = np.clip(np.nan_to_num(c0, nan=0.0, posinf=ub_scalar, neginf=0.0), 0.0, ub_scalar)

        f = np.zeros_like(c0v)

        # Left boundary (Dirichlet-like with feed at inlet)
        f[0] = 0.5 * ((c0v[1] - c0v[0]) / dx + (c1v[1] - c1v[0]) / dx) - u * (c1v[0] - fc)

        # Interior slices
        c0m1, c0i, c0p1 = c0v[:-2], c0v[1:-1], c0v[2:]
        c1m1, c1i, c1p1 = c1v[:-2], c1v[1:-1], c1v[2:]

        # Langmuir denominators at t^n (c0) and t^{n+1} (c1)
        num  = (1.0 - eps) * qm * K
        den0 = num / (((( -K * c0i + 1.0 )**2) * eps) + 1.0)
        den1 = num / (((( -K * c1i + 1.0 )**2) * eps) + 1.0)
        den_min = 1e-12
        den0 = np.maximum(den0, den_min)
        den1 = np.maximum(den1, den_min)

        inv_dx  = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx

        # derivatives
        c0_second = (c0m1 - 2.0*c0i + c0p1) * inv_dx2
        c1_second = (c1m1 - 2.0*c1i + c1p1) * inv_dx2
        c0_first  = (c0p1 - c0m1) * (0.5 * inv_dx)
        c1_first  = (c1p1 - c1m1) * (0.5 * inv_dx)

        # Crank–Nicolson terms
        disper = 0.5 * ((D/den0) * c0_second + (D/den1) * c1_second)
        conv   = 0.5 * ((u/den0) * c0_first  + (u/den1) * c1_first)
        time   = (c1i - c0i) / dt

        f[1:-1] = disper - conv - time

        # Right boundary (Neumann 0-gradient)
        f[-1] = 0.5 * ((c0v[-1]-c0v[-2])/dx + (c1v[-1]-c1v[-2])/dx)

        # final sanitize – never return NaN/Inf
        return np.nan_to_num(f, nan=1e8, posinf=1e8, neginf=-1e8)


    def deepCopy(self):
        """Create a deep copy of the NonLinColumn object"""
        copy = NonLinColumn(self.length, self.diameter, self.porosity)
        copy.columnType = self.columnType
        copy.flowRate = self.flowRate
        copy.Nx = self.Nx
        copy.dt = self.dt
        copy.flowSpeed = self.flowSpeed
        copy.x = self.x
        copy.dx = self.dx

        # Copy component data
        copy.components = [comp.copy() for comp in self.components]
        for comp, copycomp in zip(self.components, copy.components):
            copycomp.C1 = comp.C1
            copycomp.C2 = comp.C2
            copycomp.A = np.copy(comp.A)
            copycomp.B = np.copy(comp.B)
            copycomp.A_diag = np.copy(comp.A_diag)
            copycomp.Aabs = np.copy(comp.Aabs)
            copycomp.Babs = np.copy(comp.Babs)
            copycomp.c = np.copy(comp.c)

        return copy

    def _qstar_competitive(self, K_vec, qm_vec, C):
        """
        Extended competitive Langmuir with per-species capacities:
          q_i* = (q_m,i * K_i * c_i) / (1 + sum_j K_j c_j)
        C shape: (Ns, Nx). Returns Q shape: (Ns, Nx).
        """
        S = 1.0 + np.sum(K_vec[:, None] * C, axis=0)
        S = np.maximum(S, 1e-12)  # floor to avoid division spikes
        return (qm_vec[:, None] * K_vec[:, None] * C) / S

    def _stack_components(self):
        """Return stacked current states (c0) with shape (Ns, Nx) and the species count."""
        Ns = len(self.components)
        C0 = np.vstack([comp.c for comp in self.components])  # (Ns, Nx)
        return C0, Ns

    def _residual_competitive(self, Y1_flat, Y0, cin_vec, eps, K_vec, qm_vec, Dax_vec, u):
        """
        Residual for competitive Langmuir, stacked species.
        Inputs:
          Y1_flat : (Ns*Nx,) next-step concentrations (flattened)
          Y0      : (Ns, Nx)  current concentrations
          cin_vec : (Ns,)     inlet concentrations for species at x=0 (mixed)
          K_vec   : (Ns,)     Langmuir K_i
          Dax_vec : (Ns,)     axial dispersion for each species [mm^2/s]
        Returns:
          r_flat  : (Ns*Nx,)  residual vector
        """
        Ns, Nx = Y0.shape
        Y1 = Y1_flat.reshape(Ns, Nx)

        dx = self.dx
        dt = self.dt

        # storage at n and n+1 (competitive)
        Q0 = self._qstar_competitive(K_vec, qm_vec, Y0)  # (Ns, Nx)
        Q1 = self._qstar_competitive(K_vec, qm_vec, Y1)  # (Ns, Nx)

        # spatial derivatives per species
        inv_dx  = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx
        half    = 0.5

        R = np.zeros_like(Y1)

        for s in range(Ns):
            c0 = Y0[s]; c1 = Y1[s]
            D  = float(Dax_vec[s])

            # left boundary (Dirichlet-like with mixed inlet)
            R[s, 0] = half * (((c0[1] - c0[0]) * inv_dx) + ((c1[1] - c1[0]) * inv_dx)) - u * (c1[0] - float(cin_vec[s]))

            # interior nodes
            c0m1 = c0[:-2]; c0i = c0[1:-1]; c0p1 = c0[2:]
            c1m1 = c1[:-2]; c1i = c1[1:-1]; c1p1 = c1[2:]

            c0_second = (c0m1 - 2.0*c0i + c0p1) * inv_dx2
            c1_second = (c1m1 - 2.0*c1i + c1p1) * inv_dx2
            c0_first  = (c0p1 - c0m1) * (0.5 * inv_dx)
            c1_first  = (c1p1 - c1m1) * (0.5 * inv_dx)

            disper = half * (D * (c0_second + c1_second))
            conv   = half * (u * (c0_first  + c1_first))

            # storage difference (this is where coupling enters)
            stor1 = eps * c1i + (1.0 - eps) * Q1[s, 1:-1]
            stor0 = eps * c0i + (1.0 - eps) * Q0[s, 1:-1]
            time_term = (stor1 - stor0) / dt

            R[s, 1:-1] = disper - conv - time_term

            # right boundary (zero gradient)
            R[s, -1] = half * (((c0[-1] - c0[-2]) * inv_dx) + ((c1[-1] - c1[-2]) * inv_dx))

        # sanitize (never return NaN/Inf)
        R = np.nan_to_num(R, nan=1e8, posinf=1e8, neginf=-1e8)
        return R.reshape(-1)

    def step_competitive(self, cins):
        """
        Advance one dt with competitive Langmuir (coupled species).
        Uses bounded least-squares on the stacked vector (Ns*Nx).
        """
        # prepare stacked current state and parameters
        C0, Ns = self._stack_components()
        Nx = self.Nx
        L  = self.length
        u  = self.flowSpeed
        eps = self.porosity

        # require shared q_m across species (typical for competitive Langmuir)
        qm_vec = np.array([float(comp.saturCoef) for comp in self.components], dtype=float)
        if np.any(~np.isfinite(qm_vec)) or np.any(qm_vec <= 0):
            raise ValueError(f"Competitive Langmuir: all q_m,i must be > 0 (got {qm_vec})")

        K_vec   = np.array([float(comp.langmuirConst) for comp in self.components], dtype=float)
        if np.any(~np.isfinite(K_vec)) or np.any(K_vec <= 0):
            raise ValueError(f"Competitive Langmuir: all K_i must be > 0 (got {K_vec})")

        delta_vec = np.array([float(comp.delta) for comp in self.components], dtype=float)
        Di_vec    = np.array([float(comp.Di)    for comp in self.components], dtype=float)
        if np.any(delta_vec <= 0) or np.any(Di_vec < 0):
            raise ValueError("Competitive Langmuir: delta>0 and Di>=0 required")

        Dax_vec = (L * u) / delta_vec + Di_vec   # per-species dispersion [mm^2/s]

        cin_vec = np.array([float(v) if np.isfinite(v) else 0.0 for v in cins], dtype=float)

        # bounds for stacked vector (non-negative)
        x0  = C0.reshape(-1)
        ub_scalar = max(1e-6 + float(np.max(cin_vec)), float(np.max(x0)) + 1.0) * 10.0
        lb = np.zeros_like(x0)
        ub = np.full_like(x0, ub_scalar)

        # solve
        lsq = optimize.least_squares(
            fun=self._residual_competitive,
            x0=x0,
            bounds=(lb, ub),
            args=(C0, cin_vec, eps, K_vec, qm_vec, Dax_vec, u),  # <-- qm_vec here
            method='trf',
            ftol=1e-6, xtol=1e-6, gtol=1e-6,
            max_nfev=120
        )
        X = lsq.x
        if (not lsq.success) or (not np.all(np.isfinite(X))):
            # larger envelope + inlet bootstrap
            ub = np.full_like(x0, ub_scalar * 5.0)
            x0b = x0.copy()
            # inject inlets at left boundary nodes
            for s in range(Ns):
                x0b[s*Nx + 0] = max(x0b[s*Nx + 0], cin_vec[s])
                lsq2 = optimize.least_squares(
                    fun=self._residual_competitive,
                    x0=x0b, bounds=(lb, ub),
                    args=(C0, cin_vec, eps, K_vec, qm_vec, Dax_vec, u),  # <-- qm_vec here
                    method='trf',
                    ftol=1e-7, xtol=1e-7, gtol=1e-7,
                    max_nfev=300
                )
            X = lsq2.x if lsq2.success and np.all(np.isfinite(lsq2.x)) else x0b

        # commit back to components (non-negative)
        X = np.clip(np.nan_to_num(X, nan=0.0, posinf=ub_scalar, neginf=0.0), 0.0, ub_scalar)
        C1 = X.reshape(Ns, Nx)
        for s, comp in enumerate(self.components):
            comp.c = C1[s].copy()

        # outputs per species (profiles)
        return [self.components[s].c.tolist() for s in range(Ns)]
