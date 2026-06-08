import numpy as np
import math
from scipy import linalg
from .GenericColumn import GenericColumn

class LinColumn(GenericColumn):
    """Chromatography column using the Equilibrium Dispersive Model (EDM) with a linear isotherm.
    Uses Crank-Nicolson implicit method with banded matrix solver.
    """

    def __init__(self, length, diameter, porosity):
        """Initialize LinColumn with column properties."""
        super().__init__(length, diameter, porosity)
        self.columnType = "EDM with Linear Isotherm"

    def init(self, flowRate, dt, Nx):
        """Initialize numerical parameters and matrices for the Crank-Nicolson solver."""
        self.flowRate = flowRate
        self.dt = dt
        self.Nx = Nx

        # Compute flow speed based on column properties
        self.flowSpeed = (flowRate * 1000 / 3600) / ((math.pi * (self.diameter ** 2) / 4) * self.porosity)
        self.dx = self.length / self.Nx  # Spatial step


        if not self.components:
            raise ValueError("LinColumn has no components! Add at least one component before initialization.")
        # Initialize matrices for each component
        for comp in self.components:
            # Compute dispersion coefficient dynamically
            disperCoef = self.length * self.flowSpeed / comp.delta
            print(disperCoef)

            # Precompute constants for numerical scheme
            C1 = disperCoef / ((((1 - self.porosity) * comp.henryConst) / self.porosity) + 1)
            C2 = self.flowSpeed / ((((1 - self.porosity) * comp.henryConst) / self.porosity) + 1)

            # Initialize tridiagonal matrices A and B
            A = np.zeros((self.Nx, self.Nx))
            B = np.zeros((self.Nx, self.Nx))

            # Apply Danckwert’s boundary condition using a fictitious point
            A[0, 0] = self.flowSpeed / disperCoef + 1 / (2 * self.dx)
            A[0, 1] = -1 / (2 * self.dx)
            A[self.Nx - 1, self.Nx - 2] = -1 / (2 * self.dx)
            A[self.Nx - 1, self.Nx - 1] = 1 / (2 * self.dx)

            B[0, 0] = -1 / (2 * self.dx)
            B[0, 1] = 1 / (2 * self.dx)
            B[self.Nx - 1, self.Nx - 2] = 1 / (2 * self.dx)
            B[self.Nx - 1, self.Nx - 1] = -1 / (2 * self.dx)

            # Populate interior points
            for i in range(1, self.Nx - 1):
                A[i, i - 1] = -((self.dt * C1) / (2 * self.dx**2)) - ((self.dt * C2) / (4 * self.dx))
                A[i, i] = 1 + ((self.dt * C1) / self.dx**2)
                A[i, i + 1] = -((self.dt * C1) / (2 * self.dx**2)) + ((self.dt * C2) / (4 * self.dx))

                B[i, i - 1] = ((self.dt * C1) / (2 * self.dx**2)) + ((self.dt * C2) / (4 * self.dx))
                B[i, i] = 1 - ((self.dt * C1) / self.dx**2)
                B[i, i + 1] = ((self.dt * C1) / (2 * self.dx**2)) - ((self.dt * C2) / (4 * self.dx))

            # Convert A into a banded matrix format for efficient solving
            comp.A_diag = self.diagonal_form(A)
            comp.B = B

            # Initialize concentration array if not already present
            if not hasattr(comp, 'c'):
                comp.c = np.zeros(self.Nx)

    def step(self, cins):
        """Perform one time step using Crank-Nicolson solver with safe optimizations (same numerical behavior)."""

        output = []

        for comp, cin in zip(self.components, cins):

            # Use local variables for faster access (reduces attribute lookup time)
            A_diag = comp.A_diag
            B = comp.B
            c = comp.c
            flowSpeed = self.flowSpeed
            Di = comp.Di
            disperCoef = self.length * self.flowSpeed / comp.delta

            # Construct right-hand side vector
            b = np.dot(B, c)

            b[0] += flowSpeed / disperCoef * cin  # Apply inlet boundary condition

            # Solve the banded system (ensuring no changes to numerical behavior)
            comp.c = linalg.solve_banded((1, 1), A_diag, b)

            output.append(comp.c.tolist())

        return output

    def diagonal_form(self, A):
        """Convert a tridiagonal matrix into the format required by `solve_banded()`."""
        ab = np.zeros((3, A.shape[1]))
        ab[1, :] = np.diag(A)  # Main diagonal
        ab[0, 1:] = np.diag(A, k=1)  # Upper diagonal
        ab[2, :-1] = np.diag(A, k=-1)  # Lower diagonal
        return ab

    def deepCopy(self):
        """Create a deep copy of the LinColumn object."""
        copy = LinColumn(self.length, self.diameter, self.porosity)
        copy.flowRate = self.flowRate
        copy.Nx = self.Nx
        copy.dt = self.dt
        copy.flowSpeed = self.flowSpeed
        copy.dx = self.dx
        copy.components = [comp.copy() for comp in self.components]

        for comp, copycomp in zip(self.components, copy.components):
            copycomp.A_diag = np.copy(comp.A_diag)
            copycomp.B = np.copy(comp.B)
            copycomp.c = np.copy(comp.c)

        return copy
