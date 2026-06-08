import numpy as np
import math
from scipy import sparse
from .GenericColumn import GenericColumn

class LinColumn(GenericColumn):
    """Implementation of column using equilibrium dispersion model with linear isotherm.
    Inherites from GenericColumn abstract class.
    """
    def __init__(self, length, diameter, porosity):
        """Initialize LinColumn object by calling the parent class constructor and updating the column type"""
        GenericColumn.__init__(self, length, diameter, porosity)
        self.columnType = "EDM with Linear isotherm"

    def init(self, flowRate, dt, Nx):
        """Initialize the LinColumn with given flow rate, time step, and number of elements"""
        self.flowRate = flowRate
        self.Nx = Nx
        self.dt = dt
        # self.flowSpeed = (self.flowRate * 1000 / 3600) / (math.pi * ((self.diameter / 2) ** 2) * self.porosity)
        self.flowSpeed = (self.flowRate * 1000/3600) / ((math.pi * (self.diameter**2) / 4) * self.porosity)
        self.x = np.linspace(0, self.length, self.Nx)
        self.dx = self.length / self.Nx  # Calculating space step [mm]

        for comp in self.components:

            # Crank-Nicolson matrices preparation
            # A.c(t+1) = B.c(t)x, where c(t+1) and c(t) are vectors of c(x) values
            # Preparation of boundaries in matrix A
            comp.A = np.zeros((self.Nx, self.Nx))  # A matrix data structure
            comp.A[0, 0] = self.flowSpeed/self.disperCoef + 1/(2*self.dx)  # Left boundary
            comp.A[0, 1] = -1/(2*self.dx)  # Left boundary
            comp.A[self.Nx - 1, self.Nx - 2] = -1/(2*self.dx)  # Right boundary
            comp.A[self.Nx - 1, self.Nx - 1] = 1/(2*self.dx)  # Right Boundary
            # Preparation of boundaries in matrix B
            comp.B = np.zeros((Nx, Nx))  # B matrix data structure
            comp.B[0, 0] = -1/(2*self.dx)  # Left boundary
            comp.B[0, 1] = 1/(2*self.dx)  # Left boundary
            comp.B[self.Nx - 1, self.Nx - 2] = 1/(2*self.dx)  # Right boundary
            comp.B[self.Nx - 1, self.Nx - 1] = -1/(2*self.dx)  # Right Boundary

             # Defining constants
            comp.C1 = self.disperCoef/((((1-self.porosity)*self.henryConst)/self.porosity)+1)  # *** !!! PODLE DOKUMENTU
            comp.C2 = self.flowSpeed/((((1-self.porosity)*self.henryConst)/self.porosity)+1)  # *** !!
            # Filling up Matrixes A and B
            for i in range(1, self.Nx - 1):
                comp.A[i, i - 1] = -((dt*comp.C1)/(2*(self.dx**2)))-((self.dt*comp.C2)/(4*self.dx))
                comp.A[i, i] = 1 + ((self.dt*comp.C1)/(self.dx**2))
                comp.A[i, i + 1] = -((self.dt*comp.C1)/(2*(self.dx**2)))+((self.dt*comp.C2)/(4*self.dx))
                comp.B[i, i - 1] = ((self.dt*comp.C1)/(2*(self.dx**2)))+((self.dt*comp.C2)/(4*self.dx))
                comp.B[i, i] = 1 - ((self.dt*comp.C1)/(self.dx**2))
                comp.B[i, i + 1] = ((self.dt*comp.C1)/(2*(self.dx**2)))-((dt*comp.C2)/(4*self.dx))

            comp.A_diag = self.diagonal_form(comp.A)
            comp.Aabs = np.abs(comp.A)
            comp.Babs = np.abs(comp.B)

            # Initializes result matrix in component with zeroes if not yet present
            if not hasattr(comp, 'c'):
                comp.c = np.zeros(len(self.x))

    def step(self, cins):
        """Perform a step in the LinColumn by solving the system of equations"""
        output = []
        for comp, cin in zip(self.components, cins):
            b = comp.B.dot(comp.c)
            b[0] = b[0] - (self.flowSpeed * 2 * self.dx * cin)  # From left boundary
            comp.c = sparse.linalg.splu(comp.A).solve(b)
            output.append(comp.c.tolist())
        return output

    def diagonal_form(self, a, lower=1, upper=1):
        """Transforms banded matrix into diagonal ordered form
           allows using scipy.linalg.solve_banded
        """
        n = a.shape[1]
        assert (np.all(a.shape == (n, n)))
        ab = np.zeros((2 * n - 1, n))
        for i in range(n):
            ab[i, (n - 1) - i:] = np.diagonal(a, (n - 1) - i)
        for i in range(n - 1):
            ab[(2 * n - 2) - i, :i + 1] = np.diagonal(a, i - (n - 1))
        mid_row_inx = int(ab.shape[0] / 2)
        upper_rows = [mid_row_inx - i for i in range(1, upper + 1)]
        upper_rows.reverse()
        upper_rows.append(mid_row_inx)
        lower_rows = [mid_row_inx + i for i in range(1, lower + 1)]
        keep_rows = upper_rows + lower_rows
        ab = ab[keep_rows, :]
        return ab

    def deepCopy(self):
        """Create a deep copy of the LinColumn object"""
        copy = LinColumn(self.length, self.diameter, self.porosity)
        copy.columnType = self.columnType
        copy.flowRate = self.flowRate
        copy.Nx = self.Nx
        copy.dt = self.dt
        copy.flowSpeed = self.flowSpeed
        copy.x = self.x
        copy.dx = self.dx
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
