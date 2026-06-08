
class GenericColumn:
    """Abstract class from which all column classes should inherit.
    Contains:
    henryConst (float) - Henry constant parameter of EDM with linear isotherm.
    langmuirConst (float) - Langmuir constant parameter of EDM with langmuir isotherm.
    disperCoef (float) - Dispersion coeficient parameter of EDM.
    saturCoef (float) - Saturation coeficient parameter of EDM with langmuir isotherm.
    feedConc (float) - Feed concentration parameter of component.
    """
    def __init__(self, length, diameter, porosity):
        """Initialize GenericColumn object with given parameters"""
        self.length = length
        self.diameter = diameter
        self.porosity = porosity
        self.columnType = "Generic column"
        self.components = []  # Initialize an empty list to store components

    def add(self, comp):
        """Add a component to the column's components list"""
        self.components.append(comp)

    def delByIdx(self, idx):
        """Delete a component from the column's components list at the given index"""
        del self.components[idx]

    def updateByIdx(self, idx, comp):
        """Update a component in the column's components list at the given index with a new component"""
        self.components[idx].update(comp)

    def init(self, flowRate, dt, Nx):
        """Abstract method, initialize the column with given flow rate, time step, and number of elements"""
        raise NotImplementedError("Called abstract method in GenericColumn class.")

    def step(self, cins):
        """Abstract method, perform a step in the column's simulation based on given inputs"""
        raise NotImplementedError("Called abstract method in GenericColumn class.")

    def getInfo(self):
        """Return a dictionary containing information about the column"""
        info = {}
        info["Column Type"] = self.columnType
        info["Length"] = self.length
        info["Diameter"] = self.diameter
        info["Porority"] = self.porosity
        return info

    def deepCopy(self):
        """Create a deep copy of the GenericColumn object"""
        copy = GenericColumn(self.length, self.diameter, self.porosity)
        copy.columnType = self.columnType
        return copy