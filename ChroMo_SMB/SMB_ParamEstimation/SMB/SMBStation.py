from .GenericColumn import GenericColumn
from .Component import Component
import copy as cp
import numpy as np
import math
import time

class SMBStation:
    """Class representing SMB Station
       Contains Columns, Tubes and Components
    """
    def __init__(self):
        """Initialize attributes"""
        self.zones = {}  # Dictionary to store zones and their columns
        self.zones[1] = []  # Columns in zone 1
        self.zones[2] = []  # Columns in zone 2
        self.zones[3] = []  # Columns in zone 3
        self.zones[4] = []  # Columns in zone 4

        self.cins = {}  # Dictionary to store inlet concentrations for each zone
        self.cins[1] = []  # Inlet concentrations for zone 1
        self.cins[2] = []  # Inlet concentrations for zone 2
        self.cins[3] = []  # Inlet concentrations for zone 3
        self.cins[4] = []  # Inlet concentrations for zone 4

        self.flowRates = {}  # Dictionary to store flow rates for each zone
        self.flowRates[1] = -1  # Flow rate for zone 1
        self.flowRates[2] = -1  # Flow rate for zone 2
        self.flowRates[3] = -1  # Flow rate for zone 3
        self.flowRates[4] = -1  # Flow rate for zone 4

        self.outVals = []  # List to store outlet concentrations
        self.settings = {}  # Dictionary to store simulation settings
        self.components = []  # List to store components
        self.switchingEnabled = False  # Flag to indicate if column rotation is enabled
        self.interval = -1  # Switch interval for column rotation
        self.countdown = -1  # Countdown timer for column rotation
        self.timer = 0  # Timer for simulation
        self.colCount = 0  # Number of columns
        self.switchState = 0  # Current state of column rotation
        self.dt = 0.0

        self.isotherm_mode = 'noncomp'

    def set_isotherm_mode(self, mode: str):
        """
        Set isotherm mode for all columns.
        mode ∈ {'noncomp', 'competitive'}
        """
        if mode not in ('noncomp', 'competitive'):
            raise ValueError(f"Invalid isotherm mode: {mode}")
        self.isotherm_mode = mode

    def setFlowRateZone(self, zone, flowRate):
        """Set flow rate for a specific zone"""
        self.flowRates[zone] = flowRate

    def setSwitchInterval(self, s):
        """Set switch interval for column rotation"""
        self.interval = s
        self.countdown = s
        if s <= 0:
            self.switchingEnabled = False
        else:
            self.switchingEnabled = True

    def setdt(self, dt):
        """Set the time step for the simulation"""
        self.settings['dt'] = dt
        self.dt = dt

    def setNx(self, Nx):
        """Set the number of grid points for the simulation"""
        self.settings['Nx'] = Nx

    def addColZone(self, zone, col, tube):
        """Add a column and a tube before it to a specific zone"""
        for comp in self.components:
            col.add(comp.copy())
            tube.add(comp.copy())
        self.zones[zone].append(tube)
        self.zones[zone].append(col)
        self.cins[zone].append([])
        self.cins[zone].append([])
        self.colCount += 1

    def delColZone(self, zone, idx):
        """Delete a column and a tube before it from a specific zone"""
        del self.zones[zone][idx]
        if idx%2 == 1:
            del self.zones[zone][idx-1]
        elif idx%2 == 0:
            del self.zones[zone][idx]
        self.colCount -= 1

    def createComponent(self, name, feedConc = 0, henryConst = -1, disperCoef = -1, langmuirConst = -1, saturCoef = -1):
        """Create a component and add it to the SMB station"""
        comp = Component(name)
        comp.feedConc = feedConc
        comp.henryConst = henryConst
        comp.disperCoef = disperCoef
        comp.langmuirConst = langmuirConst
        comp.saturCoef = saturCoef
        self.components.append(comp)
        for zone in self.zones:
            for col in self.zones[zone]:
                col.add(comp.copy())

    def createComponentAB(self, name, feedConc = 0, henryConst = -1, A = -1, B = -1):
        """Create a component and add it to the SMB station"""
        self.disperCalc = True
        for zone in self.zones:
            for col in self.zones[zone]:
                if isinstance(col, GenericColumn):
                    comp = Component(name)
                    comp.feedConc = feedConc
                    comp.henryConst = henryConst
                    comp.delta = A
                    comp.Di = B
                    #flowSpeed = (self.flowRates[zone] * 1000 / 3600) / ((math.pi * (col.diameter ** 2) / 4) * col.porosity)
                    #comp.disperCoef = (1/A) * col.length * flowSpeed + B
                    comp.langmuirConst = -1
                    comp.saturCoef = -1
                    self.components.append(comp)
                    col.add(comp.copy())
                else:
                    comp = Component(name)
                    comp.feedConc = feedConc
                    col.add(comp.copy())

    def delComponent(self, idx):
        # Delete a component from the SMB station
        del self.components[idx]
        for zone in self.zones:
            for col in self.zones[zone]:
                col.delByIdx(idx)

    def updateComponentByName(self, name, feedConc = 0, henryConst = -1, disperCoef = -1, langmuirConst = -1, saturCoef = -1):
        """Update the properties of a component based on its name"""
        # Finds component with given name and updates it
        for idx, comp in enumerate(self.components):
            if comp.name == name:
                if feedConc > 0:
                    comp.feedConc = feedConc
                if henryConst > 0:
                    comp.henryConst = henryConst
                if disperCoef > 0:
                    comp.disperCoef = disperCoef
                if langmuirConst > 0:
                    comp.langmuirConst = langmuirConst
                if saturCoef > 0:
                    comp.saturCoef = saturCoef
                for zone in self.zones:
                    for col in self.zones[zone]:
                        col.updateByIdx(idx, comp)
                return
        # Creates component if it doesn't exists
        comp = Component(name)
        comp.feedConc = feedConc
        comp.henryConst = henryConst
        comp.disperCoef = disperCoef
        comp.langmuirConst = langmuirConst
        comp.saturCoef = saturCoef
        self.components.append(comp)
        for zone in self.zones:
            for col in self.zones[zone]:
                col.add(comp.copy())

    def updateComponentByIndex(self, idx, feedConc = 0, henryConst = -1, disperCoef = -1, langmuirConst = -1, saturCoef = -1):
        """Update the properties of a component based on its index"""
        comp = self.components[idx]
        if feedConc > 0:
            comp.feedConc = feedConc
        if henryConst > 0:
            comp.henryConst = henryConst
        if disperCoef > 0:
            comp.disperCoef = disperCoef
        if langmuirConst > 0:
            comp.langmuirConst = langmuirConst
        if saturCoef > 0:
            comp.saturCoef = saturCoef
        for zone in self.zones:
            for col in self.zones[zone]:
                col.updateByIdx(idx, comp)

    def setPorosity(self, porosity):
        """Set the porosity of all the columns in the SMB station"""
        for zone in self.zones:
            for col in self.zones[zone]:
                if isinstance(col, GenericColumn):
                    col.porosity = porosity

    def initCols(self):
        """Initialize the columns in the SMB station"""
        for zone in self.zones:
            for idx, col in enumerate(self.zones[zone]):
                col.init(self.flowRates[zone], self.settings['dt'], self.settings['Nx'])
                self.cins[zone][idx] = [0] * len(col.components)
                self.outVals = [0] * len(col.components)

    def step(self, steps = 1):
        """Perform one or more simulation steps on all columns and connecting them"""
        '''flowRates[n-1] where n is zone number [1,2,3,4]'''
        cins = [comp.feedConc for comp in self.components]
        for x in range(steps):
            self.timer += self.settings['dt']
            res = {}
            for zone in self.zones:
                res[zone] = []
                for i, col in enumerate(self.zones[zone]):
                    inVals = self.cins[zone][i]
                    self.cins[zone][i] = self.outVals
                    self.outVals = []
                    #update_start = time.perf_counter()


                    if i == 0 and zone == 3: # Mixing of the streams before zone III (Feed inlet)
                        for idx, inVal in enumerate(inVals):
                            inVals[idx] = ((inVal * self.flowRates[2]) + (cins[idx] * (self.flowRates[3]-self.flowRates[2])))/col.flowRate
                            #inVals[idx] = ((inVal * self.flowRates[1]) + (cins[idx] * (self.flowRates[2] - self.flowRates[1]))) / col.flowRate


                    if i == 0 and zone == 1: # Mixing of the streams before zone I (Eluent inlet)
                        for idx, inVal in enumerate(inVals):
                            inVals[idx] = (inVal * self.flowRates[2])/col.flowRate
                            #inVals[idx] = (inVal * self.flowRates[3]) / col.flowRate

                    if getattr(self, 'isotherm_mode', 'noncomp') == 'competitive' and hasattr(col, 'step_competitive'):
                        out = col.step_competitive(inVals)
                    else:
                        out = col.step(inVals)
                    res[zone].append(out)

                    next_out = []
                    if out and len(out) > 0:
                        for k, arr in enumerate(out):
                            try:
                                last = float(arr[-1])
                                next_out.append(last if np.isfinite(last) else float(inVals[k]))
                            except Exception:
                                next_out.append(float(inVals[k]) if k < len(inVals) else 0.0)
                    else:
                        next_out = [float(v) for v in inVals]

                    self.outVals = next_out
            if self.switchingEnabled:
                self.countdown -= self.settings['dt']
                if self.countdown <= 0:
                    self.countdown += self.interval
                    self.rotate()
        return res

    def rotate(self):
        """Rotate the columns in the SMB station based on the switch interval"""
        tmptube = self.zones[1][0]
        tmpcol = self.zones[1][1]
        self.zones[1].pop(0)
        self.zones[1].pop(0)
        self.zones[1].append(self.zones[2][0])
        self.zones[1].append(self.zones[2][1])
        self.zones[2].pop(0)
        self.zones[2].pop(0)
        self.zones[2].append(self.zones[3][0])
        self.zones[2].append(self.zones[3][1])
        self.zones[3].pop(0)
        self.zones[3].pop(0)
        self.zones[3].append(self.zones[4][0])
        self.zones[3].append(self.zones[4][1])
        self.zones[4].pop(0)
        self.zones[4].pop(0)
        self.zones[4].append(tmptube)
        self.zones[4].append(tmpcol)
        self.initCols()
        self.switchState = (self.switchState+1)%self.colCount

    def getColInfo(self):
        """Get information about each column in each zone"""
        info = {}
        for zone in self.zones:
            info[zone] = []
            for col in self.zones[zone]:
                info[zone].append(col.getInfo())
        return info

    def getCompInfo(self):
        """Get information about each component"""
        info = {}
        for comp in self.components:
            info[comp.name] = {}
            info[comp.name]["Feed Concentration"] = comp.feedConc
            info[comp.name]["Henry Constant"] = comp.henryConst
            info[comp.name]["Langmuir Constant"] = comp.langmuirConst
            info[comp.name]["Saturation Coefficient"] = comp.saturCoef
            info[comp.name]["Dispersion Coefficient"] = comp.disperCoef
        return info

    def getZoneReady(self):
        """Check if all zones have at least one column"""
        for zone in self.zones:
            if len(self.zones[zone]) == 0:
                return False
        return True

    def getSettingsInfo(self):
        """Get information about the settings of the station"""
        info = {}
        info['Flow Rate'] = self.flowRates
        info['dt'] = self.settings['dt']
        info['Nx'] = self.settings['Nx']
        info['Switch Interval'] = self.interval
        info['Countdown'] = self.countdown
        info['timer'] = self.timer
        return info

    def deepCopy(self):
        """Create a deep copy of the SMBStation object"""
        copy = SMBStation()
        for zone in self.zones:
            for col in self.zones[zone]:
                copy.zones[zone].append(col.deepCopy())
        copy.cins = cp.deepcopy(self.cins)
        copy.flowRates = cp.deepcopy(self.flowRates)
        copy.outVals = cp.deepcopy(self.outVals)
        copy.settings = cp.deepcopy(self.settings)
        copy.components = [comp.copy() for comp in self.components]
        copy.outVals = cp.deepcopy(self.outVals)
        copy.switchingEnabled = self.switchingEnabled
        copy.timer = self.timer
        copy.colCount = self.colCount
        copy.switchState = self.switchState
        copy.interval = self.interval
        copy.countdown = self.countdown
        copy.dt = self.dt
        return copy

    def step_fast_outlets(self):
        """
        Advance the plant by one dt and return only outlet concentrations:
        (c_ex_man, c_ex_gal, c_ra_man, c_ra_gal).
        Numerics identical to step(1), just no big dict allocations.
        """
        cins = [float(comp.feedConc) for comp in self.components]
        self.timer += self.settings['dt']

        zones = self.zones
        flowRates = self.flowRates
        outVals_local = self.outVals
        self.outVals = []  # reset for this step

        for zone in (1, 2, 3, 4):
            objs = zones[zone]
            for i, col in enumerate(objs):
                inVals = self.cins[zone][i]
                # carry forward previous object's outlet as our inlet
                self.cins[zone][i] = outVals_local
                outVals_local = []

                # --- mixing nodes, sanitized ---
                if i == 0 and zone == 3:  # feed mixing
                    den = float(getattr(col, "flowRate", 0.0))
                    if not np.isfinite(den) or den <= 0.0:
                        raise RuntimeError("Zone III mixing: invalid col.flowRate (<=0 or non-finite)")
                    for idx, inVal in enumerate(inVals):
                        a = float(inVal) * flowRates[2]
                        b = float(cins[idx]) * (flowRates[3] - flowRates[2])
                        val = (a + b) / den
                        inVals[idx] = val if np.isfinite(val) else 0.0

                if i == 0 and zone == 1:  # eluent mixing
                    den = float(getattr(col, "flowRate", 0.0))
                    if not np.isfinite(den) or den <= 0.0:
                        raise RuntimeError("Zone I mixing: invalid col.flowRate (<=0 or non-finite)")
                    for idx, inVal in enumerate(inVals):
                        val = (float(inVal) * flowRates[2]) / den
                        inVals[idx] = val if np.isfinite(val) else 0.0

                # --- step the object ---
                if getattr(self, 'isotherm_mode', 'noncomp') == 'competitive' and hasattr(col, 'step_competitive'):
                    x = col.step_competitive(inVals)
                else:
                    x = col.step(inVals)

                # --- robust outlet extraction (skip empties / sanitize) ---
                next_out = []
                if x and len(x) > 0:
                    for k, arr in enumerate(x):
                        try:
                            last = float(arr[-1])
                            next_out.append(last if np.isfinite(last) else float(inVals[k]))
                        except Exception:
                            next_out.append(float(inVals[k]) if k < len(inVals) else 0.0)
                else:
                    # pass-through if object returns nothing
                    next_out = [float(v) for v in inVals]

                outVals_local.extend(next_out)
                self.outVals = list(next_out)

        # switching
        if self.switchingEnabled:
            self.countdown -= self.settings['dt']
            if self.countdown <= 0:
                self.countdown += self.interval
                self.rotate()

        # extract outlet = zone 1, last column; raffinate = zone 3, last column
        try:
            ex_last = self.zones[1][-1].components
            ra_last = self.zones[3][-1].components
            c_ex_man = float(ex_last[0].c[-1]); c_ex_gal = float(ex_last[1].c[-1])
            c_ra_man = float(ra_last[0].c[-1]); c_ra_gal = float(ra_last[1].c[-1])
            # sanitize
            vals = [c_ex_man, c_ex_gal, c_ra_man, c_ra_gal]
            vals = [v if np.isfinite(v) else 0.0 for v in vals]
            c_ex_man, c_ex_gal, c_ra_man, c_ra_gal = vals
        except Exception:
            c_ex_man = c_ex_gal = c_ra_man = c_ra_gal = 0.0

        return c_ex_man, c_ex_gal, c_ra_man, c_ra_gal


    def createNonLinComponentAB(self, name, feedConc=0, henryConst=-1, A=-1, B=-1, langmuirConst=-1, saturCoef=-1):
        """
        Non-linear EDM component (Langmuir) with dispersion via A (delta) and B (Di).
        Appends one logical component to self.components, and adds copies to each object.
        """
        base = Component(name)
        base.feedConc = feedConc
        base.henryConst = henryConst
        base.delta = A
        base.Di = B
        base.langmuirConst = langmuirConst
        base.saturCoef = saturCoef
        self.components.append(base)  # <-- append ONCE

        for zone in self.zones:
            for col in self.zones[zone]:
                if isinstance(col, GenericColumn):
                    comp_col = base.copy()
                    # NonLinColumn.step() computes Dax = (L*u)/A + B each step.
                    col.add(comp_col)
                else:
                    # tube: just carry the species; tube's own model handles transport
                    col.add(base.copy())

    def apply_zone_flows_now(self):
        # Reinitialize columns with current flow rates and keep profiles.
        # NonLinColumn.init() will recompute flowSpeed; it only creates comp.c if missing.
        self.initCols()
