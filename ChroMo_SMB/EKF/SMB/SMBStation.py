from .GenericColumn import GenericColumn
from .Component import Component
import copy as cp
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

    def createComponentAB(self, name, feedConc=0, henryConst=-1, A=-1, B=-1):
        """Create one base component (unique) and add a copy to every tube/column."""
        self.disperCalc = True

        base = Component(name)
        base.feedConc = feedConc
        base.henryConst = henryConst
        base.delta = A
        base.Di = B
        base.langmuirConst = -1
        base.saturCoef = -1

        # keep exactly one logical entry for species
        self.components.append(base)

        # add a copy to every object already in all zones
        for zone in self.zones:
            for obj in self.zones[zone]:
                obj.add(base.copy())

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

    def step(self, steps: int = 1):
        """Perform `steps`×dt with delayed ring wiring and return per-zone profiles (old API)."""
        cins_feed = [comp.feedConc for comp in self.components]
        if len(cins_feed) != len(self.zones[3][0].components):
            print(f"[SMBStation] WARNING: feed vector length {len(cins_feed)} "
                  f"!= species in objects {len(self.zones[3][0].components)}", flush=True)

        for _ in range(steps):
            self.timer += self.settings['dt']
            res = {1: [], 2: [], 3: [], 4: []}

            # one shared buffer for this dt (delayed chaining around the ring)
            ring_out = self.outVals
            self.outVals = []  # reset buffer for this dt

            for z in (1, 2, 3, 4):
                for i, obj in enumerate(self.zones[z]):
                    # inlet saved from previous dt (or zeros at cold start)
                    inVals = self.cins[z][i] if self.cins[z][i] else [0.0] * len(self.components)

                    # schedule inlet for NEXT dt from outlet of PREVIOUS object processed now
                    self.cins[z][i] = ring_out
                    ring_out = []  # consumed

                    # stream mixing at the first object of Z3 (feed) and Z1 (eluent)
                    if i == 0 and z == 3:
                        for k, rec in enumerate(inVals):
                            inVals[k] = ((rec * self.flowRates[2]) +
                                         (cins_feed[k] * (self.flowRates[3] - self.flowRates[2]))
                                        ) / max(obj.flowRate, 1e-12)
                    if i == 0 and z == 1:
                        for k, rec in enumerate(inVals):
                            inVals[k] = (rec * self.flowRates[2]) / max(obj.flowRate, 1e-12)

                    # advance object and collect full profiles (old API behavior)
                    profs = obj.step(inVals)           # list of arrays per component
                    res[z].append(profs)

                    # append outlets to both buffers (for same-dt chaining and logging)
                    outlets = [p[-1] for p in profs]
                    ring_out.extend(outlets)
                    self.outVals.extend(outlets)

            # switching logic unchanged
            if self.switchingEnabled:
                self.countdown -= self.settings['dt']
                if self.countdown <= 0:
                    self.countdown += self.interval
                    self.rotate()

        return res


    def rotate(self):
        """Rotate the ring without clearing states; preserve profiles in tubes/columns."""
        # ---- pure permutation of objects (tube, col) across zones ----
        tmptube = self.zones[1][0]
        tmpcol  = self.zones[1][1]
        self.zones[1] = self.zones[1][2:] + [self.zones[2][0], self.zones[2][1]]
        self.zones[2] = self.zones[2][2:] + [self.zones[3][0], self.zones[3][1]]
        self.zones[3] = self.zones[3][2:] + [self.zones[4][0], self.zones[4][1]]
        self.zones[4] = self.zones[4][2:] + [tmptube, tmpcol]

        # ---- refresh numerics/alpha WITHOUT clearing concentration profiles ----
        dt = float(self.dt if self.dt else self.settings.get("dt", 0.05))
        Nx = int(self.settings.get("Nx", 100))

        for z in (1, 2, 3, 4):
            qz = float(self.flowRates[z])
            for obj in self.zones[z]:
                # Tubes: only recompute alpha (keep c)
                try:
                    from .Tube import Tube
                except Exception:
                    from Tube import Tube  # flat layout
                if isinstance(obj, Tube):
                    # new helper preserves comp.c
                    if hasattr(obj, "reconfigure"):
                        obj.reconfigure(qz, dt)
                    else:
                        # fallback: recompute alpha manually
                        t_res = (obj.deadVolume / max(qz, 1e-12)) * 3600.0
                        dt_cell = t_res / obj.Nx if obj.Nx > 0 else dt
                        obj._alpha = (dt / dt_cell) if dt_cell > 0 else 1.0
                        obj.flowRate = max(qz, 1e-12)
                        obj.dt = dt
                else:
                    # Columns: rebuild matrices if needed, but DO NOT zero profiles
                    # LinColumn.init() in this project keeps comp.c unless shape changes.
                    obj.init(qz, dt, Nx)

        # keep counters/flags
        self.switchState = (self.switchState + 1) % self.colCount

    def apply_zone_flows_now(self) -> None:
        """
        Push self.flowRates[z] into all tubes/columns NOW (without clearing
        concentration profiles). Use right after changing Q1..Q4.
        """
        dt = float(self.dt if self.dt else self.settings.get("dt", 0.05))
        Nx = int(self.settings.get("Nx", 100))
        for z in (1, 2, 3, 4):
            qz = float(self.flowRates[z])
            for obj in self.zones[z]:
                # Tube vs Column handling
                try:
                    from .Tube import Tube
                except Exception:
                    from Tube import Tube
                if isinstance(obj, Tube):
                    if hasattr(obj, "reconfigure"):
                        obj.reconfigure(qz, dt)        # preserves profiles
                    else:
                        # fallback: update CFL/alpha, keep profiles
                        t_res = (obj.deadVolume / max(qz, 1e-12)) * 3600.0
                        dt_cell = t_res / obj.Nx if obj.Nx > 0 else dt
                        obj._alpha = (dt / dt_cell) if dt_cell > 0 else 1.0
                        obj.flowRate = max(qz, 1e-12)
                        obj.dt = dt
                else:
                    # Columns: rebuild numerics; LinColumn.init keeps comp.c in this project
                    obj.init(qz, dt, Nx)

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
        Identical numerics to step(1), just no big dict allocations.
        """
        cins = [comp.feedConc for comp in self.components]
        self.timer += self.settings['dt']

        # locals for speed
        zones = self.zones
        flowRates = self.flowRates
        outVals_local = self.outVals
        self.outVals = []  # reset for this step

        # walk zones 1..4
        for zone in (1,2,3,4):
            # cache list reference
            objs = zones[zone]
            # iterate objects (tube, column, tube, column, ...)
            for i, col in enumerate(objs):
                inVals = self.cins[zone][i]
                self.cins[zone][i] = outVals_local
                outVals_local = []
                # mixing nodes
                if i == 0 and zone == 3:
                    for idx, inVal in enumerate(inVals):
                        inVals[idx] = ((inVal * flowRates[2]) + (cins[idx] * (flowRates[3]-flowRates[2])))/col.flowRate
                if i == 0 and zone == 1:
                    for idx, inVal in enumerate(inVals):
                        inVals[idx] = (inVal * flowRates[2]) / col.flowRate

                x = col.step(inVals)
                for x1 in x:
                    outVals_local.append(x1[-1])
                self.outVals.extend([arr[-1] for arr in x])  # keep last values for chaining

        # switching
        if self.switchingEnabled:
            self.countdown -= self.settings['dt']
            if self.countdown <= 0:
                self.countdown += self.interval
                self.rotate()

        # extract outlet = zone 1, last column; raffinate = zone 3, last column
        # Man = comp index 0, Gal = comp index 1
        try:
            # last objects in zone lists are columns; take their last profiles
            ex_last = self.zones[1][-1].components
            ra_last = self.zones[3][-1].components
            c_ex_man = float(ex_last[0].c[-1]); c_ex_gal = float(ex_last[1].c[-1])
            c_ra_man = float(ra_last[0].c[-1]); c_ra_gal = float(ra_last[1].c[-1])
        except Exception:
            c_ex_man = c_ex_gal = c_ra_man = c_ra_gal = 0.0

        return c_ex_man, c_ex_gal, c_ra_man, c_ra_gal

    def retime(self, dt: float) -> None:
        """
        Change the station time step WITHOUT clearing concentration profiles.
        Propagates to tubes (reconfigure alpha) and columns (rebuild matrices).
        """
        dt = float(dt)
        self.setdt(dt)  # updates self.settings['dt'] and self.dt

        Nx = int(self.settings.get("Nx", getattr(self, "Nx", 100)))
        for z in (1, 2, 3, 4):
            qz = float(self.flowRates[z])
            for obj in self.zones[z]:
                try:
                    from .Tube import Tube
                except Exception:
                    from Tube import Tube
                if isinstance(obj, Tube):
                    # update CFL without touching profiles
                    if hasattr(obj, "reconfigure"):
                        obj.reconfigure(flowRate=qz, dt=dt)
                    else:
                        # fallback: manual refresh
                        obj.flowRate = max(qz, 1e-12)
                        obj.dt = dt
                        if hasattr(obj, "_update_alpha"):
                            obj._update_alpha()
                else:
                    # Columns: rebuild numerics, keep profiles (LinColumn.init preserves comp.c shape)
                    obj.init(qz, dt, Nx)

    def force_switch_now(self) -> None:
        """
        Rotate the ring immediately (to match a plant boundary) and keep all
        concentration profiles intact. Also schedules the next switch from `interval`.
        """
        if not self.switchingEnabled or self.interval <= 0:
            return
        # Perform one rotation using the station's normal permutation logic.
        self.rotate()
        # After a boundary, the next switch should occur after a full SI.
        self.countdown = float(self.interval)

    def resetSwitchCountdown(self) -> None:
        """
        Reset the internal countdown so the next rotation occurs after a full SI.
        Use this when SI is changed (even if no rotation is needed right now).
        """
        if self.interval > 0:
            self.countdown = float(self.interval)
