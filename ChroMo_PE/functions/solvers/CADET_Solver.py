import os
import sys
import numpy as np

_CADET_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'CADET'))

CADET_DEFAULT_MOLAR_MASS = 180.16  # [g/mol] fallback when component not in the dict
cadet_molar_mass = {}              # {component_name: molar_mass_g_per_mol} — set by Web_Server before each run
cadet_install_path = None          # override from Solver Settings; None = auto-detect

# CADET uses Finite Volume, not Finite Difference.  ncol is the number of FV *cells*,
# not grid points — the two discretisation schemes are not directly comparable.
# FVM is typically accurate with far fewer cells than FDM needs grid points.
# A good default for chromatography EDM with LumpedRateModelWithoutPores is 50–100 cells.
cadet_ncol = 10  # number of FV cells for CADET column (independent of spacialDiff / Nx)


def _find_cadet_path():
    """Search for CADET binary using several strategies. Returns install-path string or None."""
    import shutil, glob
    # 1. Check PATH first
    cli = shutil.which("cadet-cli") or shutil.which("cadet-cli.exe")
    if cli:
        return str(os.path.dirname(os.path.dirname(cli)))  # bin/../
    # 2. CADET_PATH env var
    env_path = os.environ.get("CADET_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    # 3. Common user-level locations (cadet_bin, cadet, cadet-core, …)
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "cadet_bin"),
        os.path.join(home, "cadet"),
        os.path.join(home, "cadet-core"),
        os.path.join(home, "AppData", "Local", "cadet"),
        os.path.join(home, "AppData", "Local", "cadet-core"),
        "C:/cadet",
        "C:/cadet-core",
    ]
    for base in candidates:
        exe = os.path.join(base, "bin", "cadet-cli.exe")
        if os.path.isfile(exe):
            return base
        exe_linux = os.path.join(base, "bin", "cadet-cli")
        if os.path.isfile(exe_linux):
            return base
    # 4. Conda environments
    for dist in ("Anaconda3", "Miniconda3", "anaconda3", "miniconda3", "mambaforge"):
        for base in (home, "C:/ProgramData"):
            lib_bin = os.path.join(base, dist, "Library", "bin")
            if os.path.isfile(os.path.join(lib_bin, "cadet-cli.exe")):
                return os.path.join(base, dist, "Library")
    return None


def _import_cadet():
    """Lazily import CADETProcess to avoid eager loading of optional deps (hopsy etc.)."""
    if _CADET_FOLDER not in sys.path:
        sys.path.insert(0, _CADET_FOLDER)
    import Conversion as _Conv
    # CADETProcess.__init__ imports the optimization sub-package which requires 'hopsy'.
    # We only need the simulator and processModel parts, so we inject a mock if hopsy
    # is absent — this lets the package initialise without touching the unused optimiser code.
    if 'hopsy' not in sys.modules:
        try:
            import hopsy  # noqa: F401
        except ModuleNotFoundError:
            from unittest.mock import MagicMock
            sys.modules['hopsy'] = MagicMock()
    from CADETProcess.simulator import Cadet as _Cadet
    from CADETProcess.processModel import (
        ComponentSystem, Inlet, Outlet, FlowSheet, Process,
        Langmuir, Linear, LumpedRateModelWithoutPores
    )
    return _Conv, _Cadet, ComponentSystem, Inlet, Outlet, FlowSheet, Process, Langmuir, Linear, LumpedRateModelWithoutPores


def _build_column_process(Inlet, FlowSheet, Process, LumpedRateModelWithoutPores, Outlet,
                           Conversion, cs, bm, feedConc_molar, flowRate, length, diameter,
                           feedVol, porosity, disperCoef, ncol, time_total):
    # ncol = number of Finite Volume cells (CADET FVM, NOT the same as FDM Nx grid points)
    feed = Inlet(cs, name='feed')
    eluent = Inlet(cs, name='eluent')
    feed.c = [feedConc_molar]
    eluent.c = [0]

    column = LumpedRateModelWithoutPores(cs, name='column')
    column.binding_model = bm
    column.length = length / 1000
    column.diameter = diameter / 1000
    column.total_porosity = porosity
    column.axial_dispersion = disperCoef / 1e6
    column.discretization.ncol = ncol
    column.solution_recorder.write_solution_bulk = True

    outlet = Outlet(cs, name='outlet')
    fs = FlowSheet(cs)
    fs.add_unit(feed, feed_inlet=True)
    fs.add_unit(eluent, eluent_inlet=True)
    fs.add_unit(column)
    fs.add_unit(outlet, product_outlet=True)
    fs.add_connection(feed, column)
    fs.add_connection(eluent, column)
    fs.add_connection(column, outlet)

    process = Process(fs, name='cadet_process')
    flowrate_m3_s = Conversion.convert_flowrate_mL_per_h_to_m3_per_s(flowRate)
    process.add_event('feed_on', 'flow_sheet.feed.flow_rate', flowrate_m3_s)
    process.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0)
    process.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', flowrate_m3_s)
    process.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0)
    process.add_duration('feed_duration')

    feed_duration = feedVol / flowRate * 3600
    process.feed_duration.time = feed_duration
    process.add_event_dependency('feed_off', ['feed_on', 'feed_duration'], [1, 1])
    process.add_event_dependency('eluent_on', ['feed_off'])
    process.add_event_dependency('eluent_off', ['feed_on'])
    process.cycle_time = time_total

    return process, feed_duration


def _create_cadet_sim(Cadet):
    """Instantiate Cadet simulator, resolving the install path if needed."""
    path = cadet_install_path or _find_cadet_path()
    if path:
        return Cadet(install_path=path)
    try:
        return Cadet()  # last-resort auto-detect
    except TypeError:
        raise RuntimeError(
            "CADET binary not found. Set the CADET install path in Solver Settings "
            "(e.g. C:/Users/yourname/cadet_bin)."
        )


def _extract_bulk(result, molar_mass):
    """Return (c_g_L, t) from a CADET simulation result."""
    t = np.asarray(result.solution.column.outlet.time).flatten()
    bulk = result.solution.column.bulk.solution
    c = np.asarray(bulk)
    if c.ndim == 3 and c.shape[-1] == 1:
        c = c[:, :, 0]
    c = (c * molar_mass) / 1000  # mol/m³ → g/L
    return c, t


def CADET_Lin_Solver(
    flowRate=150,
    length=235,
    diameter=16,
    feedVol=1,
    feedConc=0.15,
    porosity=0.2,
    henryConst=0.5,
    disperCoef=3.0,
    ncol=None,    # FV cells — if None, uses module-level cadet_ncol (default 50)
    Nt=3000,
    time=10800,
    debugPrint=False,
    debugGraph=False,
    full=False,
    molar_mass=180.16
):
    Conversion, Cadet, ComponentSystem, Inlet, Outlet, FlowSheet, Process, Langmuir, Linear, LumpedRateModelWithoutPores = _import_cadet()

    cs = ComponentSystem()
    cs.add_component('A')

    bm = Linear(cs, name='linear')
    bm.is_kinetic = False
    # Henry constant is dimensionless (q [g/L] = K × c [g/L]), same value in molar and mass units
    bm.adsorption_rate = [henryConst]
    bm.desorption_rate = [1]

    _ncol = ncol if ncol is not None else cadet_ncol
    feedConc_molar = Conversion.concentration_mass_to_molar(feedConc, molar_mass)
    process, feed_duration = _build_column_process(
        Inlet, FlowSheet, Process, LumpedRateModelWithoutPores, Outlet,
        Conversion, cs, bm, feedConc_molar, flowRate, length, diameter,
        feedVol, porosity, disperCoef, _ncol, time
    )

    sim = _create_cadet_sim(Cadet)
    sim.time_resolution = time / Nt
    result = sim.simulate(process)

    c, t = _extract_bulk(result, molar_mass)

    if debugPrint:
        print(f"CADET_Lin: ncol={_ncol} (FVM cells), t.shape={t.shape}, outlet_max={c[:,-1].max():.4f} g/L")

    if full:
        feed = np.zeros(len(t))
        feed[t <= feed_duration] = feedConc
        return [c, t, feed, np.zeros(len(t)), 0.0]
    return [c, t]


def CADET_Nonlin_Solver(
    flowRate=150,
    length=235,
    diameter=16,
    feedVol=1,
    feedConc=0.15,
    porosity=0.2,
    langmuirConst=0.5,
    disperCoef=3.0,
    saturCoef=20.0,
    ncol=None,    # FV cells — if None, uses module-level cadet_ncol (default 50)
    Nt=3000,
    time=10800,
    debugPrint=False,
    debugGraph=False,
    full=False,
    molar_mass=180.16
):
    Conversion, Cadet, ComponentSystem, Inlet, Outlet, FlowSheet, Process, Langmuir, Linear, LumpedRateModelWithoutPores = _import_cadet()

    cs = ComponentSystem()
    cs.add_component('A')

    bm = Langmuir(cs, name='langmuir')
    bm.is_kinetic = False
    bm.adsorption_rate = [Conversion.langmuir_constant_mass_to_molar(langmuirConst, molar_mass)]
    bm.desorption_rate = [1]
    bm.capacity = [Conversion.concentration_mass_to_molar(saturCoef, molar_mass)]

    _ncol = ncol if ncol is not None else cadet_ncol
    feedConc_molar = Conversion.concentration_mass_to_molar(feedConc, molar_mass)
    process, feed_duration = _build_column_process(
        Inlet, FlowSheet, Process, LumpedRateModelWithoutPores, Outlet,
        Conversion, cs, bm, feedConc_molar, flowRate, length, diameter,
        feedVol, porosity, disperCoef, _ncol, time
    )

    sim = _create_cadet_sim(Cadet)
    sim.time_resolution = time / Nt
    result = sim.simulate(process)

    c, t = _extract_bulk(result, molar_mass)

    if debugPrint:
        print(f"CADET_Nonlin: ncol={_ncol} (FVM cells), t.shape={t.shape}, outlet_max={c[:,-1].max():.4f} g/L")

    if full:
        feed = np.zeros(len(t))
        feed[t <= feed_duration] = feedConc
        return [c, t, feed, np.zeros(len(t)), 0.0]
    return [c, t]
