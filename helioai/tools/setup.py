"""Register all tools with the ToolRegistry.

Import this module once at startup (cli.py / web main.py) to make all tools
available to the agent loop. Tools are registered via the registry singleton.
"""

from __future__ import annotations

from helioai.tools.registry import registry
from helioai.tools import sandbox as _sb
from helioai.tools import speasy_tools as _spz
from helioai.tools import plasmapy_tools as _ppy
from helioai.tools import recipes as _rcp
from helioai.tools import catalog_tools as _cat


registry.register(
    name="search_parameters",
    description=(
        "Semantic search over 65 000+ speasy parameters (CDAWeb, AMDA, CSA, SSC…). "
        "Use English natural-language queries. Always call this before get_timeseries "
        "to resolve the exact parameter id. "
        "To resolve SEVERAL parameters, pass `queries` (a list) in ONE call — strongly "
        "preferred over many separate calls. Use `query` (string) for a single parameter. "
        "Examples: query='IMF magnitude ACE'; "
        "queries=['solar wind ion density MMS', 'electron energy spectrum Solar Orbiter']."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text English query for a single parameter.",
            },
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of queries to resolve several parameters in one call.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results per query (default 5).",
                "default": 5,
            },
            "provider": {
                "type": "string",
                "enum": ["amda", "cda", "csa", "ssc"],
                "description": (
                    "Optional: restrict to one provider (applies to all queries). CDA dominates "
                    "the catalog (~68k of 83k) — set amda or csa when hunting their products."
                ),
            },
        },
        "required": [],
    },
)(_spz.search_parameters)


registry.register(
    name="get_timeseries",
    description=(
        "Download a time series from any speasy provider. "
        "Always resolve the parameter id via search_parameters first. "
        "Returns a data preview (first 10 rows) and metadata."
    ),
    parameters={
        "type": "object",
        "properties": {
            "param_id": {
                "type": "string",
                "description": "Speasy parameter id (e.g. 'amda/imf', 'cdaweb/MMS1_FGM_SRVY_L2/mms1_fgm_b_gse_srvy_l2').",
            },
            "start": {
                "type": "string",
                "description": "ISO 8601 start (e.g. '2024-01-01T00:00:00').",
            },
            "stop": {
                "type": "string",
                "description": "ISO 8601 stop (e.g. '2024-01-01T06:00:00').",
            },
            "max_points": {
                "type": "integer",
                "description": "Maximum samples to return (default 5000, downsampled if needed).",
                "default": 5000,
            },
        },
        "required": ["param_id", "start", "stop"],
    },
)(_spz.get_timeseries)


registry.register(
    name="list_missions",
    description="List available speasy data providers and missions.",
    parameters={"type": "object", "properties": {}},
)(_spz.list_missions)


registry.register(
    name="run_python",
    description=(
        "Execute Python code in an isolated sandbox. "
        "Pre-imported: speasy as `spz`, numpy as `np`, matplotlib (plt.show() captures PNG), "
        "scipy (signal/stats/fft), plasmapy as `pf`, astropy.units as `u`. "
        "Use for: FFT, power spectra, plasma parameter calculations (gyrofrequency, Debye length, "
        "plasma beta), event detection, custom plots. "
        "plt.show() captures the figure as PNG and returns it as an artifact."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            },
            "timeout": {
                "type": "number",
                "description": "Max execution time in seconds (default 30).",
                "default": 30.0,
            },
        },
        "required": ["code"],
    },
)(_sb.run_python)


registry.register(
    name="plasma_beta",
    description=(
        "Compute plasma beta (ratio of thermal to magnetic pressure). "
        "SPASE: ActivityIndex / PlasmaBeta. "
        "Inputs: B in nT, density in cm⁻³, temperature in eV."
    ),
    parameters={
        "type": "object",
        "properties": {
            "B_nT": {"type": "number", "description": "Magnetic field magnitude (nT)."},
            "n_cm3": {"type": "number", "description": "Number density (cm⁻³)."},
            "T_eV": {"type": "number", "description": "Temperature (eV)."},
        },
        "required": ["B_nT", "n_cm3", "T_eV"],
    },
)(_ppy.plasma_beta)


registry.register(
    name="gyrofrequency",
    description=(
        "Compute particle gyrofrequency (cyclotron frequency). "
        "SPASE: FieldQuantity.Gyrofrequency / ParticleQuantity.Gyrofrequency. "
        "Returns frequency in Hz and angular frequency in rad/s."
    ),
    parameters={
        "type": "object",
        "properties": {
            "B_nT": {"type": "number", "description": "Magnetic field magnitude (nT)."},
            "particle": {
                "type": "string",
                "description": "'proton', 'electron', or 'alpha' (default: proton).",
            },
        },
        "required": ["B_nT"],
    },
)(_ppy.gyrofrequency)


registry.register(
    name="debye_length",
    description=("Compute electron Debye length. Returns length in meters and km."),
    parameters={
        "type": "object",
        "properties": {
            "n_cm3": {"type": "number", "description": "Electron number density (cm⁻³)."},
            "T_eV": {"type": "number", "description": "Electron temperature (eV)."},
        },
        "required": ["n_cm3", "T_eV"],
    },
)(_ppy.debye_length)


registry.register(
    name="alfven_speed",
    description=(
        "Compute Alfvén speed. SPASE: ParticleQuantity.AlfvenVelocity. Returns speed in km/s."
    ),
    parameters={
        "type": "object",
        "properties": {
            "B_nT": {"type": "number", "description": "Magnetic field magnitude (nT)."},
            "n_cm3": {"type": "number", "description": "Ion number density (cm⁻³)."},
            "mass_amu": {
                "type": "number",
                "description": "Ion mass in atomic mass units (default 1.0 = proton).",
            },
        },
        "required": ["B_nT", "n_cm3"],
    },
)(_ppy.alfven_speed)


registry.register(
    name="inertial_length",
    description=(
        "Compute ion or electron inertial length (plasma skin depth). "
        "Inputs: density in cm⁻³, particle 'proton' or 'electron'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "n_cm3": {"type": "number", "description": "Number density (cm⁻³)."},
            "particle": {
                "type": "string",
                "description": "'proton' or 'electron' (default: proton).",
            },
        },
        "required": ["n_cm3"],
    },
)(_ppy.inertial_length)


registry.register(
    name="power_spectrum",
    description=(
        "Compute power spectral density (Welch method) from a time series. "
        "SPASE: MeasurementType.Spectrum. "
        "Returns frequencies (Hz), PSD values, and peak frequency."
    ),
    parameters={
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Time series as array of floats.",
            },
            "dt_s": {"type": "number", "description": "Sampling interval in seconds."},
            "nperseg": {
                "type": "integer",
                "description": "Samples per FFT segment (default: auto).",
            },
        },
        "required": ["values", "dt_s"],
    },
)(_ppy.power_spectrum)


registry.register(
    name="list_recipes",
    description=(
        "List available derived scientific recipes (reusable Python scripts). "
        "Returns a catalogue with name, description, inputs and outputs for each recipe. "
        "Use before load_recipe to discover what is available."
    ),
    parameters={"type": "object", "properties": {}},
)(_rcp.list_recipes)


registry.register(
    name="list_catalogs",
    description=(
        "List available AMDA event catalogs and timetables (29 catalogs + 188 timetables). "
        "Returns id, name, type, number of events, survey range and description. "
        "Use the `id` with get_catalog() to inspect events or get_events_timeseries() to download data. "
        "Filter by type ('catalog'/'timetable'/'all') and region keyword (e.g. 'ICME', 'MMS', 'shock')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["all", "catalog", "timetable"],
                "description": "Filter by product type (default: all).",
            },
            "region": {
                "type": "string",
                "description": "Optional keyword filter on name/description (e.g. 'ICME', 'bow shock', 'MMS').",
            },
        },
        "required": [],
    },
)(_cat.list_catalogs)


registry.register(
    name="get_catalog",
    description=(
        "Download and summarize an AMDA event catalog or timetable. "
        "Returns event count, columns and a sample of events (start, stop + metadata). "
        "Use list_catalogs() first to find the catalog id. "
        "Use get_events_timeseries() to download a parameter over all events."
    ),
    parameters={
        "type": "object",
        "properties": {
            "catalog_id": {
                "type": "string",
                "description": "Speasy catalog uid from list_catalogs (e.g. 'amda/sharedcatalog_41').",
            },
            "start": {
                "type": "string",
                "description": "Optional ISO 8601 start — filter events beginning after this time.",
            },
            "stop": {
                "type": "string",
                "description": "Optional ISO 8601 stop — filter events beginning before this time.",
            },
            "max_events": {
                "type": "integer",
                "description": "Max events to include in the sample (default 50).",
                "default": 50,
            },
        },
        "required": ["catalog_id"],
    },
)(_cat.get_catalog)


registry.register(
    name="get_events_timeseries",
    description=(
        "Download a parameter for every event in an AMDA catalog (superposed epoch analysis). "
        "Uses speasy's native multi-interval API — one call for N events. "
        "Returns per-event statistics (mean/std/min/max) and the data for plotting. "
        "Use for: stack-plots across ICMEs/shocks, statistical surveys, epoch analysis. "
        "Workflow: list_catalogs → get_catalog (inspect) → get_events_timeseries (download)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "catalog_id": {
                "type": "string",
                "description": "Speasy catalog uid (e.g. 'amda/sharedcatalog_41').",
            },
            "param_id": {
                "type": "string",
                "description": "Speasy parameter id (e.g. 'amda/imf_gsm') — resolve via search_parameters first.",
            },
            "start": {
                "type": "string",
                "description": "ISO 8601 start — restrict to events in this window.",
            },
            "stop": {
                "type": "string",
                "description": "ISO 8601 stop — restrict to events in this window.",
            },
            "max_events": {
                "type": "integer",
                "description": "Max events to download (default 20).",
                "default": 20,
            },
        },
        "required": ["catalog_id", "param_id", "start", "stop"],
    },
)(_cat.get_events_timeseries)


registry.register(
    name="load_recipe",
    description=(
        "Load the source code of a named derived recipe. "
        "Use list_recipes() first to discover recipe names. "
        "Returns the Python source — pass it to run_python to execute it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Recipe name without .py (e.g. 'theta_bn', 'walen_test', 'mvab').",
            },
        },
        "required": ["name"],
    },
)(_rcp.load_recipe)
