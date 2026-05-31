"""Register all tools with the ToolRegistry.

Import this module once at startup (cli.py / web main.py) to make all tools
available to the agent loop. Tools are registered via the registry singleton.
"""

from __future__ import annotations

from helioai.tools.registry import registry
from helioai.tools import sandbox as _sb
from helioai.tools import speasy_tools as _spz


registry.register(
    name="search_parameters",
    description=(
        "Semantic search over 65 000+ speasy parameters (CDAWeb, AMDA, CSA, SSC…). "
        "Use English natural-language queries. Always call this before get_timeseries "
        "to resolve the exact parameter id. "
        "Examples: 'solar wind ion density MMS', 'IMF magnitude ACE', "
        "'electron energy spectrum Solar Orbiter'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Free-text English search query."},
            "top_k": {"type": "integer", "description": "Number of results (default 5).", "default": 5},
        },
        "required": ["query"],
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
            "start": {"type": "string", "description": "ISO 8601 start (e.g. '2024-01-01T00:00:00')."},
            "stop": {"type": "string", "description": "ISO 8601 stop (e.g. '2024-01-01T06:00:00')."},
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
