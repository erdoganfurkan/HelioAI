# name: solar_mach
# description: Multi-spacecraft solar magnetic connectivity plot (Parker spirals) — where probes and planets sit and which solar longitude they connect to at a given date.
# inputs: date (ISO string), bodies (list, e.g. ["Earth", "STEREO A", "Parker Solar Probe", "Solar Orbiter"]), vsw (optional list of solar wind speeds in km/s, one per body, default 400)
# outputs: figure — polar Parker-spiral connectivity plot; connectivity table printed
# reference: Gieseler et al. (2023), Front. Astron. Space Sci. 9, doi:10.3389/fspas.2022.1058810; https://solar-mach.github.io

"""Solar-MACH connectivity plot.

Standard community figure for SEP / CME multi-spacecraft studies: positions of
the selected observers, their Parker spiral field lines and the longitude of
their solar magnetic footpoint. Requires the optional `solarmach` package
(heavy sunpy stack) — install with `pip install "helioai-agent[solarmach]"`.

Usage inside run_python:
    date = "2021-10-28T15:15:00"
    bodies = ["Earth", "STEREO A", "Parker Solar Probe", "Solar Orbiter"]
    vsw = [420, 400, 380, 400]
    # then run this script
"""

try:
    from solarmach import SolarMACH
except ImportError:
    SolarMACH = None

date = globals().get("date", "2021-10-28T15:15:00")
bodies = globals().get("bodies", ["Earth", "STEREO A", "Parker Solar Probe", "Solar Orbiter"])
vsw = globals().get("vsw", [400] * len(bodies))

if SolarMACH is None:
    print(
        "solarmach is not installed in this environment. "
        'Install the optional extra: pip install "helioai-agent[solarmach]" (or: uv add solarmach), '
        "then re-run this recipe."
    )
else:
    sm = SolarMACH(date.replace("T", " "), bodies, vsw)
    sm.plot(plot_spirals=True, plot_sun_body_line=True, numbered_markers=True)
    plt.show()  # noqa: F821 — provided by the sandbox preamble
    print(sm.coord_table.to_string())
