# name: fill_values
# description: Blank a mission's fill values to NaN from the variable's declared FILLVAL — the logic to copy into any script that reads speasy directly instead of through HelioAI.
# inputs: a speasy variable (the object returned by spz.get_data), or a values array plus its declared fillval
# outputs: values with every non-measurement set to NaN, so a mean or a max fails loudly instead of returning a sentinel
# reference: CDF ISTP metadata guidelines, FILLVAL attribute — https://spdf.gsfc.nasa.gov/istp_guide/vattributes.html

"""Fill-value blanking for code that runs outside HelioAI.

Inside HelioAI this is already done: `get_timeseries` blanks fills before it
persists anything, so `load_data()` hands the sandbox NaN and nothing else. This
recipe exists for the *other* case — a standalone script, a notebook you hand to a
colleague, anything calling `spz.get_data` directly, where the raw sentinels are
still in the array.

Copy `fill_mask` and `blank_fill` into that script verbatim. They need numpy only.

Three conventions have to be caught at once, which is why one function does all
three rather than each caller inventing a threshold:

- non-finite (NaN/inf) — already unusable;
- the ~1e31 magnitude convention, used by ACE among others;
- the value the dataset *declares* in its CDF FILLVAL, the only way to catch
  Wind/SWE's 99999.9.

Two traps that have each produced a wrong published number:

- **FILLVAL is often a list, not a scalar.** `cda/WI_H1_SWE/Proton_V_nonlin`
  declares `[99999.8984375]`. A bare `float(fillval)` raises TypeError, and inside
  a `try/except: pass` that means no filtering at all happens — silently. One
  surviving fill point out of 1361 is enough to turn a 510 km/s downstream mean
  into 3987 km/s and a Mach number into -103.
- **A hard-coded cutoff is wrong in both directions.** `>= 99999` throws away
  OMNI's real 99093 K proton temperature; `> 1e30` sails straight past Wind's
  99999.9. Read what the variable declares.

The float32 round-trip is why the comparison is `isclose(rtol=1e-6)` and not `==`:
Wind declares 99999.9 and stores 99999.8984375.

Usage inside run_python:
    var = spz.get_data("cda/WI_H1_SWE/Proton_V_nonlin", start, stop)
    t, v = clean_variable(var)
"""

import numpy as np


def fill_mask(values, fillval=None):
    """Boolean mask of samples that carry no measurement."""
    bad = ~np.isfinite(values) | (np.abs(values) >= 1e30)
    if fillval is not None:
        try:
            fv = float(np.asarray(fillval).ravel()[0])
            if np.isfinite(fv):
                bad = bad | np.isclose(values, fv, rtol=1e-6)
        except (TypeError, ValueError, IndexError):
            pass
    return bad


def blank_fill(values, fillval=None):
    """Return `values` as float64 with every fill sample replaced by NaN."""
    numeric = np.array(values, dtype="float64")
    numeric[fill_mask(numeric, fillval)] = np.nan
    return numeric


def clean_variable(var):
    """Return (time, values) for a speasy variable, fills already blanked.

    Reads FILLVAL out of `var.meta` rather than guessing, and tolerates its
    absence — a variable that declares none still gets the non-finite and 1e31
    conventions applied.
    """
    meta = getattr(var, "meta", None) or {}
    fillval = meta.get("FILLVAL") if isinstance(meta, dict) else None
    return np.array(var.time), blank_fill(var.values, fillval)


# Self-check on the exact values that have caused wrong results, so a change that
# breaks the contract fails here rather than in someone's Mach number.
_wind = np.array([420.0, 99999.8984375, 435.0])
assert np.isnan(blank_fill(_wind, [99999.9])[1]), "Wind: FILLVAL comes as a list"
assert np.isnan(blank_fill(_wind, 99999.9)[1]), "Wind: float32 round-trip needs rtol"
assert not np.isnan(blank_fill(np.array([99093.0]), [99999.9])[0]), (
    "OMNI: 99093 K is a real temperature, a >=99999 cutoff would eat it"
)
assert np.isnan(blank_fill(np.array([5.0, -1e31]), None)[1]), "ACE: 1e31 needs no FILLVAL"
assert blank_fill(np.array([1.0, 2.0]), None).tolist() == [1.0, 2.0], "clean data untouched"

print("fill_values: self-check passed — copy fill_mask/blank_fill into standalone scripts")
