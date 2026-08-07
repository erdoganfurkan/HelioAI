"""Session datastore — persist downloaded timeseries as .npz for reuse in run_python.

Layout:  <session_dir>/data/<name>.npz      — compressed numpy archive
         <session_dir>/data/manifest.json   — index: name → metadata

Datasets are accessible in the sandbox via load_data("name").
The persisted data is the full-resolution download (before any downsampling).
All I/O errors are silently swallowed — persistence must never break a tool call.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_MAX_BYTES = 100 * 1024 * 1024  # 100 MB cap per dataset
DATA_SUBDIR = "data"


def fill_mask(values, fillval: float | list | None = None):
    """Boolean mask of samples that carry no measurement.

    Three conventions have to be caught at once, which is why every caller shares
    this one function instead of applying its own threshold:

    - non-finite (NaN/inf) — already unusable;
    - the ~1e31 magnitude convention, used by ACE among others;
    - the value the dataset *declares* in its CDF FILLVAL, which is the only way
      to catch Wind/SWE's 99999.9. A blanket "reject >= 99999" rule is wrong: OMNI
      carries a real proton temperature of 99093 K in the 2003 Halloween window.

    Args:
        fillval: the declared FILLVAL, when the provider exposes it. A list as often
            as a scalar — Wind/SWE declares `[99999.8984375]`, and a bare
            `float(fillval)` on it raises. Compared with rtol=1e-6 to survive a
            float32 round-trip while staying far tighter than the ~1% gap to real
            nearby values.
    """
    import numpy as np

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
    """Return (values with fill blanked to NaN, mask), or (values, None) if not numeric.

    Applied at every point where downloaded data is persisted, so that anything
    reading it back — the sandbox, the exported notebook — sees NaN for "no
    measurement" rather than a sentinel that looks like a plausible reading.
    Leaving it to the reader meant remembering to call clean(), which cannot see
    FILLVAL, so one forgotten call put a 99999.9 "speed" into a plot and a mean.

    Non-numeric parameters (string labels, epochs) have no fill convention and are
    passed through untouched.
    """
    import numpy as np

    try:
        numeric = np.array(values, dtype="float64")
    except (TypeError, ValueError):
        return values, None
    mask = fill_mask(numeric, fillval)
    numeric[mask] = np.nan
    return numeric, mask


def _session_data_dir() -> Path | None:
    try:
        from helioai.workspace import get_session_dir

        sdir = get_session_dir()
        d = sdir / DATA_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception as e:
        log.warning("datastore: cannot resolve session dir: %s", e)
        return None


# Names the sandbox preamble binds. A dataset may not take one of these: the model
# writes `np = load_data("np")` for ACE density, which rebinds numpy — and since
# load_data itself calls np.load, every LATER load_data in the same cell dies with
# "SimpleNamespace has no attribute 'load'". One plasma variable poisons the rest.
_SANDBOX_NAMES = frozenset(
    "np plt spz pf u os json sys re math scipy matplotlib plasmapy "
    "load_data clean export time magnitude interp_to param_card".split()
)


def _slug(param_id: str) -> str:
    last = param_id.rstrip("/").split("/")[-1]
    slug = re.sub(r"[^a-z0-9]+", "_", last.lower()).strip("_") or "dataset"
    return f"{slug}_data" if slug in _SANDBOX_NAMES else slug


def _read_manifest_file(data_dir: Path) -> dict:
    mfile = data_dir / "manifest.json"
    if mfile.exists():
        try:
            return json.loads(mfile.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"datasets": {}}


def _write_manifest_file(data_dir: Path, manifest: dict) -> None:
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_manifest(session_dir: Path) -> dict:
    """Return the manifest dict for a given session directory."""
    return _read_manifest_file(session_dir / DATA_SUBDIR)


def find_existing(param_id: str, start: str, stop: str) -> str | None:
    """Name of an already-persisted dataset for this exact param and window, or None.

    The same matching `_unique_name` uses to reuse a slot — but consulted *before* the
    download rather than after, so a repeat request costs a dict lookup instead of a
    network round-trip. The prompt has always said "download each parameter ONCE"; a
    real run still re-fetched the same Wind field three times across three turns, with
    the dataset name sitting in plain sight in its own history. Discipline the model
    does not reliably apply belongs in the tool.
    """
    try:
        data_dir = _session_data_dir()
        if data_dir is None:
            return None
        for name, entry in _read_manifest_file(data_dir).get("datasets", {}).items():
            if (
                entry.get("param_id") == param_id
                and entry.get("start") == start
                and entry.get("stop") == stop
                and entry.get("kind") == "timeseries"
            ):
                return name
    except Exception as e:  # noqa: BLE001 — a cache miss must never block a download
        log.debug("find_existing failed: %s", e)
    return None


def _unique_name(manifest: dict, base: str, param_id: str, start: str, stop: str) -> str:
    datasets = manifest.get("datasets", {})
    existing = datasets.get(base)
    if (
        existing
        and existing.get("param_id") == param_id
        and existing.get("start") == start
        and existing.get("stop") == stop
    ):
        return base
    if base not in datasets:
        return base
    i = 2
    while f"{base}_{i}" in datasets:
        i += 1
    return f"{base}_{i}"


def save_timeseries(
    name_hint: str,
    *,
    time,
    values,
    param_id: str,
    units: str,
    start: str,
    stop: str,
    columns,
    source: str,
) -> dict | None:
    """Persist a timeseries download. Returns {"dataset": name} or None on failure."""
    try:
        import time as _time

        import numpy as np

        data_dir = _session_data_dir()
        if data_dir is None:
            return None

        time_arr = np.asarray(time)
        values_arr = np.asarray(values, dtype=float)

        if values_arr.nbytes > _MAX_BYTES:
            log.warning(
                "datastore: skipping %r — %d MB exceeds cap",
                param_id,
                values_arr.nbytes // (1024 * 1024),
            )
            return None

        manifest = _read_manifest_file(data_dir)
        base = _slug(name_hint or param_id)
        name = _unique_name(manifest, base, param_id, start, stop)
        fname = f"{name}.npz"

        np.savez_compressed(data_dir / fname, time=time_arr, values=values_arr)

        cols = list(columns) if isinstance(columns, (list, tuple)) else []
        # Derived from what was actually written rather than passed in, so it can
        # never drift from the file: get_timeseries blanks fill values to NaN
        # before saving, so this is the fraction with no measurement.
        missing_pct = round(100 * float(np.isnan(values_arr).mean()), 1) if values_arr.size else 0.0
        manifest.setdefault("datasets", {})[name] = {
            "kind": "timeseries",
            "file": fname,
            "param_id": param_id,
            "units": units,
            "start": start,
            "stop": stop,
            "shape": list(values_arr.shape),
            "columns": cols,
            "missing_pct": missing_pct,
            "source": source,
            "created": str(int(_time.time())),
        }
        _write_manifest_file(data_dir, manifest)
        return {"dataset": name}
    except Exception as e:
        log.warning("datastore: save_timeseries failed for %r: %s", param_id, e)
        return None


def save_event_collection(
    name_hint: str,
    *,
    series,
    param_id: str,
    units: str,
    source: str,
) -> dict | None:
    """Persist a batch of per-event timeseries. series = [(ev_start, ev_stop, ts|None), ...].
    Returns {"dataset": name} or None on failure.
    """
    try:
        import time as _time

        import numpy as np

        data_dir = _session_data_dir()
        if data_dir is None:
            return None

        arrays: dict[str, any] = {}
        events_meta: list[dict] = []
        total_bytes = 0
        capped = False

        for i, (ev_start, ev_stop, ts) in enumerate(series):
            if ts is None or not hasattr(ts, "time") or not hasattr(ts, "values"):
                events_meta.append(
                    {"idx": i, "start": ev_start, "stop": ev_stop, "status": "no_data"}
                )
                continue
            try:
                t_arr = np.asarray(ts.time)
                v_arr, _ = blank_fill(ts.values, (getattr(ts, "meta", {}) or {}).get("FILLVAL"))
                v_arr = np.asarray(v_arr, dtype=float)
                total_bytes += t_arr.nbytes + v_arr.nbytes
                if total_bytes > _MAX_BYTES:
                    if not capped:
                        log.warning(
                            "datastore: event collection for %r exceeds 100 MB cap, truncating",
                            param_id,
                        )
                        capped = True
                    events_meta.append(
                        {"idx": i, "start": ev_start, "stop": ev_stop, "status": "truncated"}
                    )
                    continue
                arrays[f"t{i}"] = t_arr
                arrays[f"v{i}"] = v_arr
                events_meta.append({"idx": i, "start": ev_start, "stop": ev_stop, "status": "ok"})
            except Exception:
                events_meta.append(
                    {"idx": i, "start": ev_start, "stop": ev_stop, "status": "no_data"}
                )

        if not arrays:
            return None

        manifest = _read_manifest_file(data_dir)
        base = _slug(name_hint or param_id) + "_events"
        name = base
        if name in manifest.get("datasets", {}):
            i2 = 2
            while f"{base}_{i2}" in manifest.get("datasets", {}):
                i2 += 1
            name = f"{base}_{i2}"
        fname = f"{name}.npz"

        np.savez_compressed(data_dir / fname, **arrays)

        manifest.setdefault("datasets", {})[name] = {
            "kind": "event_collection",
            "file": fname,
            "param_id": param_id,
            "units": units,
            "n_events": len(series),
            "events": events_meta,
            "source": source,
            "created": str(int(_time.time())),
        }
        _write_manifest_file(data_dir, manifest)
        return {"dataset": name}
    except Exception as e:
        log.warning("datastore: save_event_collection failed for %r: %s", param_id, e)
        return None
