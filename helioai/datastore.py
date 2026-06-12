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


def _slug(param_id: str) -> str:
    last = param_id.rstrip("/").split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "_", last.lower()).strip("_") or "dataset"


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
        import numpy as np
        import time as _time

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
        manifest.setdefault("datasets", {})[name] = {
            "kind": "timeseries",
            "file": fname,
            "param_id": param_id,
            "units": units,
            "start": start,
            "stop": stop,
            "shape": list(values_arr.shape),
            "columns": cols,
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
        import numpy as np
        import time as _time

        data_dir = _session_data_dir()
        if data_dir is None:
            return None

        arrays: dict[str, any] = {}
        events_meta: list[dict] = []
        total_bytes = 0

        for i, (ev_start, ev_stop, ts) in enumerate(series):
            if ts is None or not hasattr(ts, "time") or not hasattr(ts, "values"):
                events_meta.append(
                    {"idx": i, "start": ev_start, "stop": ev_stop, "status": "no_data"}
                )
                continue
            try:
                t_arr = np.asarray(ts.time)
                v_arr = np.asarray(ts.values, dtype=float)
                total_bytes += t_arr.nbytes + v_arr.nbytes
                if total_bytes > _MAX_BYTES:
                    log.warning(
                        "datastore: event collection for %r exceeds 100 MB cap, truncating",
                        param_id,
                    )
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
