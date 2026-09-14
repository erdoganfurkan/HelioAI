"""`helioai doctor` — what this install can and cannot do, in one screen.

The first support question a PyPI user asks is never about physics: it is "why does
`search_parameters` say the index is missing", "is my code actually sandboxed", "which
`.env` did it read". Each of those is a check below, run offline by default; `--online`
adds a single request to the configured provider. The checks are what the audits found
people getting wrong: a data directory split in two, a `.env` in the wrong folder, a
bwrap binary that the kernel refuses, and a workspace tree quietly reaching gigabytes.

`--json` emits the same report as a list of objects, which is what the CI's Docker
smoke test and a bug report both want.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time

from helioai import __version__
from helioai.config import settings

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    """One line of the report."""

    name: str
    status: str
    detail: str

    @property
    def blocking(self) -> bool:
        return self.status == FAIL


def _size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)


def _age_days(path: Path) -> float:
    return (time() - path.stat().st_mtime) / 86400


def check_python() -> Check:
    v = sys.version_info
    status = OK if (v.major, v.minor) >= (3, 12) else FAIL
    return Check("python", status, f"{v.major}.{v.minor}.{v.micro} ({sys.executable})")


def check_install() -> Check:
    from helioai.config import _IN_REPO

    where = "git clone" if _IN_REPO else "installed package"
    return Check("helioai", OK, f"{__version__}, {where}, data in {settings.data_dir}")


def check_env_file() -> Check:
    from dotenv import find_dotenv

    from helioai.config import _ROOT

    repo_env = _ROOT / ".env"
    found = str(repo_env) if repo_env.is_file() else find_dotenv(usecwd=True)
    if found:
        return Check("env file", OK, found)
    return Check("env file", WARN, "no .env found — every setting comes from the shell")


def check_provider() -> Check:
    from helioai.core.llm.factory import build_llm_client

    provider = settings.llm.provider
    try:
        build_llm_client(provider)
    except RuntimeError as e:
        return Check("llm provider", FAIL, f"{provider}: {e}")
    return Check("llm provider", OK, f"{provider}: configured")


def check_provider_online(timeout_s: float = 5.0) -> Check:
    """One GET to the provider's model list, for the OpenAI-compatible providers.

    Azure and Gemini are not probed: their SDKs have no cheap unauthenticated-shaped
    endpoint, and a wrong deployment name is only visible on a real completion.
    """
    import httpx

    from helioai.core.llm.factory import OPENAI_COMPAT

    provider = settings.llm.provider
    spec = OPENAI_COMPAT.get(provider)
    if spec is None:
        return Check("provider reachable", WARN, f"{provider}: not probed (no /models endpoint)")
    cfg = getattr(settings.llm, spec["config"])
    base_url = spec["base_url"] or f"{getattr(cfg, 'base_url', '').rstrip('/')}/v1"
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if getattr(cfg, "api_key", "") else {}
    try:
        r = httpx.get(f"{base_url}/models", headers=headers, timeout=timeout_s)
    except httpx.HTTPError as e:
        return Check("provider reachable", FAIL, f"{base_url}: {type(e).__name__}: {e}")
    if r.status_code >= 400:
        return Check("provider reachable", FAIL, f"{base_url}/models → HTTP {r.status_code}")
    return Check("provider reachable", OK, f"{base_url}/models → HTTP {r.status_code}")


def check_index() -> Check:
    chroma_dir = Path(settings.rag.chroma_dir)
    if not chroma_dir.exists():
        from helioai.tools.rag import _index_missing_message

        return Check("search index", FAIL, _index_missing_message())
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_dir))
        count = client.get_collection(settings.rag.collection_name).count()
    except Exception as e:
        return Check("search index", FAIL, f"{chroma_dir}: cannot open ({e})")
    age = _age_days(chroma_dir)
    detail = f"{count} products at {chroma_dir}, last built {age:.0f} days ago"
    return Check("search index", OK if count else WARN, detail)


def check_sandbox() -> Check:
    from helioai.tools.sandbox import _bwrap_works, _isolation_gap

    if _bwrap_works():
        return Check("sandbox", OK, "bubblewrap functional — run_python is isolated")
    gap = _isolation_gap()
    detail = "bubblewrap unavailable or refused by the kernel; run_python falls back to a plain subprocess"
    if gap:
        detail += f" and {gap}"
    return Check("sandbox", WARN, detail)


def check_speasy_inventory() -> Check:
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "speasy"
    if not base.is_dir():
        return Check("speasy inventory", WARN, f"{base} absent — first import will build it")
    return Check(
        "speasy inventory",
        OK,
        f"{base}: {_size_mb(base):.0f} MB, refreshed {_age_days(base):.0f} days ago",
    )


def check_workspaces() -> Check:
    users = Path(settings.data_dir) / "users"
    if not users.is_dir():
        return Check("workspaces", OK, "none yet")
    sessions = [p for p in users.glob("*/workspace/*") if p.is_dir()]
    size = _size_mb(users)
    status = WARN if size > 2048 else OK
    ttl_days = settings.workspace.ttl_seconds / 86400
    return Check(
        "workspaces",
        status,
        f"{len(sessions)} session dir(s), {size:.0f} MB under {users} (TTL {ttl_days:.0f} d)",
    )


def check_sessions_db() -> Check:
    from helioai.core.session import DEFAULT_DB

    db = Path(DEFAULT_DB)
    if not db.exists():
        return Check("session store", OK, f"{db} (created on first use)")
    return Check("session store", OK, f"{db}, {db.stat().st_size / 1024:.0f} KB")


def check_extras() -> Check:
    found = []
    for module, extra in (("solarmach", "solarmach"), ("sentence_transformers", "index")):
        try:
            __import__(module)
            found.append(extra)
        except Exception:
            continue
    return Check("optional extras", OK, ", ".join(found) if found else "none installed")


def run_checks(online: bool = False) -> list[Check]:
    """Run every check, never raising: a doctor that crashes explains nothing.

    Args:
        online: Also probe the configured provider with one request.

    Returns:
        The checks in display order.
    """
    # Looked up on the module at call time, so a test can replace one check by name.
    names = [
        "python",
        "install",
        "env_file",
        "provider",
        "index",
        "sandbox",
        "speasy_inventory",
        "workspaces",
        "sessions_db",
        "extras",
    ]
    if online:
        names.insert(4, "provider_online")
    module = sys.modules[__name__]
    out: list[Check] = []
    for name in names:
        fn = getattr(module, f"check_{name}")
        try:
            out.append(fn())
        except Exception as e:
            out.append(Check(name.replace("_", " "), FAIL, f"check crashed: {e}"))
    return out


_ICON = {OK: "✓", WARN: "!", FAIL: "✗"}


def format_report(checks: list[Check]) -> str:
    """The human report: one aligned line per check."""
    width = max(len(c.name) for c in checks)
    lines = [f"{_ICON[c.status]} {c.name.ljust(width)}  {c.detail}" for c in checks]
    failing = sum(c.blocking for c in checks)
    lines.append("")
    lines.append(
        "everything HelioAI needs is in place"
        if not failing
        else f"{failing} check(s) need attention before HelioAI can work"
    )
    return "\n".join(lines)


def run_doctor(online: bool = False, as_json: bool = False) -> int:
    """Print the report and return the process exit code (1 when a check fails).

    Args:
        online: Probe the provider too.
        as_json: Emit a JSON list instead of the table.
    """
    checks = run_checks(online=online)
    if as_json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        print(format_report(checks))
    return 1 if any(c.blocking for c in checks) else 0
