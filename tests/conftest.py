"""Shared pytest fixtures for HelioAI tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helioai.core.llm.base import LLMClient, Message
from helioai.tools.sandbox import _sandbox_env

_SANDBOX_WARMUP = """\
import os; os.environ['MPLBACKEND'] = 'Agg'
import numpy, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot
import scipy, scipy.signal, scipy.stats, scipy.fft
import astropy, astropy.units
try:
    import plasmapy, plasmapy.formulary
except Exception:
    pass
try:
    import geopack, geopack.geopack
except Exception:
    pass
"""


@pytest.fixture(scope="session", autouse=True)
def _warm_sandbox_imports():
    """Pre-compile .pyc for all sandbox heavy imports.

    Runs once per test session as a subprocess so that subsequent sandbox
    subprocess calls find pre-compiled bytecode instead of compiling from scratch.
    Without this, cold-start CI runners time out on trivial sandbox tests.

    Best-effort only: a slow/hanging warmup must not fail the whole test session
    over a perf optimization — fall through and let tests cold-start instead.

    speasy is deliberately NOT warmed here. Its import builds the provider
    inventory, which goes to the network whenever the sandbox HOME is cold —
    and while the SciQLop inventory cache is down (HTTP 502) that costs the full
    timeout on EVERY pytest invocation, whatever the selection: 180 s of setup
    for a suite whose tests are all under a second. The rest of the stack warms
    in 2 s. The sandbox imports speasy lazily anyway, so nothing else pays for it.

    Runs under _sandbox_env() so the caches land in the exact HOME (/tmp) that
    sandbox subprocesses will read — warming the runner's real HOME leaves the
    sandbox cold.
    """
    try:
        subprocess.run(
            [sys.executable, "-c", _SANDBOX_WARMUP],
            timeout=60,
            capture_output=True,
            env=_sandbox_env(),
        )
    except subprocess.TimeoutExpired:
        pass


@pytest.fixture(autouse=True)
def _reset_workspace_ctx():
    """Keep the workspace contextvars (user/label/session) from leaking across tests."""
    import helioai.workspace as ws

    yield
    ws._current_user.set(None)
    ws._current_label.set(None)
    ws._current_session.set(None)


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test writes under its own `tmp_path`, never under the configured `data/`.

    `settings` is a module-level singleton built at import, so a test that forgets to
    repoint it lands sessions, workspaces and a 300 MB speasy seed in the real data
    directory of whoever runs the suite — noticed twice, both times by `ls`, months
    apart. The store is redirected too: its database path was fixed at import from the
    same singleton, so patching `data_dir` alone still wrote `sessions.db` in place.
    A test that needs another layout patches over this; the default is now hermetic.
    """
    from helioai.config import settings
    from helioai.core import session
    from helioai.tools import sandbox

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings.rag, "chroma_dir", tmp_path / "chroma")
    monkeypatch.setattr(settings.catalogs, "catalogs_dir", tmp_path / "catalogs")
    monkeypatch.setattr(settings.profile, "profile_path", tmp_path / "profile.md")
    monkeypatch.setattr(session.store, "_db_path", tmp_path / "sessions.db")
    monkeypatch.setattr(session.store, "_schema_ready", False)
    monkeypatch.setattr(session.store, "_cache", {})
    monkeypatch.setattr(session.store, "_turn_locks", {})
    # The developer's inventory is 300 MB under the VS Code snap; copied into each
    # test's data directory it filled /tmp with 11 GB. Seeding has its own tests.
    monkeypatch.setattr(sandbox, "_host_speasy_inventory", lambda: tmp_path / "no-host-inventory")
    return tmp_path


def _data_tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    snapshot = {".": (-1, root.lstat().st_mtime_ns)}
    for p in root.rglob("*"):
        try:
            st = p.lstat()
        except OSError:
            continue
        snapshot[str(p.relative_to(root))] = (st.st_size if p.is_file() else -1, st.st_mtime_ns)
    return snapshot


def pytest_sessionstart(session: pytest.Session) -> None:
    """Remember what the real data directory looked like before any test ran."""
    from helioai.config import settings

    root = Path(settings.data_dir)
    session.config._helioai_data_root = root  # type: ignore[attr-defined]
    session.config._helioai_data_before = _data_tree_snapshot(root)  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the run when a test wrote under the real data directory.

    A leak is invisible from any single test — the writes succeed — and it is only ever
    noticed when someone lists `data/users` and finds sessions named after fixtures.
    Comparing the tree before and after the whole run catches it whatever the test
    selection, and names the paths so the guilty test is a `grep` away. Pre-existing
    files are not the suite's to judge: only what changed during the run counts.
    """
    root = getattr(session.config, "_helioai_data_root", None)
    if root is None:
        return
    before = session.config._helioai_data_before  # type: ignore[attr-defined]
    after = _data_tree_snapshot(root)
    changed = sorted(
        {k for k in before.keys() ^ after.keys()}
        | {k for k in before.keys() & after.keys() if before[k] != after[k]}
    )
    if not changed:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    lines = [f"tests wrote under the real data directory {root}:"] + [
        f"  {p}" for p in changed[:40]
    ]
    if len(changed) > 40:
        lines.append(f"  … and {len(changed) - 40} more")
    if reporter is not None:
        reporter.ensure_newline()
        reporter.section("data/ leak", sep="!", red=True, bold=True)
        for line in lines:
            reporter.line(line)
    else:
        print("\n".join(lines), file=sys.stderr)
    if session.exitstatus == 0:
        session.exitstatus = 1


@pytest.fixture
def tmp_chroma_dir(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


@pytest.fixture
def sample_skill_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    return d


class _FakeLLM(LLMClient):
    """Replays a scripted list of Messages, one per chat() call."""

    def __init__(self, responses: list[Message]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages, tools, system_prompt=None):
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "system_prompt": system_prompt,
            }
        )
        return self._responses.pop(0)


@pytest.fixture
def fake_llm_factory():
    return _FakeLLM


class _FakeEmbedModel:
    """Returns deterministic unit-norm embeddings for any text."""

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
        rng = np.random.default_rng(42)
        vecs = rng.random((len(texts), 128)).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-9)
        return vecs


@pytest.fixture
def fake_embed_model():
    return _FakeEmbedModel()
