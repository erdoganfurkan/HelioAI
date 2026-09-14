"""Every setting the code reads from the environment is documented where users look.

Four variables (`HELIOAI_CATALOGS_DIR`, `HELIOAI_OLLAMA_HEADERS`, `HELIOAI_LOG_LEVEL`, and
the since-removed `HELIOAI_WORKSPACE`) were read by production code and absent from
`.env.example`; a user could not learn they existed without grepping the source. This
test greps the source so the next one fails in CI instead.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "helioai"

_ENV_READ = re.compile(r"""os\.environ(?:\.get)?\s*[\[(]\s*["']([A-Z][A-Z0-9_]+)["']""")

# Read by the code but not HelioAI settings: the platform's own variables, and the
# editor the CLI opens for `helioai profile`.
_NOT_SETTINGS = {"XDG_DATA_HOME", "APPDATA", "HOME", "EDITOR", "PATH", "MPLBACKEND"}


def _env_vars_read_by_the_code() -> set[str]:
    found: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        found.update(_ENV_READ.findall(path.read_text(encoding="utf-8")))
    return found - _NOT_SETTINGS


def _env_vars_in_example() -> set[str]:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return set(re.findall(r"^\s*#?\s*([A-Z][A-Z0-9_]+)=", text, flags=re.MULTILINE))


def test_the_code_reads_settings_this_test_knows_about():
    """Sanity check on the grep itself: an empty set would make the test below vacuous."""
    read = _env_vars_read_by_the_code()
    assert {"HELIOAI_LLM_PROVIDER", "HELIOAI_DATA_DIR", "ADS_API_TOKEN"} <= read


def test_every_setting_read_from_the_environment_is_in_env_example():
    missing = _env_vars_read_by_the_code() - _env_vars_in_example()
    assert not missing, f"read by the code, absent from .env.example: {sorted(missing)}"


def test_env_example_documents_nothing_the_code_no_longer_reads():
    """The reverse drift: a variable kept in the example after the code stopped reading it
    (HELIOAI_WORKSPACE was one) sends users configuring a knob that does nothing."""
    stale = _env_vars_in_example() - _env_vars_read_by_the_code()
    assert not stale, f"in .env.example, read nowhere: {sorted(stale)}"
