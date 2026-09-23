"""`helioai doctor`: every check answers offline, and the report is honest about failures."""

from __future__ import annotations

import json

import pytest

from helioai import doctor


@pytest.fixture
def quiet_install(monkeypatch, tmp_path):
    """An install with nothing built yet, pointed at tmp_path, no network, no bwrap."""
    from helioai.config import settings

    monkeypatch.setattr(settings.rag, "chroma_dir", tmp_path / "chroma")
    monkeypatch.setattr(settings.llm, "provider", "ollama")
    monkeypatch.setattr("helioai.tools.sandbox._bwrap_works", lambda: False)
    monkeypatch.setattr("helioai.tools.sandbox._isolation_gap", lambda: "server is not root")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    return tmp_path


def test_every_check_runs_and_reports_a_known_status(quiet_install):
    checks = doctor.run_checks()
    assert {c.status for c in checks} <= {doctor.OK, doctor.WARN, doctor.FAIL}
    names = [c.name for c in checks]
    assert "search index" in names and "sandbox" in names and "llm provider" in names


def test_missing_index_is_a_blocking_failure_that_names_the_fix(quiet_install):
    (index,) = [c for c in doctor.run_checks() if c.name == "search index"]
    assert index.status == doctor.FAIL
    assert "helioai index" in index.detail


def test_sandbox_fallback_is_a_warning_with_the_reason(quiet_install):
    (sb,) = [c for c in doctor.run_checks() if c.name == "sandbox"]
    assert sb.status == doctor.WARN
    assert "plain subprocess" in sb.detail and "not root" in sb.detail


def test_functional_bwrap_is_ok(quiet_install, monkeypatch):
    monkeypatch.setattr("helioai.tools.sandbox._bwrap_works", lambda: True)
    (sb,) = [c for c in doctor.run_checks() if c.name == "sandbox"]
    assert sb.status == doctor.OK


def test_a_provider_without_its_key_fails_with_the_factory_message(quiet_install, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings.llm, "provider", "groq")
    monkeypatch.setattr(settings.llm.groq, "api_key", "")
    (llm,) = [c for c in doctor.run_checks() if c.name == "llm provider"]
    assert llm.status == doctor.FAIL and "GROQ_API_KEY" in llm.detail


def test_a_built_index_reports_its_product_count(quiet_install):
    import chromadb

    from helioai.config import settings
    from helioai.indexer import HNSW_SYNC_THRESHOLD

    client = chromadb.PersistentClient(path=str(settings.rag.chroma_dir))
    coll = client.create_collection(
        settings.rag.collection_name,
        configuration={"hnsw": {"space": "cosine", "sync_threshold": HNSW_SYNC_THRESHOLD}},
    )
    coll.add(ids=["a", "b"], embeddings=[[0.0, 1.0], [1.0, 0.0]], documents=["x", "y"])
    (index,) = [c for c in doctor.run_checks() if c.name == "search index"]
    assert index.status == doctor.OK and index.detail.startswith("2 products")


def test_an_index_built_before_writes_were_persisted_at_once_is_a_warning(quiet_install):
    """Its unpersisted tail is replayed into the graph at every start, in a varying order,
    so the same query can rank differently from one process to the next."""
    import chromadb

    from helioai.config import settings

    client = chromadb.PersistentClient(path=str(settings.rag.chroma_dir))
    coll = client.create_collection(settings.rag.collection_name, metadata={"hnsw:space": "cosine"})
    coll.add(ids=["a", "b"], embeddings=[[0.0, 1.0], [1.0, 0.0]], documents=["x", "y"])
    (index,) = [c for c in doctor.run_checks() if c.name == "search index"]
    assert index.status == doctor.WARN
    assert "sync_threshold 1000" in index.detail and "helioai index" in index.detail


def test_large_workspaces_are_flagged(quiet_install, monkeypatch):
    ws = quiet_install / "users" / "web" / "workspace" / "big_session"
    ws.mkdir(parents=True)
    (ws / "blob.npz").write_bytes(b"0" * 1024)
    monkeypatch.setattr(doctor, "_size_mb", lambda path: 4096.0)
    (w,) = [c for c in doctor.run_checks() if c.name == "workspaces"]
    assert w.status == doctor.WARN and "1 session dir" in w.detail


def test_a_crashing_check_is_reported_not_raised(quiet_install, monkeypatch):
    def boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr(doctor, "check_python", boom)
    (py,) = [c for c in doctor.run_checks() if c.name == "python"]
    assert py.status == doctor.FAIL and "kaboom" in py.detail


def test_json_report_and_exit_code(quiet_install, capsys):
    code = doctor.run_doctor(as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert code == 1  # the index is missing
    assert {"name", "status", "detail"} <= set(payload[0])


def test_text_report_counts_the_failures(quiet_install, capsys):
    doctor.run_doctor()
    out = capsys.readouterr().out
    assert "need attention" in out
    assert "✗ search index" in out


def test_online_probe_is_skipped_for_providers_without_a_models_endpoint(
    quiet_install, monkeypatch
):
    from helioai.config import settings

    monkeypatch.setattr(settings.llm, "provider", "gemini")
    check = doctor.check_provider_online()
    assert check.status == doctor.WARN and "not probed" in check.detail


def test_online_probe_reports_an_unreachable_endpoint(quiet_install, monkeypatch):
    import httpx

    from helioai.config import settings

    monkeypatch.setattr(settings.llm, "provider", "ollama")
    monkeypatch.setattr(settings.llm.ollama, "base_url", "http://127.0.0.1:9")

    def refuse(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", refuse)
    check = doctor.check_provider_online()
    assert check.status == doctor.FAIL and "ConnectError" in check.detail


def test_cli_routes_doctor_without_touching_the_workspace(monkeypatch, capsys):
    """`doctor` diagnoses; it must not create directories or clean anything up."""
    import sys

    import helioai.interfaces.cli as cli
    import helioai.workspace as ws

    def boom(name):
        def _fail(*a, **k):
            raise AssertionError(f"{name} must not run for doctor")

        return _fail

    monkeypatch.setattr(ws, "set_user", boom("set_user"))
    monkeypatch.setattr(ws, "cleanup_old_runs", boom("cleanup_old_runs"))
    monkeypatch.setattr(
        "helioai.doctor.run_checks", lambda online=False: [doctor.Check("x", doctor.OK, "fine")]
    )
    monkeypatch.setattr(sys, "argv", ["helioai", "doctor", "--json"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out)[0]["name"] == "x"


def test_help_lists_doctor():
    import helioai.interfaces.cli as cli

    assert "helioai doctor" in (cli.__doc__ or "")


def test_the_judged_answers_report_their_count_and_date(quiet_install, tmp_path):
    """The one part of the index code cannot rebuild carries its date into the report."""
    from helioai.doctor import check_judged
    from helioai.indexer import local_judged_path, save_judged

    check = check_judged()
    assert check.status == "ok" and check.name == "judged products"
    assert "snapshot 2026-09-22 (jev-1.13.0)" in check.detail and "82" in check.detail
    assert "local file" not in check.detail

    save_judged(
        local_judged_path(),
        {"date": "2026-10-01", "models": ["jev-1.13.0"]},
        {"cda/NEW/x": {"name": "x", "mtype": {"choice": "Waves", "confidence": 0.95}}},
    )
    assert "local file adds 1 products (2026-10-01)" in check_judged().detail
