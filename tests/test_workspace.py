def test_no_session_calls_share_one_scratch_dir():
    """A fresh mkdtemp per call orphaned ~200 dirs per notebook run."""
    import helioai.workspace as ws

    ws._no_session_dir.cache_clear()
    ws.reset_session(ws._current_session.set(None))
    ws.reset_label(ws._current_label.set(None))
    try:
        dirs = {ws.get_session_dir() for _ in range(5)}
    finally:
        ws._no_session_dir.cache_clear()
    assert len(dirs) == 1


def test_is_under_workspace_does_not_assume_a_posix_separator(tmp_path, monkeypatch):
    """The check compared against str(root) + "/", so Windows rejected every path.

    Fail-closed, so no leak — but /figure and /code returned 404 for everything the
    web UI produced. Asserting on the separator the OS actually uses catches it here
    instead of only on windows-latest.
    """
    import os

    import helioai.workspace as ws

    monkeypatch.setattr(ws, "_users_root", lambda: tmp_path)
    inside = tmp_path / "web" / "workspace" / "sess" / "fig_0.png"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"x")

    assert ws.is_under_workspace(inside)
    assert ws.is_under_workspace(str(inside))
    assert os.sep in str(inside)
    assert ws.is_under_workspace(tmp_path), "the root itself is under the root"
    assert not ws.is_under_workspace(tmp_path.parent / "elsewhere")
    assert not ws.is_under_workspace(inside.parent / ".." / ".." / ".." / ".." / "etc")


def test_session_label_cannot_escape_the_workspace(tmp_path, monkeypatch):
    """The slug was sanitised, the session_id suffix was not.

    `make_session_label("hi", "../../etc")` produced a label whose path components
    climbed out of the workspace — and that label is what the export filename and the
    session delete are built from.
    """
    import helioai.workspace as ws

    monkeypatch.setattr(ws, "_users_root", lambda: tmp_path)

    label = ws.make_session_label("plot imf bz", "../../etc/passwd")
    assert "/" not in label and ".." not in label

    token = ws.set_label(label)
    try:
        d = ws.get_session_dir()
    finally:
        ws.reset_label(token)
    assert d.resolve().is_relative_to(tmp_path.resolve())


def test_safe_id_keeps_real_ids_intact_and_never_returns_empty():
    """Every id the project mints is a uuid4, so sanitising must be lossless."""
    import uuid

    import helioai.workspace as ws

    real = str(uuid.uuid4())
    assert ws.safe_id(real) == real
    assert ws.safe_id("../..") == "session"
    assert ws.safe_id("") == "session"
    assert "/" not in ws.safe_id("a/../../b")
