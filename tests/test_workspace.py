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
