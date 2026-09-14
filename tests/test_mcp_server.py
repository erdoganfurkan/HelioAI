"""Tests for the MCP server handlers — no network I/O.

A subprocess is used where the behaviour under test *is* an import-time one, and the
sandbox tests run the real thing rather than mocking it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import MCPError, types
from mcp.types import CallToolRequestParams, ReadResourceRequestParams
from starlette.testclient import TestClient

import helioai.tools.setup  # noqa: F401
from helioai import mcp_server as ms
from helioai.config import settings


async def test_list_tools_count():
    """Every registered tool, no more, no fewer — `>= 10` let a tool silently vanish."""
    from helioai.tools.registry import registry

    result = await ms._list_tools(None, None)
    assert len(result.tools) == len(registry.list_tool_defs())


async def test_list_tools_contains_core():
    result = await ms._list_tools(None, None)
    names = {t.name for t in result.tools}
    assert "search_parameters" in names
    assert "run_python" in names
    assert "list_missions" in names


async def test_list_tools_schemas_are_objects():
    result = await ms._list_tools(None, None)
    for tool in result.tools:
        assert isinstance(tool.input_schema, dict)
        assert tool.input_schema.get("type") == "object"


async def test_list_tools_returns_mcp_tool_instances():
    result = await ms._list_tools(None, None)
    for tool in result.tools:
        assert isinstance(tool, types.Tool)


async def test_call_tool_unknown_returns_error():
    result = await ms._call_tool(None, CallToolRequestParams(name="does_not_exist", arguments={}))
    assert len(result.content) == 1
    assert isinstance(result.content[0], types.TextContent)
    assert result.is_error is True
    body = json.loads(result.content[0].text)
    assert "error" in body


async def test_call_tool_list_missions_returns_json():
    result = await ms._call_tool(None, CallToolRequestParams(name="list_missions", arguments={}))
    body = json.loads(result.content[0].text)
    assert "providers" in body
    assert isinstance(body["providers"], list)
    assert len(body["providers"]) > 0


async def test_call_tool_returns_text_content():
    result = await ms._call_tool(None, CallToolRequestParams(name="list_missions", arguments={}))
    assert all(isinstance(c, types.TextContent) for c in result.content)


async def test_call_tool_plasma_beta_correct_value():
    params = CallToolRequestParams(
        name="plasma_beta", arguments={"B_nT": 5.0, "n_cm3": 10.0, "T_eV": 10.0}
    )
    result = await ms._call_tool(None, params)
    body = json.loads(result.content[0].text)
    assert "beta" in body
    assert isinstance(body["beta"], float)
    assert body["beta"] > 0


async def test_call_tool_carries_the_payload_as_structured_content():
    """A client that supports `structuredContent` reads the object instead of parsing
    the text; the two must be the same result, and a failure has `isError` set."""
    params = CallToolRequestParams(
        name="plasma_beta", arguments={"B_nT": 5.0, "n_cm3": 10.0, "T_eV": 10.0}
    )
    result = await ms._call_tool(None, params)
    assert result.structured_content == json.loads(result.content[0].text)
    assert result.is_error is False

    failed = await ms._call_tool(None, CallToolRequestParams(name="does_not_exist", arguments={}))
    assert failed.is_error is True
    assert failed.structured_content["error"] == json.loads(failed.content[0].text)["error"]


async def test_call_tool_gyrofrequency_proton():
    result = await ms._call_tool(
        None, CallToolRequestParams(name="gyrofrequency", arguments={"B_nT": 10.0})
    )
    body = json.loads(result.content[0].text)
    assert "frequency_Hz" in body
    assert body["frequency_Hz"] > 0


async def test_call_tool_none_arguments():
    result = await ms._call_tool(None, CallToolRequestParams(name="list_missions", arguments=None))
    body = json.loads(result.content[0].text)
    assert "providers" in body


async def test_list_resources_includes_recipes_and_skills():
    result = await ms._list_resources(None, None)
    uris = [r.uri for r in result.resources]
    assert any(u.startswith("recipe://") for u in uris)
    assert any(u.startswith("skill://") for u in uris)


async def test_read_resource_recipe_returns_python_source():
    listing = await ms._list_resources(None, None)
    uri = next(r.uri for r in listing.resources if r.uri.startswith("recipe://"))
    result = await ms._read_resource(None, ReadResourceRequestParams(uri=uri))
    assert result.contents[0].mime_type == "text/x-python"
    assert result.contents[0].uri == uri


async def test_read_resource_skill_returns_markdown():
    listing = await ms._list_resources(None, None)
    uri = next(r.uri for r in listing.resources if r.uri.startswith("skill://"))
    result = await ms._read_resource(None, ReadResourceRequestParams(uri=uri))
    assert result.contents[0].mime_type == "text/markdown"


async def test_read_resource_unknown_recipe_raises():
    with pytest.raises(MCPError):
        await ms._read_resource(None, ReadResourceRequestParams(uri="recipe://does-not-exist"))


async def test_read_resource_unknown_scheme_raises():
    with pytest.raises(MCPError):
        await ms._read_resource(None, ReadResourceRequestParams(uri="bogus://x"))


def test_build_http_app_has_mcp_route():
    from starlette.applications import Starlette

    app = ms.build_http_app()
    assert isinstance(app, Starlette)
    paths = [str(r.path) for r in app.routes]
    assert any("/mcp" in p for p in paths)


def test_http_app_open_when_token_unset(monkeypatch):
    monkeypatch.setattr(ms.settings.mcp, "token", "")
    with TestClient(ms.build_http_app()) as client:
        assert client.post("/mcp", json={}).status_code != 401


def test_http_app_requires_token_when_configured(monkeypatch):
    monkeypatch.setattr(ms.settings.mcp, "token", "s3cret")
    with TestClient(ms.build_http_app()) as client:
        assert client.post("/mcp", json={}).status_code == 401
        assert (
            client.post("/mcp", json={}, headers={"Authorization": "Bearer wrong"}).status_code
            == 401
        )
        assert (
            client.post("/mcp", json={}, headers={"Authorization": "Bearer s3cret"}).status_code
            != 401
        )


def test_init_options_returns_initialization_options():
    from mcp.server.models import InitializationOptions

    opts = ms._init_options()
    assert isinstance(opts, InitializationOptions)


def test_initialize_advertises_the_package_version():
    """The installed 0.3.0 candidate answered `initialize` with an empty server version:
    `Server("helioai")` never passed one, so a client showed the name and nothing else."""
    import helioai

    opts = ms._init_options()
    assert opts.server_name == "helioai"
    assert opts.server_version == helioai.__version__


def test_arg_helper_found():
    assert ms._arg(["--host", "0.0.0.0", "--port", "9000"], "--host", "127.0.0.1") == "0.0.0.0"


def test_arg_helper_default():
    assert ms._arg(["--http"], "--port", "8765") == "8765"


def test_arg_helper_flag_at_end():
    assert ms._arg(["--host"], "--host", "127.0.0.1") == "127.0.0.1"


class _RecordingLogger:
    def __init__(self):
        self.events = []

    def warning(self, event, **kw):
        self.events.append(("warning", event))

    def error(self, event, **kw):
        self.events.append(("error", event))


def test_http_refused_without_token_on_public_interface(monkeypatch):
    """--http exposes run_python; on a public interface with no token it must refuse to start.

    Over stdio that is the MCP contract — the client owns the process. Over HTTP on a
    public interface with no auth it is remote code execution, so this is now a hard
    startup failure rather than a warning someone might miss.
    """
    rec = _RecordingLogger()
    monkeypatch.setattr(ms, "get_logger", lambda _name: rec)
    monkeypatch.setattr(ms, "serve_http", lambda host, port: pytest.fail("must not start"))
    monkeypatch.setattr(ms.settings.mcp, "token", "")
    monkeypatch.setattr(ms.sys, "argv", ["helioai-mcp", "--http", "--host", "0.0.0.0"])

    with pytest.raises(SystemExit):
        ms.main()

    assert ("error", "mcp_http_refused_without_auth") in rec.events


def test_http_on_loopback_stays_quiet(monkeypatch):
    """The default bind is the supported one — logging on it trains people to ignore logs."""
    rec = _RecordingLogger()
    monkeypatch.setattr(ms, "get_logger", lambda _name: rec)
    monkeypatch.setattr(ms, "serve_http", lambda host, port: None)
    monkeypatch.setattr(ms.sys, "argv", ["helioai-mcp", "--http"])

    ms.main()

    assert rec.events == []


def test_http_on_public_interface_with_token_starts(monkeypatch):
    rec = _RecordingLogger()
    monkeypatch.setattr(ms, "get_logger", lambda _name: rec)
    started = []
    monkeypatch.setattr(ms, "serve_http", lambda host, port: started.append((host, port)))
    monkeypatch.setattr(ms.settings.mcp, "token", "secret")
    monkeypatch.setattr(ms.sys, "argv", ["helioai-mcp", "--http", "--host", "0.0.0.0"])

    ms.main()

    assert started == [("0.0.0.0", 8765)]


class _FakeConnection:
    """Stand-in for the SDK's Connection — the only object that lives as long as the link."""


class _FakeSession:
    """Stand-in for ServerSession.

    The SDK builds a fresh one for every inbound message (`runner._make_context`),
    so it is the connection it carries, not the session, that identifies the client.
    """

    def __init__(self, connection: object):
        self._connection = connection


def _ctx(connection: object | None = None):
    """One request's context. A new ServerSession each time, as the SDK does."""
    return SimpleNamespace(session=_FakeSession(connection or _FakeConnection()))


async def _run(ctx, code: str) -> dict:
    result = await ms._call_tool(
        ctx, CallToolRequestParams(name="run_python", arguments={"code": code})
    )
    return json.loads(result.content[0].text)


async def test_two_mcp_calls_get_distinct_run_indices(monkeypatch, tmp_path):
    """Two run_python calls on one connection must not write the same code file.

    The whole point: without a session bound, `_run_idx` stays 0 and the second call
    silently overwrites the first one's code and figures.
    """
    ctx = _ctx()
    first = await _run(ctx, "print('first')")
    second = await _run(ctx, "print('second')")
    assert first["code_path"] != second["code_path"]


async def test_mcp_call_writes_under_its_own_user(monkeypatch, tmp_path):
    out = await _run(_ctx(), "print('hello')")
    assert str(tmp_path / "users" / "mcp") in out["code_path"]


async def test_two_mcp_connections_do_not_share_a_workspace(monkeypatch, tmp_path):
    a = await _run(_ctx(), "print('a')")
    b = await _run(_ctx(), "print('b')")
    assert Path(a["code_path"]).parent != Path(b["code_path"]).parent


async def test_one_connection_keeps_one_workspace_across_calls(monkeypatch, tmp_path):
    """Two calls on the same connection must land in the same session directory.

    The SDK mints a new ServerSession per inbound message, so keying the workspace on
    the session gives every call its own directory — which silently breaks the whole
    point: get_timeseries writes its npz in one directory and the next run_python
    looks for the manifest in another. Measured against a live client before this
    was pinned: load_data raised "no dataset manifest found" on the call right after
    a successful download.
    """
    connection = _FakeConnection()
    first = await _run(_ctx(connection), "print('first')")
    second = await _run(_ctx(connection), "print('second')")
    assert Path(first["code_path"]).parent == Path(second["code_path"]).parent
    assert first["code_path"] != second["code_path"]


def test_mcp_server_imports_without_any_llm_key():
    """The MCP server is a pure tool provider — the client brings its own model.

    A subprocess with the keys blanked is the only honest way to test import-time
    behaviour: by the time this test runs, config is long since imported.
    """
    env = {**os.environ, "HELIOAI_LLM_PROVIDER": "azure"}
    for key in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "GROQ_API_KEY", "GEMINI_API_KEY"):
        env[key] = ""
    proc = subprocess.run(
        [sys.executable, "-c", "import helioai.mcp_server"],
        capture_output=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode()


def test_build_llm_client_still_reports_the_missing_key(monkeypatch):
    """Moving the check must not lose it — the error only had to arrive later."""
    from helioai.core.llm.factory import build_llm_client

    monkeypatch.setattr(settings.llm.azure, "api_key", "")
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY is not set in .env"):
        build_llm_client("azure")


async def test_call_tool_rejected_private_argument_is_an_error():
    result = await ms._call_tool(
        None, CallToolRequestParams(name="run_python", arguments={"_plot_dir": "/etc"})
    )
    assert result.is_error is True


async def test_call_tool_success_is_not_an_error():
    result = await ms._call_tool(None, CallToolRequestParams(name="list_missions", arguments={}))
    assert result.is_error is False


async def test_sandbox_failure_is_reported_as_an_error(monkeypatch, tmp_path):
    result = await ms._call_tool(
        _ctx(), CallToolRequestParams(name="run_python", arguments={"code": "1 / 0"})
    )
    assert result.is_error is True


async def test_list_tools_marks_the_three_writers_as_not_read_only():
    result = await ms._list_tools(None, None)
    writers = {t.name for t in result.tools if not t.annotations.read_only_hint}
    assert writers == {"run_python", "run_recipe", "save_catalog"}


async def test_list_tools_annotates_every_tool():
    result = await ms._list_tools(None, None)
    assert all(t.annotations is not None for t in result.tools)


async def test_list_prompts_offers_every_skill():
    from helioai.core.skills_loader import list_skills

    result = await ms._list_prompts(None, None)
    assert {p.name for p in result.prompts} == {m.name for m in list_skills()}


async def test_get_prompt_returns_the_skill_body():
    from mcp.types import GetPromptRequestParams

    result = await ms._get_prompt(None, GetPromptRequestParams(name="plotting", arguments=None))
    assert result.messages[0].role == "user"
    assert len(result.messages[0].content.text) > 100


async def test_get_prompt_appends_the_task_argument():
    from mcp.types import GetPromptRequestParams

    result = await ms._get_prompt(
        None,
        GetPromptRequestParams(name="plotting", arguments={"task": "plot IMF Bz for 2015-03-17"}),
    )
    assert result.messages[0].content.text.endswith("plot IMF Bz for 2015-03-17")


async def test_get_prompt_unknown_skill_raises():
    from mcp.types import GetPromptRequestParams

    with pytest.raises(MCPError):
        await ms._get_prompt(None, GetPromptRequestParams(name="not-a-skill", arguments=None))


async def test_skills_stay_available_as_resources_too():
    """Prompts are added, not swapped in — an existing client reading skill:// keeps working."""
    result = await ms._list_resources(None, None)
    assert any(str(r.uri).startswith("skill://") for r in result.resources)


async def test_resource_templates_declare_both_schemes():
    result = await ms._list_resource_templates(None, None)
    assert {t.uri_template for t in result.resource_templates} == {
        "recipe://{name}",
        "skill://{name}",
    }


async def test_run_python_returns_the_figure_as_image_content(monkeypatch, tmp_path):
    result = await ms._call_tool(
        _ctx(),
        CallToolRequestParams(
            name="run_python",
            arguments={"code": "import matplotlib.pyplot as plt; plt.plot([1, 2, 3]); plt.show()"},
        ),
    )
    images = [c for c in result.content if isinstance(c, types.ImageContent)]
    assert len(images) == 1
    assert images[0].mime_type == "image/png"
    assert result.content[0].type == "text"


async def test_returned_figure_is_downscaled(monkeypatch, tmp_path):
    import base64
    import io

    from PIL import Image

    code = (
        "import matplotlib.pyplot as plt\n"
        "fig = plt.figure(figsize=(30, 20), dpi=100)\n"
        "plt.plot([1, 2, 3])\n"
        "plt.show()\n"
    )
    result = await ms._call_tool(
        _ctx(), CallToolRequestParams(name="run_python", arguments={"code": code})
    )
    image = next(c for c in result.content if isinstance(c, types.ImageContent))
    with Image.open(io.BytesIO(base64.b64decode(image.data))) as img:
        assert max(img.size) <= ms.FIGURE_MAX_PX


async def test_tool_without_a_figure_returns_text_only():
    result = await ms._call_tool(None, CallToolRequestParams(name="list_missions", arguments={}))
    assert all(isinstance(c, types.TextContent) for c in result.content)


def test_unreadable_figure_is_skipped_not_raised():
    """A figure that cannot be read must not fail a call that already succeeded."""
    assert ms._figure_content({"figure_paths": ["/nope/missing.png"]}) == []


def test_figure_content_ignores_non_dict_payloads():
    assert ms._figure_content("a remote tool's plain text") == []
