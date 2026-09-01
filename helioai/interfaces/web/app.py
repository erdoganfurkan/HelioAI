"""FastAPI web interface for HelioAI.

Single-user, no auth. Streams agent events as SSE.
Figures from the sandbox are served via /figure?path=<abs_path>.
"""

from __future__ import annotations

import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import helioai.tools.setup  # noqa: F401 — registers all tools at import time
from helioai.config import dev_unlock, settings
from helioai.core.agent_loop import stream_chat
from helioai.core.llm.factory import build_llm_client
from helioai.core.session import store
from helioai.logging_config import get_logger
from helioai.workspace import is_under_workspace, user_home

log = get_logger(__name__)

_STATIC = Path(__file__).parent / "static"
_DEFAULT_USER = "web"


async def require_user(x_helio_token: str | None = Header(default=None)) -> str:
    """Resolve the caller's user_id from the X-Helio-Token header.

    No users configured (local dev) → single shared user, no auth. Once
    HELIOAI_USERS is set (deployment), a valid nominative token is required.
    """
    users = settings.web_auth.users
    if not users:
        return _DEFAULT_USER
    if not x_helio_token or x_helio_token not in users:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return users[x_helio_token]


def _profile_path(user_id: str) -> Path:
    return user_home(user_id) / "profile.md"


def _owns_path(user_id: str, path: str) -> bool:
    """True if `path` is physically under the user's own storage home.

    Storage is namespaced as <data>/users/<user>/workspace/... so ownership is
    a physical containment check. No-op (always True) when auth is disabled.
    """
    if not settings.web_auth.users:
        return True
    try:
        Path(path).resolve().relative_to((user_home(user_id) / "workspace").resolve())
    except (ValueError, OSError):
        return False
    return True


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from helioai.tools.mcp_client import discover_and_register

    await discover_and_register()
    yield


app = FastAPI(title="HelioAI", docs_url=None, redoc_url=None, lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


class _ChatRequest(BaseModel):
    message: str
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    provider: str | None = None


class _ProfileBody(BaseModel):
    content: str


@app.get("/")
async def index():
    """Serve the single-page web UI."""
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
async def health():
    """Liveness probe. Returns `{"status": "ok"}`."""
    return {"status": "ok"}


@app.get("/api/config")
async def api_config():
    """Server-side settings the UI cannot know on its own.

    The provider selector used to default to whichever option came first in the markup
    — `azure` — and sent it on every message, so a server configured for another
    provider was quietly overridden by the browser.
    """
    return {"provider": settings.llm.provider}


@app.post("/chat/stream")
async def chat_stream(
    req: _ChatRequest,
    x_helio_dev_token: str | None = Header(default=None),
    user_id: str = Depends(require_user),
):
    """Stream one agent turn as Server-Sent Events.

    Each agent event — tool calls, results, artifacts, sub-agent activity — is
    forwarded as it happens, which is what drives the live activity dock.
    """
    # Authenticated nominative users are trusted → unrestricted; the legacy dev
    # token still unlocks scope when no users are configured (local dev).
    restricted = not (bool(settings.web_auth.users) or dev_unlock(x_helio_dev_token))

    async def gen():
        llm = None
        try:
            llm = build_llm_client(req.provider)
            async for ev in stream_chat(
                llm, user_id, req.session_id, req.message, restricted=restricted
            ):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(e)}})}\n\n"
        finally:
            # One client per request, so the pool has to be released per request —
            # including when the browser disconnects mid-stream and this generator
            # is closed early.
            if llm is not None:
                await llm.aclose()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/api/sessions")
async def list_sessions(user_id: str = Depends(require_user)):
    """List the calling user's sessions, most recent first."""
    return store.list_summaries(user_id)


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, user_id: str = Depends(require_user)):
    """Replay a session: its messages plus any figures and figure reviews."""
    history = store.get_or_create(user_id, session_id)
    out: list[dict] = []
    pending_figures: list[str] = []
    pending_cards: list[dict] = []
    pending_catalogs: list[dict] = []
    pending_code: list[dict] = []
    pending_recipes: list[dict] = []
    for m in history:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant" and m.content:
            entry: dict = {"role": "assistant", "content": m.content}
            if pending_figures:
                entry["figures"] = pending_figures[:]
                pending_figures = []
            if pending_cards:
                entry["cards"] = pending_cards[:]
                pending_cards = []
            if pending_catalogs:
                entry["catalogs"] = pending_catalogs[:]
                pending_catalogs = []
            if pending_code:
                entry["code"] = pending_code[:]
                pending_code = []
            if pending_recipes:
                entry["recipes"] = pending_recipes[:]
                pending_recipes = []
            out.append(entry)
        elif m.role == "tool" and m.content:
            try:
                data = json.loads(m.content)
                if isinstance(data, dict):
                    if data.get("figure_paths"):  # run_python direct
                        pending_figures.extend(data["figure_paths"])
                    for card in data.get(
                        "cards", []
                    ):  # param_card()/document_method() in run_python
                        if not isinstance(card, dict):
                            continue
                        if card.get("kind") == "parameter_card":
                            pending_cards.append(card)
                        elif card.get("kind") == "method_used":
                            pending_recipes.append(
                                {
                                    "kind": "recipe_used",
                                    "name": card.get("name", ""),
                                    "reference": card.get("reference", ""),
                                    "description": card.get("method", ""),
                                }
                            )
                    if data.get("code_path"):  # run_python direct — artifact code
                        pending_code.append(
                            {
                                "kind": "code",
                                "code_path": data["code_path"],
                                "name": Path(data["code_path"]).name,
                                "n_lines": data.get("n_lines"),
                            }
                        )
                    if "metadata" in data and data.get("name") and data.get("code"):  # load_recipe
                        _meta = data.get("metadata") or {}
                        pending_recipes.append(
                            {
                                "kind": "recipe_used",
                                "name": data["name"],
                                "reference": _meta.get("reference", ""),
                                "description": _meta.get("description", ""),
                            }
                        )
                    if data.get("_kind") == "catalog_preview":  # get_catalog
                        pending_catalogs.append(
                            {
                                "kind": "catalog_preview",
                                "catalog_id": data.get("catalog_id"),
                                "name": data.get("name"),
                                "type": data.get("type"),
                                "nb_events_total": data.get("nb_events_total"),
                                "columns": data.get("columns", []),
                                "sample": (data.get("sample") or [])[:5],
                                "survey_start": data.get("survey_start"),
                                "survey_stop": data.get("survey_stop"),
                            }
                        )
                    if data.get("param_id") and "preview" in data:  # get_timeseries direct
                        pending_cards.append(
                            {
                                "kind": "parameter_card",
                                "param_id": data.get("param_id"),
                                "name": data.get("name"),
                                "mission": data.get("mission"),
                                "instrument": data.get("instrument"),
                                "units": data.get("units"),
                                "cadence": data.get("cadence"),
                                "components": data.get("components"),
                                "n_points": data.get("n_points"),
                                "start": data.get("start"),
                                "stop": data.get("stop"),
                            }
                        )
                    for art in data.get("artifacts", []):  # résultat sous-agent
                        if not isinstance(art, dict):
                            continue
                        if art.get("figure_paths"):
                            pending_figures.extend(art["figure_paths"])
                        if art.get("kind") == "parameter_card":
                            pending_cards.append(art)
                        if art.get("kind") == "catalog_preview":
                            pending_catalogs.append(art)
                        if art.get("kind") == "code":
                            pending_code.append(art)
                        if art.get("kind") == "recipe_used":
                            pending_recipes.append(art)
            except (ValueError, TypeError):
                pass
    return {"messages": out}


@app.get("/api/profile")
async def get_profile(user_id: str = Depends(require_user)):
    """Return the caller's profile markdown."""
    p = _profile_path(user_id)
    content = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    return {"content": content}


@app.put("/api/profile")
async def put_profile(body: _ProfileBody, user_id: str = Depends(require_user)):
    """Replace the caller's profile markdown."""
    p = _profile_path(user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body.content, encoding="utf-8")
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(require_user)):
    """Delete one of the caller's sessions and its workspace."""
    wdir = store.get_workspace_dir(user_id, session_id)
    store.reset(user_id, session_id)
    if wdir:
        # Containment, not trust: the label is persisted data, and a row written by
        # an older build (before session ids were sanitised) would walk this rmtree
        # straight out of the user's home.
        ws_root = (user_home(user_id) / "workspace").resolve()
        ws_path = (ws_root / wdir).resolve()
        if ws_path.is_relative_to(ws_root) and ws_path.exists():
            shutil.rmtree(ws_path, ignore_errors=True)
    return {"deleted": session_id}


@app.get("/api/export")
async def export_notebook(session_id: str, user_id: str = Depends(require_user)):
    """Export a session as a standalone `.ipynb` and return it."""
    from helioai.export import export_session_notebook

    if session_id not in store.all_sessions(user_id):
        raise HTTPException(status_code=404, detail="Unknown session")
    path = export_session_notebook(user_id, session_id)
    return FileResponse(
        path,
        media_type="application/x-ipynb+json",
        filename=path.name,
    )


@app.get("/code")
async def serve_code(path: str, user_id: str = Depends(require_user)):
    """Return a generated script, rewritten to standalone form.

    Ownership is checked against the caller before anything is read, so a path
    outside the caller's workspace is a 404 rather than a leak.
    """
    path = path.strip()
    if not is_under_workspace(path) or not _owns_path(user_id, path):
        log.warning("code_rejected", path=path, reason="outside workspace or not owner")
        raise HTTPException(status_code=404, detail="Not found")
    p = Path(path).resolve()
    if p.suffix != ".py" or not p.is_file():
        log.warning("code_rejected", path=path, reason="file not found or not .py")
        raise HTTPException(status_code=404, detail="Not found")
    from helioai.datastore import read_manifest
    from helioai.export import to_standalone

    manifest = read_manifest(p.parent)
    standalone = to_standalone(p.read_text(encoding="utf-8"), manifest, with_header=True)
    return PlainTextResponse(standalone)


_FIGURE_TYPES = {".png": "image/png", ".pdf": "application/pdf"}


@app.get("/figure")
async def serve_figure(path: str, user_id: str = Depends(require_user)):
    """Serve a figure (PNG or PDF) from the caller's workspace."""
    path = path.strip()
    if not is_under_workspace(path) or not _owns_path(user_id, path):
        log.warning("figure_rejected", path=path, reason="outside workspace or not owner")
        raise HTTPException(status_code=404, detail="Not found")
    p = Path(path).resolve()
    media_type = _FIGURE_TYPES.get(p.suffix.lower())
    if media_type is None:
        log.warning("figure_rejected", path=path, reason="unsupported type")
        raise HTTPException(status_code=404, detail="Not found")
    if not p.is_file():
        log.warning("figure_rejected", path=path, reason="file not found")
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(p, media_type=media_type)


def serve_web(host: str = "127.0.0.1", port: int = 7890) -> None:
    """Run the web UI with uvicorn.

    Binds to localhost by default. The open-source build ships no authentication
    and `run_python` executes model-written code, so do not expose this on a
    network without putting auth in front of it.
    """
    import uvicorn
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    from helioai.workspace import cleanup_old_runs

    if host in {"127.0.0.1", "localhost", "::1"}:
        # A loopback bind is not a boundary: any web page can resolve its own
        # domain to 127.0.0.1 and reach this server (DNS rebinding). Pinning Host
        # costs nothing here and CORS does not cover it.
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
    else:
        log.warning("web_exposed_beyond_loopback", host=host)
    cleanup_old_runs()
    uvicorn.run(app, host=host, port=port)
