"""FastAPI web interface for HelioAI.

Single-user, no auth. Streams agent events as SSE.
Figures from the sandbox are served via /figure?path=<abs_path>.
"""

from __future__ import annotations

import hmac
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

    Args:
        x_helio_token: The `X-Helio-Token` header, absent in local dev.

    Returns:
        The user id owning storage for this request.

    Raises:
        HTTPException: 401 when users are configured and the token is unknown.
    """
    users = settings.web_auth.users
    if not users:
        return _DEFAULT_USER
    # Compared token by token in constant time, like the dev token and the MCP bearer:
    # a dict lookup leaks how much of a guess matched through its timing.
    if x_helio_token:
        for token, user_id in users.items():
            if hmac.compare_digest(token, x_helio_token):
                return user_id
    raise HTTPException(status_code=401, detail="Invalid or missing token")


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

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Everything the page needs is served from this origin — the markdown and syntax
# highlighting libraries are vendored under /static — so inline scripts injected
# through a tool result would have nowhere to run even if DOMPurify let one through.
# `unsafe-inline` for styles only: the UI sets a few style attributes from JS.
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'"
)


@app.middleware("http")
async def _security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", _CSP)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response


def harden_for_host(app: FastAPI, host: str) -> FastAPI:
    """Add the middleware a given bind address calls for, and return the app.

    Kept apart from `serve_web` so a test can build exactly what uvicorn will serve:
    added inside `serve_web`, the host guard was never on the `app` the TestClient
    imported, and the DNS-rebinding defence went untested for a year.

    Args:
        app: The FastAPI application.
        host: The address about to be bound.

    Returns:
        The same app, so the call reads as an expression.
    """
    if host in _LOOPBACK_HOSTS:
        # A loopback bind is not a boundary: any web page can resolve its own domain
        # to 127.0.0.1 and reach this server (DNS rebinding). Pinning Host costs
        # nothing here and CORS does not cover it.
        from starlette.middleware.trustedhost import TrustedHostMiddleware

        app.add_middleware(TrustedHostMiddleware, allowed_hosts=sorted(_LOOPBACK_HOSTS))
    else:
        log.warning("web_exposed_beyond_loopback", host=host)
    return app


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
) -> StreamingResponse:
    """Stream one agent turn as Server-Sent Events.

    Each agent event — tool calls, results, artifacts, sub-agent activity — is
    forwarded as it happens, which is what drives the live activity dock.

    Args:
        req: Body carrying the question and the session to continue.
        x_helio_dev_token: Optional dev token lifting the scope guardrail.
        user_id: Resolved by `require_user`.

    Returns:
        A `StreamingResponse` of SSE frames, one per agent event.
    """
    # Authenticated nominative users are trusted → unrestricted; the legacy dev
    # token still unlocks scope when no users are configured (local dev).
    restricted = not (bool(settings.web_auth.users) or dev_unlock(x_helio_dev_token))

    # stream_chat serialises turns per session with a lock; answering 409 here is
    # only so a second tab fails fast instead of looking hung while it queues.
    if store.is_busy(user_id, req.session_id):
        raise HTTPException(status_code=409, detail="a reply is already streaming for this session")

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


@app.get("/api/me")
async def me(user_id: str = Depends(require_user)) -> dict:
    """Who the caller is and what they have spent.

    The first thing a per-user quota needs is a number to compare against; until now
    nothing summed the token counts the providers report. Totals for today (UTC-ish:
    the last 24 h), the last 30 days and all time.

    Returns:
        `{"user_id", "usage": {"day", "month", "total"}}` — each a dict of
        prompt/completion/cached tokens and call count.
    """
    return {
        "user_id": user_id,
        "usage": {
            "day": store.usage_totals(user_id, since_days=1),
            "month": store.usage_totals(user_id, since_days=30),
            "total": store.usage_totals(user_id),
        },
    }


@app.get("/api/sessions")
async def list_sessions(user_id: str = Depends(require_user)) -> list:
    """List the calling user's sessions, most recent first.

    Args:
        user_id: Resolved by `require_user`; scopes the listing, so no caller
            can enumerate another's sessions.

    Returns:
        Session summaries, newest first.
    """
    return store.list_summaries(user_id)


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, user_id: str = Depends(require_user)) -> dict:
    """Replay a session: its messages plus any figures and figure reviews.

    Artifacts accumulate from tool results and are attached to the assistant message
    that closes the turn. A turn cut short — browser closed mid-stream, iteration cap —
    has no such message, so its figures and scripts are flushed as an empty assistant
    entry instead: once at the next user message, so they cannot be pinned onto an
    unrelated later answer, and once at the end of the history, so they are not
    dropped altogether. Both happened in the audit replay.

    Args:
        session_id: Session to replay.
        user_id: Resolved by `require_user`; a session belonging to anyone else
            reads as empty rather than as a 403, which says nothing about
            whether it exists.

    Returns:
        `{"messages": [...]}` — the stored messages in order, each assistant
        entry carrying the figures, cards, catalogs, code and recipes that the
        tool calls before it produced.
    """
    history = store.get_or_create(user_id, session_id)
    out: list[dict] = []
    pending_figures: list[str] = []
    pending_cards: list[dict] = []
    pending_catalogs: list[dict] = []
    pending_code: list[dict] = []
    pending_recipes: list[dict] = []

    def _flush(content: str) -> None:
        nonlocal pending_figures, pending_cards, pending_catalogs, pending_code, pending_recipes
        entry: dict = {"role": "assistant", "content": content}
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
        if content or len(entry) > 2:
            out.append(entry)

    for m in history:
        if m.role == "user":
            _flush("")
            if m.origin:
                # HelioAI's own note (an automated correction), sent with the user role
                # because that is the only role the providers forward — shown as a
                # system line so the person is not credited with writing it.
                out.append({"role": "system", "origin": m.origin, "content": m.content})
            else:
                out.append({"role": "user", "content": m.content})
        elif m.role == "assistant" and m.content:
            _flush(m.content)
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
    _flush("")
    return {"messages": out}


@app.get("/api/profile")
async def get_profile(user_id: str = Depends(require_user)) -> dict:
    """Return the caller's profile markdown.

    Args:
        user_id: Resolved by `require_user`.

    Returns:
        `{"content": markdown}`, empty for a user who has written none.
    """
    p = _profile_path(user_id)
    content = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    return {"content": content}


@app.put("/api/profile")
async def put_profile(body: _ProfileBody, user_id: str = Depends(require_user)) -> dict:
    """Replace the caller's profile markdown.

    Args:
        body: New profile content, replacing the previous one wholesale.
        user_id: Resolved by `require_user`.

    Returns:
        `{"ok": True}` once written.
    """
    p = _profile_path(user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body.content, encoding="utf-8")
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(require_user)) -> dict:
    """Delete one of the caller's sessions and its workspace.

    Args:
        session_id: Session to delete. Sanitised through `safe_id` before it
            reaches the `rmtree` behind this route.
        user_id: Resolved by `require_user`; only the owner's tree is touched.

    Returns:
        `{"deleted": session_id}`, whether or not anything existed — a caller
        learns nothing about other users' session ids from the answer.
    """
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
async def export_notebook(session_id: str, user_id: str = Depends(require_user)) -> FileResponse:
    """Export a session as a standalone `.ipynb` and return it.

    Args:
        session_id: Session to export.
        user_id: Resolved by `require_user`.

    Returns:
        The notebook as a file download, built in memory rather than written to
        the workspace.
    """
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
async def serve_code(path: str, user_id: str = Depends(require_user)) -> PlainTextResponse:
    """Return a generated script, rewritten to standalone form.

    Ownership is checked against the caller before anything is read, so a path
    outside the caller's workspace is a 404 rather than a leak.

    Args:
        path: Absolute path of the generated script, as the artifact reported it.
        user_id: Resolved by `require_user`.

    Returns:
        The script rewritten to standalone speasy calls.

    Raises:
        HTTPException: 404 for a path outside the caller's workspace or absent.
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
async def serve_figure(path: str, user_id: str = Depends(require_user)) -> FileResponse:
    """Serve a figure (PNG or PDF) from the caller's workspace.

    Args:
        path: Absolute path of the figure, as the artifact reported it.
        user_id: Resolved by `require_user`.

    Returns:
        The file, with a content type derived from its extension. Only PNG and
        PDF are served, so a traversal that reached another file type still
        returns nothing.

    Raises:
        HTTPException: 404 outside the caller's workspace, or absent.
    """
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

    Args:
        host: Bind address. Anything but loopback exposes an arbitrary code
            executor; read SECURITY.md before changing it.
        port: TCP port.
    """
    import uvicorn

    from helioai.workspace import cleanup_old_runs

    refuse_unauthenticated_public_bind(host)
    harden_for_host(app, host)
    cleanup_old_runs()
    uvicorn.run(app, host=host, port=port)


def refuse_unauthenticated_public_bind(host: str) -> None:
    """Exit rather than serve `run_python` to a network with no one authenticated.

    The same rule `helioai-mcp --http` applies to a bind without a token: a public
    address with no `HELIOAI_USERS` is a deployment error, and a warning someone might
    read after the fact is not a boundary. `HELIOAI_ALLOW_UNAUTHENTICATED_PUBLIC=1` is
    the explicit opt-out for a container that binds 0.0.0.0 behind a loopback publish.

    Args:
        host: The address about to be bound.

    Raises:
        SystemExit: On a non-loopback host with neither users nor the opt-out.
    """
    if host in _LOOPBACK_HOSTS or settings.web_auth.users:
        return
    if settings.web_auth.allow_unauthenticated_public:
        log.warning("web_public_unauthenticated_by_choice", host=host)
        return
    log.error(
        "web_refused_without_auth",
        host=host,
        detail=(
            "set HELIOAI_USERS, bind to loopback, or set "
            "HELIOAI_ALLOW_UNAUTHENTICATED_PUBLIC=1 behind a loopback port publish"
        ),
    )
    raise SystemExit(1)
