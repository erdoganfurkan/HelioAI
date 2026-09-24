"""Centralized configuration — loads .env once at startup.

`settings` is a module-level singleton imported everywhere.

Importing this module never requires an API key. Credentials are validated where they
are used, by `llm.factory.build_llm_client`, which already raised the same errors — the
import-time copy only meant that surfaces needing no LLM at all could not start. The MCP
server is exactly that: a pure tool provider whose client brings its own model, and it
died on `AZURE_OPENAI_API_KEY is not set` before serving a single tool.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent

load_dotenv(_ROOT / ".env")
# Installed from PyPI, _ROOT is site-packages/ — no .env there ever exists, so the line
# above is silently a no-op and every key comes from the shell. README says
# `pip install helioai-agent` then "copy .env.example to .env": search upward from the
# working directory too, so that file is the one actually read. override=False (the
# load_dotenv default) on both calls, so a repo-clone .env still wins when both exist.
load_dotenv(find_dotenv(usecwd=True))

# Recipes ship inside the wheel: they are read-only assets, not user data.
# Override with HELIOAI_RECIPES_DIR to use your own set.
_PKG_RECIPES = _PKG / "data" / "recipes"

# Running from a git clone keeps writing to <repo>/data so existing installs and
# the developer workflow are untouched. Installed as a package, _ROOT would be
# site-packages/ — write user data to the XDG data dir instead.
_IN_REPO = (_ROOT / "pyproject.toml").is_file()


def _default_data_dir() -> Path:
    if _IN_REPO:
        return _ROOT / "data"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "helioai"


_DATA = _default_data_dir()


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI deployment settings.

    Azure routes by deployment name rather than model name, and reasoning models
    (GPT-5, o-series) reject an explicit temperature — hence `temperature=None`
    by default, which omits the field entirely.
    """

    deployment: str = "models-gpt-53-chat"
    api_version: str = "2024-12-01-preview"
    # 8192, not the 2048 this used to be and not the 4096 the other providers use.
    # Azure draws reasoning tokens from this same allowance, so a reasoning
    # deployment can spend the whole budget thinking and return an empty message —
    # no text, no tool call. At 2048 that happened on any request that generates a
    # file: the standalone-script export in examples/02 produced nothing at all.
    max_output_tokens: int = 8192
    temperature: float | None = None
    api_key: str = ""
    endpoint: str = ""


@dataclass
class GeminiConfig:
    """Google Gemini settings, used with the native `google-genai` client."""

    model: str = "gemini-2.5-flash"
    max_output_tokens: int = 4096
    temperature: float = 0.2
    api_key: str = ""


@dataclass
class GroqConfig:
    """Groq settings. Reached through the shared OpenAI-compatible client."""

    model: str = "llama-3.3-70b-versatile"
    max_output_tokens: int = 4096
    temperature: float = 0.2
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class OpenCodeConfig:
    """OpenCode's Zen gateway — OpenAI-compatible, whichever way you reach it: the
    flat-rate Go subscription, a BYOK-routed key, or any other model Zen hosts.

    `base_url` defaults to the Go-plan endpoint, since that flat-rate tier is what
    most accounts actually have. It is a DIFFERENT catalogue from the general Zen
    endpoint (`.../zen/v1`, no `/go/`) — that one serves premium/BYOK-only models
    (Claude, ...) a Go subscription cannot reach, confirmed by querying both
    `/models` endpoints directly. Override `HELIOAI_OPENCODE_URL` to the plain Zen
    path if your access is not the Go plan.

    No default model: what is reachable depends on your plan/BYOK setup and Zen's
    rotating catalogue (GLM, Kimi, DeepSeek, Qwen, MiniMax...). Set
    `HELIOAI_OPENCODE_MODEL` to the exact id from your dashboard — an empty string
    fails at the API with a clear "unknown model" rather than silently routing to a
    guessed default that may not exist on your plan.

    16384 output tokens because everything this gateway serves is a reasoning model,
    and reasoning, prose AND tool-call arguments all draw on the same budget. At 4096,
    DeepSeek v4's run_python calls (~12k chars of JSON once a plot script is in them)
    were cut mid-string: the model saw "missing 1 required positional argument: 'code'",
    could not know why, and burned five turns re-sending the same truncated call.
    Same lesson as Azure's 2048→8192, one provider later.
    """

    base_url: str = "https://opencode.ai/zen/go"
    model: str = ""
    max_output_tokens: int = 16384
    temperature: float = 0.2
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class OllamaConfig:
    """Local Ollama settings.

    Ollama serves an OpenAI-compatible API on `/v1`, so it needs no client of its
    own and no API key. Point `base_url` elsewhere for any other local endpoint.
    """

    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:14b-instruct"
    max_output_tokens: int = 4096
    temperature: float = 0.2
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Which provider to use, and the settings for each one."""

    provider: str = "azure"
    azure: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    opencode: OpenCodeConfig = field(default_factory=OpenCodeConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class AgentConfig:
    """Agent loop limits, and which model each delegated role runs on.

    `max_iterations` caps how many tool-calling rounds one question may take
    before the loop gives up, bounding both runtime and token spend.

    `role_models` maps a sub-agent role to `(provider, model)`. A `parameter_hunter`
    resolves ids from search results and needs no frontier model; a `data_analyst`
    writes the physics and does. Left empty, every role runs on the lead's client, as
    it always did. Parsed from `HELIOAI_ROLE_MODELS="parameter_hunter=groq:llama-3.3-70b-
    versatile,data_analyst=opencode"` — the model part is optional and defaults to the
    provider's configured model.

    `experiments` names the behaviours of `EXPERIMENTS` that are switched on, from
    `HELIOAI_EXPERIMENTS`. Empty — the default — is the loop as it behaved before any of
    them existed.
    """

    max_iterations: int = 10
    role_models: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    experiments: frozenset[str] = frozenset()


EXPERIMENTS: frozenset[str] = frozenset({"deferred_tools", "search_budget", "search_variables"})
"""The behaviours that change what the model sees or is told, each off by default.

Every one of them was committed on the strength of a single live run and never measured
against the loop it replaced. They stay in the code as named experiments so that each can
be switched on alone and compared on the same questions, N runs each
(`scripts/bench_live.py`):

- `deferred_tools`: the lead sees the formulary and catalogue tools only after asking
  for them with `search_tools`, and its prompt says so.
- `search_budget`: a `data_analyst` past three lookups (a `plasma_physicist` past two)
  with nothing downloaded receives a correction listing the ids it already has.
- `search_variables`: the top hit of a parameter search lists every variable of its
  dataset.

`final_answer` was one of them and is now the default: on the third bench (34 clean runs,
four configurations) it cost nothing on any question, re-enabled the claim verdict — zero
contradictions over nine — and named rejected candidates as often as the loop it replaced.
The two search experiments showed no value there (eight id-resolution runs correct without
them); they stay off until a question that fails without them is recorded.
"""


def _parse_experiments(raw: str) -> frozenset[str]:
    """Parse HELIOAI_EXPERIMENTS='final_answer,deferred_tools' → the set of names.

    Kept lenient on purpose: this runs at `import helioai.config`, which must not fail —
    the MCP server and the web app start from it before any model is built. A misspelt
    name is refused by `validate_experiments`, called where the API key is already
    validated (`build_llm_client`, `helioai doctor`), so nothing that runs the agent can
    measure an experiment that silently does nothing.
    """
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def validate_experiments(experiments: frozenset[str] | None = None) -> frozenset[str]:
    """Refuse unknown experiment names — an error, not a warning.

    An experiment that silently does nothing would be measured as if it did, and the
    comparison would be wrong without anyone knowing.

    Args:
        experiments: The set to check; `settings.agent.experiments` when None.

    Returns:
        The same set, when every name is known.

    Raises:
        RuntimeError: Naming the unknown entries and the known ones.
    """
    if experiments is None:
        experiments = settings.agent.experiments
    unknown = experiments - EXPERIMENTS
    if unknown:
        raise RuntimeError(
            f"HELIOAI_EXPERIMENTS: unknown {sorted(unknown)}; known: {sorted(EXPERIMENTS)}"
        )
    return experiments


@dataclass
class RAGConfig:
    """Parameter search settings.

    Retrieval is hybrid: dense embeddings for descriptions, BM25 for exact tokens
    like `BGSEc`, fused by Reciprocal Rank Fusion with parameter `rrf_k`.

    There is no cross-encoder reranking stage, and that is a measured decision, not
    an omission: a generic MS MARCO cross-encoder (`ms-marco-MiniLM-L-6-v2`) was tried
    over the fused candidates and *degraded* results — trained on web prose, it
    discards the dense+sparse consensus that makes exact-code matching work. The
    plumbing sat disabled for a year and was removed; only a domain-tuned reranker
    would be worth adding back.
    """

    chroma_dir: Path = field(default_factory=lambda: _DATA / "chroma")
    collection_name: str = "speasy_catalog"
    catalogs_collection_name: str = "speasy_catalogs"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hybrid_enabled: bool = True
    hybrid_fetch_k: int = 50
    rrf_k: int = 60


@dataclass
class WorkspaceConfig:
    """Retention of per-session working directories.

    Their location is not a setting: `workspace.user_home` derives it from `data_dir`
    per user. A `workspace_dir` field (and `HELIOAI_WORKSPACE`) used to sit here, read
    from the environment and consumed by nothing since storage became per-user.
    """

    ttl_seconds: int = 86400 * 7  # 7 days


@dataclass
class ProfileConfig:
    """Legacy location of the single-user profile, kept for `helioai migrate-storage`.

    The agent reads `users/<user>/profile.md` (`workspace.user_home`) since storage
    became per user; `helioai profile`, `%helioai_profile` and the web UI all edit that
    file. Nothing injects this path any more, so `HELIOAI_PROFILE` — which only moved
    it — was a knob that did nothing, and is gone. The path stays as the place the
    migration looks for a profile written by an older install.
    """

    profile_path: Path = field(default_factory=lambda: _DATA / "profile.md")


@dataclass
class RecipesConfig:
    """Where scientific recipes are loaded from.

    Defaults to the copy shipped inside the package so `pip install` works;
    override with `HELIOAI_RECIPES_DIR` to use your own set.
    """

    recipes_dir: Path = field(default_factory=lambda: _PKG_RECIPES)


@dataclass
class CatalogsConfig:
    """Where user-saved event catalogs are written, in speasy format."""

    catalogs_dir: Path = field(default_factory=lambda: _DATA / "catalogs")


@dataclass
class LiteratureConfig:
    """NASA ADS credentials for `find_papers`. Free token, no key means no tool."""

    ads_token: str = ""


@dataclass
class MCPConfig:
    """Remote MCP servers to mount, plus auth for HelioAI's own MCP HTTP transport."""

    servers_json: str = ""
    # Shared secret required as `Authorization: Bearer <token>` on the HTTP transport
    # (stdio needs none — the client owns the process). Empty (default) → no auth
    # required on loopback; a non-loopback bind then refuses to start (mcp_server.main).
    token: str = ""


@dataclass
class VisionConfig:
    """Multimodal review of generated figures.

    A stateless side-call outside the agent loop: the image is downscaled, sent
    once, and only the text verdict enters the history — never the image, which
    would otherwise be resent on every subsequent turn. Off by default.
    """

    # Reviews sandbox figures with a multimodal side-call; only the text
    # verdict enters the history, never the image.
    enabled: bool = False
    provider: str = "azure"
    model: str = ""
    timeout_s: float = 20.0


@dataclass
class DevConfig:
    """Shared secret unlocking unrestricted mode past the heliophysics guardrail.

    Empty by default, which means no token is valid and every request stays
    scoped. Compared in constant time.
    """

    # Shared-secret that unlocks unrestricted LLM access (bypasses scope guardrail).
    # Empty (default) → no token is valid → all requests stay restricted.
    token: str = ""


@dataclass
class WebAuthConfig:
    """Nominative tokens for the web UI, parsed from `HELIOAI_USERS`.

    Empty means no authentication and a single local user, which is the intended
    behaviour for local development only.
    """

    # Nominative tokens for the web UI: {token: user_id}. Parsed from
    # HELIOAI_USERS="tok1:vincent,tok2:alice". Empty → no auth, single local user.
    # ponytail: env-driven map, fine for a handful of researchers; move to a DB
    # table if tokens must be added/revoked at runtime.
    users: dict[str, str] = field(default_factory=dict)
    # `serve --web` refuses a non-loopback bind with no users configured, the way the
    # MCP HTTP server refuses one without a token: run_python is arbitrary code
    # execution. The one legitimate exception is a container, which must bind 0.0.0.0
    # inside its own network namespace while the host publishes the port on loopback —
    # docker-compose.yml sets HELIOAI_ALLOW_UNAUTHENTICATED_PUBLIC=1 for exactly that.
    allow_unauthenticated_public: bool = False


@dataclass
class Settings:
    """Root settings object.

    Imported as the module-level `settings` singleton and read everywhere; built
    once at import by `_load()`. Credentials are not checked here — importing must
    work with no key at all — but in `llm.factory.build_llm_client`, where the
    selected provider is actually used.
    """

    data_dir: Path = field(default_factory=lambda: _DATA)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    recipes: RecipesConfig = field(default_factory=RecipesConfig)
    catalogs: CatalogsConfig = field(default_factory=CatalogsConfig)
    literature: LiteratureConfig = field(default_factory=LiteratureConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    dev: DevConfig = field(default_factory=DevConfig)
    web_auth: WebAuthConfig = field(default_factory=WebAuthConfig)


def _parse_users(raw: str) -> dict[str, str]:
    """Parse HELIOAI_USERS='tok1:vincent,tok2:alice' → {token: user_id}."""
    users: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        token, user_id = pair.split(":", 1)
        token, user_id = token.strip(), user_id.strip()
        if token and user_id:
            users[token] = user_id
    return users


def _parse_role_models(raw: str) -> dict[str, tuple[str, str | None]]:
    """Parse HELIOAI_ROLE_MODELS='role=provider:model,role=provider' → {role: (provider, model)}.

    Provider names are lowercased like `HELIOAI_LLM_PROVIDER`; a model is taken verbatim
    since providers are case-sensitive about theirs. Malformed pairs are skipped rather
    than fatal: a typo in an optimisation must not stop the agent from answering.
    """
    out: dict[str, tuple[str, str | None]] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        role, target = pair.split("=", 1)
        role, target = role.strip(), target.strip()
        if not role or not target:
            continue
        provider, _, model = target.partition(":")
        out[role] = (provider.strip().lower(), model.strip() or None)
    return out


def _parse_headers(raw: str) -> dict[str, str]:
    """Parse HELIOAI_<PROVIDER>_HEADERS='x-session={uuid},x-team=plasma' → {name: value}.

    Extra HTTP headers for an OpenAI-compatible endpoint. Corporate proxies, access
    gateways and metered relays each want their own, and hard-coding one vendor's in
    the factory both dates the code and ships a policy decision nobody asked for.

    A value of `{uuid}` is replaced with a fresh random id when the client is built, so
    a gateway that wants a per-conversation token gets a real one instead of the same
    literal string forever.
    """
    headers: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, value = pair.split("=", 1)
        name, value = name.strip(), value.strip()
        if name:
            headers[name] = value
    return headers


def _load() -> Settings:
    provider = os.environ.get("HELIOAI_LLM_PROVIDER", "azure").lower()
    max_iterations = int(os.environ.get("HELIOAI_MAX_ITERATIONS", "10"))
    # One knob for every provider rather than four: what a user wants when a long
    # generation comes back empty is simply "give the model more room", and the
    # error raised in that case points here. Left unset, each provider keeps its
    # own default.
    max_out = os.environ.get("HELIOAI_MAX_OUTPUT_TOKENS", "")
    out_override = int(max_out) if max_out.strip().isdigit() else None

    # Every derived path hangs off data_dir, read first — not off the module-level
    # default computed before the variable was. Deriving them from `_DATA` honoured
    # HELIOAI_DATA_DIR for sessions and user homes only, and left the index, the
    # catalogues and the profile in the default tree: a Docker volume held two trees.
    data_dir = Path(os.environ.get("HELIOAI_DATA_DIR", str(_DATA)))
    workspace_ttl = int(os.environ.get("HELIOAI_WORKSPACE_TTL_S", str(86400 * 7)))
    recipes_dir = Path(os.environ.get("HELIOAI_RECIPES_DIR", str(_PKG_RECIPES)))
    catalogs_dir = Path(os.environ.get("HELIOAI_CATALOGS_DIR", str(data_dir / "catalogs")))
    hybrid_enabled = os.environ.get("HELIOAI_RAG_HYBRID", "1") != "0"

    dev_token = os.environ.get("HELIOAI_DEV_TOKEN", "")
    web_users = _parse_users(os.environ.get("HELIOAI_USERS", ""))

    s = Settings(
        data_dir=data_dir,
        web_auth=WebAuthConfig(
            users=web_users,
            allow_unauthenticated_public=os.environ.get("HELIOAI_ALLOW_UNAUTHENTICATED_PUBLIC", "0")
            not in ("0", "", "false"),
        ),
        workspace=WorkspaceConfig(ttl_seconds=workspace_ttl),
        profile=ProfileConfig(profile_path=data_dir / "profile.md"),
        recipes=RecipesConfig(recipes_dir=recipes_dir),
        catalogs=CatalogsConfig(catalogs_dir=catalogs_dir),
        literature=LiteratureConfig(ads_token=os.environ.get("ADS_API_TOKEN", "")),
        mcp=MCPConfig(
            servers_json=os.environ.get("HELIOAI_MCP_SERVERS", ""),
            token=os.environ.get("HELIOAI_MCP_TOKEN", ""),
        ),
        vision=VisionConfig(
            enabled=os.environ.get("HELIOAI_VISION_ENABLED", "0") not in ("0", "", "false"),
            provider=os.environ.get("HELIOAI_VISION_PROVIDER", "azure").lower(),
            model=os.environ.get("HELIOAI_VISION_MODEL", ""),
        ),
        rag=RAGConfig(chroma_dir=data_dir / "chroma", hybrid_enabled=hybrid_enabled),
        dev=DevConfig(token=dev_token),
        llm=LLMConfig(
            provider=provider,
            azure=AzureOpenAIConfig(
                deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "models-gpt-53-chat"),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
                endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            ),
            gemini=GeminiConfig(
                api_key=os.environ.get("GEMINI_API_KEY", ""),
            ),
            groq=GroqConfig(
                api_key=os.environ.get("GROQ_API_KEY", ""),
                headers=_parse_headers(os.environ.get("HELIOAI_GROQ_HEADERS", "")),
            ),
            opencode=OpenCodeConfig(
                base_url=os.environ.get("HELIOAI_OPENCODE_URL", "https://opencode.ai/zen/go"),
                model=os.environ.get("HELIOAI_OPENCODE_MODEL", ""),
                api_key=os.environ.get("OPENCODE_API_KEY", ""),
                headers=_parse_headers(os.environ.get("HELIOAI_OPENCODE_HEADERS", "")),
            ),
            ollama=OllamaConfig(
                base_url=os.environ.get("HELIOAI_OLLAMA_URL", "http://localhost:11434"),
                model=os.environ.get("HELIOAI_OLLAMA_MODEL", "qwen2.5:14b-instruct"),
                headers=_parse_headers(os.environ.get("HELIOAI_OLLAMA_HEADERS", "")),
            ),
        ),
        agent=AgentConfig(
            max_iterations=max_iterations,
            role_models=_parse_role_models(os.environ.get("HELIOAI_ROLE_MODELS", "")),
            experiments=_parse_experiments(os.environ.get("HELIOAI_EXPERIMENTS", "")),
        ),
    )

    if out_override:
        for f in fields(s.llm):
            if f.name != "provider":
                getattr(s.llm, f.name).max_output_tokens = out_override

    return s


settings = _load()


def dev_unlock(supplied: str | None) -> bool:
    """True iff the supplied token matches the configured dev secret.

    Args:
        supplied: Token offered by the caller — a `--dev` flag or a request
            header. Compared with `hmac.compare_digest`, so a wrong token costs
            the same time as a right one.

    Returns:
        True only when a dev token is configured and the supplied one matches.
        An unconfigured instance answers False for every input, including
        `None` and the empty string, so a blank secret cannot unlock anything.
    """
    return (
        bool(settings.dev.token)
        and supplied is not None
        and hmac.compare_digest(supplied, settings.dev.token)
    )
