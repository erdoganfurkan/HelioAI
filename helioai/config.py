"""Centralized configuration — loads .env once at startup.

`settings` is a module-level singleton imported everywhere.
Fails fast at import time if the configured LLM provider is missing an API key.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent

load_dotenv(_ROOT / ".env")

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
    deployment: str = "models-gpt-53-chat"
    api_version: str = "2024-12-01-preview"
    max_output_tokens: int = 2048
    temperature: float | None = None
    api_key: str = ""
    endpoint: str = ""


@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-flash"
    max_output_tokens: int = 4096
    temperature: float = 0.2
    api_key: str = ""


@dataclass
class GroqConfig:
    model: str = "llama-3.3-70b-versatile"
    max_output_tokens: int = 4096
    temperature: float = 0.2
    api_key: str = ""


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:14b-instruct"
    max_output_tokens: int = 4096
    temperature: float = 0.2
    api_key: str = ""


@dataclass
class LLMConfig:
    provider: str = "azure"
    azure: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class AgentConfig:
    max_iterations: int = 10


@dataclass
class RAGConfig:
    chroma_dir: Path = field(default_factory=lambda: _DATA / "chroma")
    collection_name: str = "speasy_catalog"
    catalogs_collection_name: str = "speasy_catalogs"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_fetch_k: int = 20
    hybrid_enabled: bool = True
    hybrid_fetch_k: int = 50
    rrf_k: int = 60


@dataclass
class WorkspaceConfig:
    workspace_dir: Path = field(default_factory=lambda: _DATA / "workspace")
    ttl_seconds: int = 86400 * 7  # 7 days


@dataclass
class ProfileConfig:
    profile_path: Path = field(default_factory=lambda: _DATA / "profile.md")


@dataclass
class RecipesConfig:
    recipes_dir: Path = field(default_factory=lambda: _PKG_RECIPES)


@dataclass
class CatalogsConfig:
    catalogs_dir: Path = field(default_factory=lambda: _DATA / "catalogs")


@dataclass
class LiteratureConfig:
    ads_token: str = ""


@dataclass
class MCPConfig:
    servers_json: str = ""


@dataclass
class VisionConfig:
    # Reviews sandbox figures with a multimodal side-call; only the text
    # verdict enters the history, never the image.
    enabled: bool = False
    provider: str = "azure"
    model: str = ""
    timeout_s: float = 20.0


@dataclass
class DevConfig:
    # Shared-secret that unlocks unrestricted LLM access (bypasses scope guardrail).
    # Empty (default) → no token is valid → all requests stay restricted.
    token: str = ""


@dataclass
class WebAuthConfig:
    # Nominative tokens for the web UI: {token: user_id}. Parsed from
    # HELIOAI_USERS="tok1:vincent,tok2:alice". Empty → no auth, single local user.
    # ponytail: env-driven map, fine for a handful of researchers; move to a DB
    # table if tokens must be added/revoked at runtime.
    users: dict[str, str] = field(default_factory=dict)


@dataclass
class Settings:
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


def _load() -> Settings:
    provider = os.environ.get("HELIOAI_LLM_PROVIDER", "azure").lower()
    max_iterations = int(os.environ.get("HELIOAI_MAX_ITERATIONS", "10"))

    data_dir = Path(os.environ.get("HELIOAI_DATA_DIR", str(_DATA)))
    workspace_dir = Path(os.environ.get("HELIOAI_WORKSPACE", str(_DATA / "workspace")))
    workspace_ttl = int(os.environ.get("HELIOAI_WORKSPACE_TTL_S", str(86400 * 7)))
    profile_path = Path(os.environ.get("HELIOAI_PROFILE", str(_DATA / "profile.md")))
    recipes_dir = Path(os.environ.get("HELIOAI_RECIPES_DIR", str(_PKG_RECIPES)))
    catalogs_dir = Path(os.environ.get("HELIOAI_CATALOGS_DIR", str(_DATA / "catalogs")))
    hybrid_enabled = os.environ.get("HELIOAI_RAG_HYBRID", "1") != "0"

    dev_token = os.environ.get("HELIOAI_DEV_TOKEN", "")
    web_users = _parse_users(os.environ.get("HELIOAI_USERS", ""))

    s = Settings(
        data_dir=data_dir,
        web_auth=WebAuthConfig(users=web_users),
        workspace=WorkspaceConfig(workspace_dir=workspace_dir, ttl_seconds=workspace_ttl),
        profile=ProfileConfig(profile_path=profile_path),
        recipes=RecipesConfig(recipes_dir=recipes_dir),
        catalogs=CatalogsConfig(catalogs_dir=catalogs_dir),
        literature=LiteratureConfig(ads_token=os.environ.get("ADS_API_TOKEN", "")),
        mcp=MCPConfig(servers_json=os.environ.get("HELIOAI_MCP_SERVERS", "")),
        vision=VisionConfig(
            enabled=os.environ.get("HELIOAI_VISION_ENABLED", "0") not in ("0", "", "false"),
            provider=os.environ.get("HELIOAI_VISION_PROVIDER", "azure").lower(),
            model=os.environ.get("HELIOAI_VISION_MODEL", ""),
        ),
        rag=RAGConfig(hybrid_enabled=hybrid_enabled),
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
            ),
            ollama=OllamaConfig(
                base_url=os.environ.get("HELIOAI_OLLAMA_URL", "http://localhost:11434"),
                model=os.environ.get("HELIOAI_OLLAMA_MODEL", "qwen2.5:14b-instruct"),
            ),
        ),
        agent=AgentConfig(max_iterations=max_iterations),
    )

    if provider == "azure":
        if not s.llm.azure.api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set in .env")
        if not s.llm.azure.endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set in .env")
    elif provider == "groq" and not s.llm.groq.api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env (https://console.groq.com/keys)")
    elif provider == "gemini" and not s.llm.gemini.api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in .env (https://aistudio.google.com/apikey)")

    return s


settings = _load()


def dev_unlock(supplied: str | None) -> bool:
    """True iff the supplied token matches the configured dev secret.

    Returns False when the server-side token is empty (guards against
    accidentally unlocking an unconfigured instance).
    """
    return (
        bool(settings.dev.token)
        and supplied is not None
        and hmac.compare_digest(supplied, settings.dev.token)
    )
