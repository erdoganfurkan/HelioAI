"""Centralized configuration — loads .env once at startup.

`settings` is a module-level singleton imported everywhere.
Fails fast at import time if the configured LLM provider is missing an API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(_ROOT / ".env")


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


@dataclass
class LLMConfig:
    provider: str = "groq"
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class AgentConfig:
    max_iterations: int = 10


@dataclass
class RAGConfig:
    chroma_dir: Path = field(default_factory=lambda: _ROOT / "data" / "chroma")
    collection_name: str = "speasy_catalog"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_fetch_k: int = 20


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)


def _load() -> Settings:
    provider = os.environ.get("HELIOAI_LLM_PROVIDER", "groq").lower()
    max_iterations = int(os.environ.get("HELIOAI_MAX_ITERATIONS", "10"))

    s = Settings(
        llm=LLMConfig(
            provider=provider,
            gemini=GeminiConfig(
                api_key=os.environ.get("GEMINI_API_KEY", ""),
            ),
            groq=GroqConfig(
                api_key=os.environ.get("GROQ_API_KEY", ""),
            ),
        ),
        agent=AgentConfig(max_iterations=max_iterations),
    )

    if provider == "groq" and not s.llm.groq.api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env (https://console.groq.com/keys)"
        )
    if provider == "gemini" and not s.llm.gemini.api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env (https://aistudio.google.com/apikey)"
        )

    return s


settings = _load()
