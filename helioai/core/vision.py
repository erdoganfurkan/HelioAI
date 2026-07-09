"""Stateless vision side-call: review sandbox figures after run_python.

The image is sent once, outside the conversation; only the short text verdict
enters the tool result (and thus the history), so the cost stays a few hundred
tokens per figure instead of re-sending images every turn. Never blocks the
loop: any failure logs a warning and returns the result unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json

from helioai.config import settings
from helioai.logging_config import get_logger

log = get_logger(__name__)

_PROMPT = (
    "You review matplotlib figures produced by a heliophysics data-analysis agent. "
    "Answer in at most 3 short lines, under 200 characters total, starting with "
    "OK or ISSUE. Flag only real rendering problems: empty axes / no visible data, "
    "missing axis labels or units, broken scale, unreadable legend. "
    "Do not describe or interpret the science."
)
_MAX_FIGS = 2
_THUMB = 768
_warned_no_creds = False


async def maybe_review(tool_name: str, result: str) -> tuple[str, str | None]:
    """Attach a vision verdict to a run_python result carrying figures."""
    if not settings.vision.enabled or tool_name != "run_python":
        return result, None
    try:
        data = json.loads(result)
    except (ValueError, TypeError):
        return result, None
    if not isinstance(data, dict) or data.get("error") or not data.get("figure_paths"):
        return result, None
    verdict = await _review(data["figure_paths"])
    if not verdict:
        return result, None
    data["figure_review"] = verdict
    return json.dumps(data, ensure_ascii=False, default=str), verdict


def _creds_ok() -> bool:
    if settings.vision.provider == "gemini":
        return bool(settings.llm.gemini.api_key)
    return bool(settings.llm.azure.api_key and settings.llm.azure.endpoint)


async def _review(figure_paths: list[str]) -> str | None:
    global _warned_no_creds
    if not _creds_ok():
        if not _warned_no_creds:
            log.warning("vision_disabled_no_credentials", provider=settings.vision.provider)
            _warned_no_creds = True
        return None
    try:
        images = [_png_b64(p) for p in figure_paths[:_MAX_FIGS]]
        verdict = await asyncio.wait_for(
            _call_vision(images, _PROMPT), timeout=settings.vision.timeout_s
        )
        return verdict.strip()[:300] or None
    except Exception as e:
        log.warning("vision_check_failed", error=str(e))
        return None


def _png_b64(path: str) -> str:
    from PIL import Image

    with Image.open(path) as im:
        im.thumbnail((_THUMB, _THUMB))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def _call_vision(pngs_b64: list[str], prompt: str) -> str:
    if settings.vision.provider == "gemini":
        return await _call_gemini(pngs_b64, prompt)
    return await _call_azure(pngs_b64, prompt)


async def _call_azure(pngs_b64: list[str], prompt: str) -> str:
    from openai import AsyncAzureOpenAI

    az = settings.llm.azure
    client = AsyncAzureOpenAI(
        api_key=az.api_key, azure_endpoint=az.endpoint, api_version=az.api_version
    )
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in pngs_b64
    ]
    r = await client.chat.completions.create(
        model=settings.vision.model or az.deployment,
        max_completion_tokens=2048,
        messages=[{"role": "user", "content": content}],
    )
    return r.choices[0].message.content or ""


async def _call_gemini(pngs_b64: list[str], prompt: str) -> str:
    from google import genai
    from google.genai import types as gt

    client = genai.Client(api_key=settings.llm.gemini.api_key)
    parts = [gt.Part(text=prompt)] + [
        gt.Part(inline_data=gt.Blob(mime_type="image/png", data=base64.b64decode(b)))
        for b in pngs_b64
    ]
    r = await client.aio.models.generate_content(
        model=settings.vision.model or settings.llm.gemini.model,
        contents=[gt.Content(role="user", parts=parts)],
    )
    return r.text or ""
