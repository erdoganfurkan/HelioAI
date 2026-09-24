"""Replay of a session recorded before the event journal existed.

Sessions written by earlier versions have messages and no journal, and the browser
rebuilt their conversation by re-parsing the JSON of every `tool` message and guessing,
from its shape, which figures, cards, catalogues, scripts and recipes belonged to which
answer. That guesswork lived in `app.py` for a year and is kept here, unchanged, for
those sessions only: `GET /api/sessions/{id}/events` serves the journal whenever there is
one, and the browser asks for this view only when there is not. Nothing new is ever
emitted through this path.
"""

from __future__ import annotations

import json
from pathlib import Path

from helioai.core.llm.base import Message


def messages_view(history: list[Message]) -> list[dict]:
    """Group a stored history into the `{"messages": [...]}` entries the browser replays.

    Artifacts accumulate from tool results and are attached to the assistant message
    that closes the turn. A turn cut short — browser closed mid-stream, iteration cap —
    has no such message, so its figures and scripts are flushed as an empty assistant
    entry instead: once at the next user message, so they cannot be pinned onto an
    unrelated later answer, and once at the end of the history, so they are not
    dropped altogether. Both happened in the audit replay.

    Args:
        history: The session's messages, in order.

    Returns:
        User, system (an automated correction) and assistant entries, each assistant
        entry carrying the figures, cards, catalogs, code and recipes produced before it.
    """
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
    return out
