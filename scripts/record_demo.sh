#!/usr/bin/env bash
# Record a real HelioAI CLI session and turn it into a GIF for the README and docs.
#
# Why the CLI and not the web UI: the CLI already renders the whole reasoning chain as
# text — the plan, every tool call and result, sub-agent spawns, skills loaded, and the
# provenance verdict. A terminal recording is kilobytes instead of megabytes, and the dead
# air between LLM turns is compressed by a playback speed factor rather than by editing,
# so nothing is staged.
#
# This records a REAL session. It needs a working LLM provider and reachable data
# services. Run it during a demo rehearsal, when you have a session that works anyway.
#
# Usage:
#     scripts/record_demo.sh                       # record, then convert
#     scripts/record_demo.sh --convert-only        # re-convert the last cast (tweak speed)
#     SPEED=3 scripts/record_demo.sh               # override playback speed
#
# Output: docs/assets/demo.gif  (+ the raw docs/assets/demo.cast, kept — it is tiny and
# lets you re-render at a different speed or theme without re-running the agent).

set -euo pipefail

cd "$(dirname "$0")/.."

CAST="docs/assets/demo.cast"
GIF="docs/assets/demo.gif"

# agg already caps idle time at 5 s, so most of the waiting is gone before SPEED applies:
# speed trims how long the text stays on screen, not the pauses. 3 keeps a ~160 s session
# under 20 s and still readable; drop to 2 if a viewer complains it flashes past.
SPEED="${SPEED:-3}"
COLS="${COLS:-100}"
ROWS="${ROWS:-30}"
FONT_SIZE="${FONT_SIZE:-16}"
# agg re-renders the terminal at whatever height you ask, so the recorded ROWS can stay
# generous while the GIF is cropped tight. A 30-row frame is three quarters black: the
# session scrolls, it never fills the screen.
GIF_ROWS="${GIF_ROWS:-22}"

# The question to record. Asking for statistics rather than "mark the shock arrival" is
# deliberate: a request to *detect* something makes the agent verify its own detection,
# which costs three or four extra run_python calls and a long recording. The shock is a
# 10 -> 25 nT step, so it is unmissable on the plot either way. The number of tool calls
# is a property of the question, not something to re-roll for.
# It exercises the whole chain and still terminates quickly:
#   search_parameters (the hybrid RAG — the actual differentiator)
#   → get_timeseries (a real download)
#   → run_python (sandboxed plotting)
#   → the provenance verdict on the way out
# Keep it to one clear ask. A multi-step investigation makes a long, boring recording.
QUERY="${QUERY:-Plot the Wind magnetic field magnitude at 3 s cadence for 2015-03-17 03:30 to 05:00, and report the mean, minimum and maximum |B|}"

need() {
    command -v "$1" >/dev/null 2>&1 && return 0
    echo "Missing: $1" >&2
    echo "  $2" >&2
    return 1
}

convert() {
    [ -f "$CAST" ] || { echo "No recording at $CAST — run without --convert-only first." >&2; exit 1; }
    # NOT `cargo install agg`: the crates.io crate of that name is an unrelated library and
    # the install fails. agg ships prebuilt binaries.
    need agg "download from https://github.com/asciinema/agg/releases into ~/.local/bin" || exit 1
    agg --speed "$SPEED" --rows "$GIF_ROWS" --font-size "$FONT_SIZE" \
        --theme asciinema "$CAST" "$GIF"
    echo
    echo "GIF:  $GIF  ($(du -h "$GIF" | cut -f1))"
    echo "Cast: $CAST ($(du -h "$CAST" | cut -f1))"
    echo
    echo "Re-render without re-running the agent — the cast is kept for exactly this:"
    echo "  SPEED=2 $0 --convert-only        # slower, easier to read"
    echo "  GIF_ROWS=30 $0 --convert-only    # taller frame"
    echo
    echo "agg caps idle time at 5 s by default, so a 160 s session is already a ~20 s GIF;"
    echo "raising SPEED trims the text on screen, not the waiting. GitHub renders a GIF"
    echo "over ~10 MB slowly or not at all."
}

if [ "${1:-}" = "--convert-only" ]; then
    convert
    exit 0
fi

need asciinema "uv tool install asciinema  (or: apt install asciinema)" || exit 1
mkdir -p docs/assets

HELIOAI_BIN="$(command -v helioai || echo .venv/bin/helioai)"
[ -x "$HELIOAI_BIN" ] || { echo "helioai not found — is the venv built?" >&2; exit 1; }

# Quieten what is noise on screen but not part of the reasoning chain. Neither hides a
# failure: HELIOAI_LOG_LEVEL only lifts the threshold above speasy's warnings about a
# provider it goes on to disable by itself, and TQDM_DISABLE drops the embedding model's
# weight-loading bar. Errors and every agent event still print.
export HELIOAI_LOG_LEVEL="${HELIOAI_LOG_LEVEL:-ERROR}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

echo "Recording a real session. Question:"
echo "  $QUERY"
echo
echo "Let it run to the end, including the provenance line — that is the payoff shot."
echo

# --overwrite so a retake does not prompt; the cast is disposable, the GIF is the artefact.
asciinema rec "$CAST" \
    --overwrite \
    --cols "$COLS" --rows "$ROWS" \
    --title "HelioAI — from a question to a plot" \
    --command "$HELIOAI_BIN \"$QUERY\""

convert
